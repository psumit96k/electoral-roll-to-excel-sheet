"""
Background job processor for PDF extraction and processing.
"""
import logging
import re
import sys
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_
import json
import numpy as np
from pathlib import Path

from app.config.settings import settings
from app.models.database import (
    Document, Page, CardSlot, Job, ExtractionCandidate,
    ReviewItem, ProcessingStatusEnum, ReviewStatusEnum
)
from app.ocr.pdf_handler import pdf_handler
from app.ocr.service import ocr_service
from app.parsing.geometry import geometry_parser
from app.parsing.field_parser import FieldParser
from app.database import SessionLocal
from app.export.xlsx_generator import generate_excel_export

logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    Process PDF documents through the complete pipeline:
    PDF -> Images -> OCR -> Geometry matching -> Field extraction -> Storage
    """

    def __init__(self, db_session: Optional[Session] = None):
        """
        Initialize the PDF processor.

        Args:
            db_session: Database session (will create one if not provided)
        """
        self.db_session = db_session or SessionLocal()
        self.close_session = db_session is None
        self.field_parser = FieldParser()

    def __del__(self):
        """Clean up database session."""
        if hasattr(self, 'close_session') and self.close_session:
            self.db_session.close()

    def process_document(self, document_id: int, pages: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Process a single document through the complete pipeline.

        Args:
            document_id: ID of the document to process
            pages: Optional list of page numbers to process (1-based). If None, process all pages.

        Returns:
            Dictionary with processing results
        """
        logger.info(f"Starting processing for document ID: {document_id}")

        try:
            # Get document record
            document = self.db_session.query(Document).filter(
                Document.id == document_id
            ).first()

            if not document:
                raise ValueError(f"Document {document_id} not found")

            # Update document status to processing
            document.processing_status = ProcessingStatusEnum.PROCESSING.value
            self.db_session.commit()

            # Create or get job record
            job = self.db_session.query(Job).filter(
                and_(
                    Job.document_id == document_id,
                    Job.job_type == "full_processing"
                )
            ).order_by(Job.id.desc()).first()

            if not job:
                job = Job(
                    document_id=document_id,
                    job_type="full_processing",
                    status=ProcessingStatusEnum.PROCESSING.value
                )
                self.db_session.add(job)
                self.db_session.commit()

            # Update job progress
            job.status = ProcessingStatusEnum.PROCESSING.value
            job.progress_percent = 10
            job.started_at = datetime.now()
            self.db_session.commit()

            # Step 1: Convert PDF to images
            logger.info("Converting PDF to images")
            images = pdf_handler.pdf_to_images(document.file_path)
            job.progress_percent = 20
            self.db_session.commit()

            # Process each page
            total_pages = len(images)
            processed_pages = 0
            total_slots = 0
            occupied_slots = 0

            # Determine which pages to process
            if pages is None:
                page_indices = range(total_pages)
            else:
                # Check for empty list
                if len(pages) == 0:
                    raise ValueError("Page list cannot be empty")

                # Convert to 0-based indices and validate all pages
                page_indices = []
                for p in pages:
                    # Validate that p is an integer (excluding bool) and within valid range
                    if not isinstance(p, int) or isinstance(p, bool):
                        raise ValueError(f"Page number must be an integer, got {type(p).__name__}: {p}")
                    if p < 1 or p > total_pages:
                        raise ValueError(f"Page number {p} is out of range for document with {total_pages} pages")
                    page_indices.append(p - 1)  # Convert to 0-based index

                # Check for duplicates and deduplicate while preserving order
                seen = set()
                deduped_indices = []
                for idx in page_indices:
                    if idx not in seen:
                        seen.add(idx)
                        deduped_indices.append(idx)
                page_indices = deduped_indices

            for idx in page_indices:
                page_num = idx + 1  # Convert back to 1-based for logging
                _, image = images[idx]  # Unpack the tuple (page_number, image)
                logger.info(f"Processing page {page_num}/{total_pages}")

                # Get or create page record
                page = self.db_session.query(Page).filter(
                    and_(
                        Page.document_id == document_id,
                        Page.page_number == page_num
                    )
                ).first()

                if not page:
                    # Get image dimensions
                    height, width = image.shape[:2]

                    page = Page(
                        document_id=document_id,
                        page_number=page_num,
                        width=width,
                        height=height,
                        processing_status=ProcessingStatusEnum.PROCESSING.value
                    )
                    self.db_session.add(page)
                    self.db_session.commit()

                # Update job progress
                progress = 20 + int((len([p for p in page_indices if p <= idx]) / len(page_indices)) * 60) if page_indices else 20
                job.progress_percent = progress
                self.db_session.commit()

                # Process page: get geometry and process slots
                page_result = self._process_page(page, image)
                processed_pages += 1
                total_slots += page_result["total_slots"]
                occupied_slots += page_result["occupied_slots"]

                # Update page status
                page.processing_status = ProcessingStatusEnum.COMPLETED.value
                self.db_session.commit()

            # Finalize job
            job.status = ProcessingStatusEnum.COMPLETED.value
            job.progress_percent = 100
            job.completed_at = datetime.now()
            self.db_session.commit()

            # Update document status
            document.processing_status = ProcessingStatusEnum.COMPLETED.value
            document.page_count = total_pages
            self.db_session.commit()

            logger.info(f"Document processing completed: {document_id}")

            # Automatically export extracted voter records to Excel.
            try:
                export_filename = f"{Path(document.original_filename).stem}_export.xlsx"
                export_path = settings.EXPORT_DIR / export_filename
                settings.EXPORT_DIR.mkdir(parents=True, exist_ok=True)

                export_result = generate_excel_export(
                    document_ids=[document_id],
                    output_path=str(export_path),
                    include_unconfirmed=True
                )

                if export_result.get("success"):
                    logger.info(
                        f"Automatic Excel export created: {export_path} "
                        f"({export_result.get('record_count', 0)} records)"
                    )
                else:
                    logger.error(
                        f"Automatic Excel export failed for document "
                        f"{document_id}: {export_result.get('error', 'Unknown error')}"
                    )
            except Exception:
                logger.exception(
                    f"Automatic Excel export failed for document {document_id}"
                )

            return {
                "success": True,
                "document_id": document_id,
                "pages_processed": processed_pages,
                "total_slots": total_slots,
                "occupied_slots": occupied_slots,
                "message": f"Processed {processed_pages} pages with {occupied_slots} occupied slots out of {total_slots}"
            }

        except Exception as e:
            logger.error(f"Error processing document {document_id}: {e}")

            # Roll back the active failed transaction before writing failure status
            self.db_session.rollback()

            # Update document and job status to failed
            if 'document' in locals():
                document.processing_status = ProcessingStatusEnum.FAILED.value
                self.db_session.commit()

            if 'job' in locals():
                job.status = ProcessingStatusEnum.FAILED.value
                job.error_message = str(e)
                job.completed_at = datetime.now()
                self.db_session.commit()

            return {
                "success": False,
                "document_id": document_id,
                "error": str(e),
                "message": f"Processing failed: {str(e)}"
            }

    def _process_page(self, page: Page, image: np.ndarray) -> Dict[str, Any]:
        """
        Process a single page: detect geometry, extract OCR, parse fields.

        Args:
            page: Page database record
            image: Image as numpy array

        Returns:
            Dictionary with processing results
        """
        logger.debug(f"Processing page {page.page_number}")

        # Get image dimensions
        height, width = image.shape[:2]

        # Validate geometry match
        geometry_valid = geometry_parser.validate_geometry_match(width, height)
        logger.debug(f"Geometry validation: {geometry_valid}")

        # Get all card slot bounds
        slots = geometry_parser.get_all_card_slots(width, height)

        total_slots = len(slots)
        occupied_slots = 0

        # Process each slot
        for slot_info in slots:
            slot_number = slot_info["slot_number"]
            bounds = slot_info["bounds"]

            # Get or create card slot record
            card_slot = self.db_session.query(CardSlot).filter(
                and_(
                    CardSlot.page_id == page.id,
                    CardSlot.physical_slot == slot_number
                )
            ).first()

            if not card_slot:
                card_slot = CardSlot(
                    page_id=page.id,
                    physical_slot=slot_number,
                    row_number=slot_info["row"],
                    column_number=slot_info["column"],
                    left=bounds[0],
                    top=bounds[1],
                    right=bounds[2],
                    bottom=bounds[3],
                    is_occupied=False  # Will be updated based on OCR results
                )
                self.db_session.add(card_slot)
                self.db_session.flush()  # Get ID without committing

            # Crop the card region from the image
            left, top, right, bottom = bounds
            card_image = image[top:bottom, left:right]

            # Extract OCR from the cropped card image
            # OCR exceptions are propagated up to fail the entire document processing
            # This prevents treating OCR failures as empty slots
            ocr_results = ocr_service.extract_text_from_image(card_image)

            # Determine if slot is occupied based on OCR results
            is_occupied = len(ocr_results) > 0 and any(
                result.get("confidence", 0) > 0.3 for result in ocr_results  # Low confidence threshold
            )

            # Update card slot
            card_slot.is_occupied = is_occupied
            if ocr_results:
                # Average confidence of all detections
                valid_results = [r for r in ocr_results if r.get("confidence") is not None]
                if valid_results:
                    avg_confidence = sum(r["confidence"] for r in valid_results) / len(valid_results)
                else:
                    avg_confidence = 0.1
            else:
                avg_confidence = 0.0
            card_slot.confidence = avg_confidence

            self.db_session.flush()

            if is_occupied:
                occupied_slots += 1

                # Create extraction candidate
                self._create_extraction_card(card_slot, ocr_results, card_image)

            # Commit every few slots to avoid large transactions
            if slot_number % 5 == 0:
                self.db_session.commit()

        # Final commit for this page
        self.db_session.commit()

        return {
            "total_slots": total_slots,
            "occupied_slots": occupied_slots
        }

    def _create_extraction_card(
        self,
        card_slot: CardSlot,
        ocr_results: List[Dict[str, Any]],
        card_image: np.ndarray
    ):
        """
        Create extraction candidate from OCR results for a card slot.

        Args:
            card_slot: CardSlot database record
            ocr_results: List of OCR results from the cropped card image
            card_image: Cropped card image as numpy array
        """
        logger.debug(f"Creating extraction candidate for slot {card_slot.physical_slot}")

        # Handle missing confidence scores safely
        valid_results = [r for r in ocr_results if r.get("confidence") is not None]
        if not valid_results:
            # If no valid confidence scores, use a default low value
            avg_confidence = 0.1
        else:
            avg_confidence = sum(r["confidence"] for r in valid_results) / len(valid_results)

        # Get card dimensions
        card_height, card_width = card_image.shape[:2]

        # Check if we already have extraction candidates for this slot to avoid duplication
        existing_extraction = self.db_session.query(ExtractionCandidate).filter(
            ExtractionCandidate.card_slot_id == card_slot.id
        ).first()

        # Check if we already have review items for this slot to avoid overwriting manual corrections
        existing_review_item = self.db_session.query(ReviewItem).filter(
            ReviewItem.card_slot_id == card_slot.id
        ).first()

        # Parse the detections using FieldParser
        try:
            parsed_fields = self.field_parser.parse_voter_card_detections(
                ocr_results,
                card_width=card_width,
                card_height=card_height
            )
        except Exception as e:
            logger.warning(f"Field parsing failed for slot {card_slot.physical_slot}: {e}")
            # Create empty parsed fields if parsing fails
            parsed_fields = {
                'epic': None,
                'name': None,
                'relation_type': None,
                'relation_name': None,
                'house_number': None,
                'age': None,
                'gender': None,
                'serial_number': None,
                '_unassigned': [],
                '_conflicts': {}
            }

        # Always use existing extraction candidate if one exists
        if not existing_extraction:
            # Create new extraction candidate only if one doesn't exist
            extraction_candidate = ExtractionCandidate(
                card_slot_id=card_slot.id,
                raw_full_name=parsed_fields.get('name'),
                raw_relation_type=parsed_fields.get('relation_type'),
                raw_relative_name=parsed_fields.get('relation_name'),
                raw_house_number=parsed_fields.get('house_number'),
                raw_age=parsed_fields.get('age'),
                raw_gender=parsed_fields.get('gender'),
                raw_epic=parsed_fields.get('epic'),
                normalized_epic=self.field_parser.normalize_epic(parsed_fields.get('epic')) if parsed_fields.get('epic') else None,
                parsed_age=self.field_parser.parse_age(str(parsed_fields.get('age'))) if parsed_fields.get('age') is not None else None,
                ocr_confidence=avg_confidence,
                ocr_model_version=settings.OCR_MODEL_VERSION if hasattr(settings, 'OCR_MODEL_VERSION') else "PaddleOCR_v2.7"
            )

            self.db_session.add(extraction_candidate)
            self.db_session.flush()  # Get the ID without committing
        else:
            # Use existing extraction candidate
            extraction_candidate = existing_extraction

        # Update extraction candidate fields from OCR data only if review item is NOT reviewed/confirmed
        # This prevents overwriting manually corrected data
        should_update_from_ocr = not (
            existing_review_item and
            existing_review_item.review_status in [ReviewStatusEnum.REVIEWED.value, ReviewStatusEnum.CONFIRMED.value]
        )

        if should_update_from_ocr:
            extraction_candidate.raw_full_name = parsed_fields.get('name')
            extraction_candidate.raw_relation_type = parsed_fields.get('relation_type')
            extraction_candidate.raw_relative_name = parsed_fields.get('relation_name')
            extraction_candidate.raw_house_number = parsed_fields.get('house_number')
            extraction_candidate.raw_age = parsed_fields.get('age')
            extraction_candidate.raw_gender = parsed_fields.get('gender')
            extraction_candidate.raw_epic = parsed_fields.get('epic')
            extraction_candidate.normalized_epic = self.field_parser.normalize_epic(parsed_fields.get('epic')) if parsed_fields.get('epic') else None
            extraction_candidate.parsed_age = self.field_parser.parse_age(str(parsed_fields.get('age'))) if parsed_fields.get('age') is not None else None

        # Create or update review item
        if not existing_review_item or existing_review_item.review_status not in [ReviewStatusEnum.REVIEWED.value, ReviewStatusEnum.CONFIRMED.value]:
            # Only update review item if it doesn't exist or hasn't been manually reviewed/confirmed
            review_item = ReviewItem(
                card_slot_id=card_slot.id,
                extraction_candidate_id=extraction_candidate.id,
                review_status=ReviewStatusEnum.UNREVIEWED.value,
                reviewed_full_name=parsed_fields.get('name'),
                reviewed_relation_type=parsed_fields.get('relation_type'),
                reviewed_relative_name=parsed_fields.get('relation_name'),
                reviewed_house_number=parsed_fields.get('house_number'),
                reviewed_age=parsed_fields.get('age'),
                reviewed_gender=parsed_fields.get('gender'),
                reviewed_epic=parsed_fields.get('epic')
            )

            if not existing_review_item:
                self.db_session.add(review_item)
            else:
                # Update existing review item with new extracted data (but preserve manual corrections)
                # Only update fields that are None in the existing review item (indicating no manual correction)
                if existing_review_item.reviewed_full_name is None:
                    existing_review_item.reviewed_full_name = parsed_fields.get('name')
                if existing_review_item.reviewed_relation_type is None:
                    existing_review_item.reviewed_relation_type = parsed_fields.get('relation_type')
                if existing_review_item.reviewed_relative_name is None:
                    existing_review_item.reviewed_relative_name = parsed_fields.get('relation_name')
                if existing_review_item.reviewed_house_number is None:
                    existing_review_item.reviewed_house_number = parsed_fields.get('house_number')
                if existing_review_item.reviewed_age is None:
                    existing_review_item.reviewed_age = parsed_fields.get('age')
                if existing_review_item.reviewed_gender is None:
                    existing_review_item.reviewed_gender = parsed_fields.get('gender')
                if existing_review_item.reviewed_epic is None:
                    existing_review_item.reviewed_epic = parsed_fields.get('epic')

                existing_review_item.updated_at = datetime.utcnow()
                review_item = existing_review_item
        else:
            # Use existing review item
            review_item = existing_review_item

        # Preserve diagnostics in a per-document sidecar JSON
        self._save_slot_diagnostics(
            card_slot,
            parsed_fields,
            ocr_results,
            avg_confidence
        )

    def _save_slot_diagnostics(
        self,
        card_slot: CardSlot,
        parsed_fields: Dict[str, Any],
        ocr_results: List[Dict[str, Any]],
        avg_confidence: float
    ):
        """
        Save diagnostics for a slot to a sidecar JSON file.

        Args:
            card_slot: CardSlot database record
            parsed_fields: Parsed fields from FieldParser
            ocr_results: OCR results used for parsing
            avg_confidence: Average confidence of OCR results
        """
        try:
            # Create diagnostics directory if it doesn't exist
            diagnostics_dir = Path(settings.BASE_DIR) / "diagnostics"
            diagnostics_dir.mkdir(parents=True, exist_ok=True)

            # Create a unique filename based on document and slot info
            document_id = card_slot.page.document_id if card_slot.page else 0
            page_number = card_slot.page.page_number if card_slot.page else 0
            slot_number = card_slot.physical_slot

            diagnostics_file = diagnostics_dir / f"doc{document_id}_page{page_number}_slot{slot_number}_diagnostics.json"

            # Prepare diagnostics data
            diagnostics_data = {
                "timestamp": datetime.now().isoformat(),
                "slot_info": {
                    "document_id": document_id,
                    "page_number": page_number,
                    "slot_number": slot_number,
                    "bounds": {
                        "left": card_slot.left,
                        "top": card_slot.top,
                        "right": card_slot.right,
                        "bottom": card_slot.bottom
                    }
                },
                "ocr_results": [
                    {
                        "text": result.get("text"),
                        "confidence": result.get("confidence"),
                        "bbox": result.get("bbox")
                    }
                    for result in ocr_results
                ],
                "parsed_fields": parsed_fields,
                "average_confidence": avg_confidence,
                "ocr_model_version": settings.OCR_MODEL_VERSION if hasattr(settings, 'OCR_MODEL_VERSION') else "PaddleOCR_v2.7"
            }

            # Write diagnostics to file
            with open(diagnostics_file, 'w', encoding='utf-8') as f:
                json.dump(diagnostics_data, f, indent=2, default=str)

            logger.debug(f"Saved diagnostics for slot {slot_number} to {diagnostics_file}")

        except Exception as e:
            logger.warning(f"Failed to save diagnostics for slot {card_slot.physical_slot}: {e}")


def process_document_background(document_id: int, pages: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Background task function for processing a document.

    Args:
        document_id: ID of the document to process
        pages: Optional list of page numbers to process (1-based). If None, process all pages.

    Returns:
        Dictionary with processing results
    """
    processor = PDFProcessor()
    return processor.process_document(document_id, pages)


def process_multiple_documents_background(document_ids: List[int]) -> Dict[str, Any]:
    """
    Background task function for processing multiple documents.

    Args:
        document_ids: List of document IDs to process

    Returns:
        Dictionary with processing results
    """
    results = {
        "success": True,
        "documents_processed": 0,
        "documents_failed": 0,
        "results": []
    }

    for doc_id in document_ids:
        logger.info(f"Processing document {doc_id}")
        result = process_document_background(doc_id)
        results["results"].append({
            "document_id": doc_id,
            "result": result
        })

        if result["success"]:
            results["documents_processed"] += 1
        else:
            results["documents_failed"] += 1
            # Continue processing other documents even if one fails

    if results["documents_failed"] > 0:
        results["success"] = False

    return results


if __name__ == "__main__":
    """Simple CLI runner for processing documents."""
    import argparse

    parser = argparse.ArgumentParser(description='Process PDF documents through the OCR pipeline.')
    parser.add_argument('--document-id', type=int, required=True, help='ID of the document to process')
    parser.add_argument('--pages', type=str, help='Comma-separated list of page numbers to process (e.g., "3,4,5" or "3-26")')

    args = parser.parse_args()

    # Parse pages argument
    pages = None
    if args.pages:
        try:
            # Handle ranges like "3-26"
            if '-' in args.pages:
                start, end = map(int, args.pages.split('-'))
                pages = list(range(start, end + 1))
            else:
                # Handle comma-separated list
                pages = [int(p.strip()) for p in args.pages.split(',')]
        except ValueError:
            print("Error: Invalid page format. Use comma-separated numbers or range (e.g., '3-26')")
            sys.exit(1)

    # Process the document
    result = process_document_background(args.document_id, pages)

    if result["success"]:
        print(f"Success: {result['message']}")
    else:
        print(f"Error: {result['message']}")
        sys.exit(1)
