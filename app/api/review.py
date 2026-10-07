"""
Review API endpoints for manual verification and correction.
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

from app.config import settings
from app.models.database import (
    ReviewItem, Document, Page, CardSlot, ExtractionCandidate,
    ReviewStatusEnum, ProcessingStatusEnum, Batch
)
from app.database import get_db

router = APIRouter()


# Pydantic models for request/response
class ReviewItemBase(BaseModel):
    reviewed_full_name: Optional[str] = None
    reviewed_relation_type: Optional[str] = None
    reviewed_relative_name: Optional[str] = None
    reviewed_house_number: Optional[str] = None
    reviewed_age: Optional[int] = None
    reviewed_gender: Optional[str] = None  # MALE, FEMALE, OTHER or empty
    reviewed_caste: Optional[str] = None
    reviewed_phone_number: Optional[str] = None
    reviewed_epic: Optional[str] = None
    review_status: Optional[str] = None
    confirmation_notes: Optional[str] = None
    exclusion_reason: Optional[str] = None


class ReviewItemCreate(ReviewItemBase):
    pass


class ReviewItemUpdate(ReviewItemBase):
    pass


class ReviewItemResponse(ReviewItemBase):
    id: int
    card_slot_id: int
    extraction_candidate_id: int
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


class CardSlotResponse(BaseModel):
    id: int
    physical_slot: int
    row_number: int
    column_number: int
    is_occupied: bool
    confidence: Optional[float] = None
    review_items: List[ReviewItemResponse] = []

    class Config:
        orm_mode = True


class PageResponse(BaseModel):
    id: int
    page_number: int
    width: int
    height: int
    processing_status: str
    card_slots: List[CardSlotResponse] = []

    class Config:
        orm_mode = True


@router.get("/document/{document_id}")
async def get_document_for_review(
    document_id: int,
    db: Session = Depends(get_db)
):
    """
    Get document with pages and card slots for review interface.
    """
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # Get pages with card slots and review items
    pages = db.query(Page).filter(Page.document_id == document_id).all()

    # Build response structure
    result = {
        "document_id": document.id,
        "filename": document.original_filename,
        "pages": []
    }

    for page in pages:
        # Get card slots for this page
        card_slots = db.query(CardSlot).filter(CardSlot.page_id == page.id).all()

        page_data = {
            "id": page.id,
            "page_number": page.page_number,
            "width": page.width,
            "height": page.height,
            "processing_status": page.processing_status,
            "card_slots": []
        }

        for slot in card_slots:
            # Get review items for this card slot
            review_items = db.query(ReviewItem).filter(
                ReviewItem.card_slot_id == slot.id
            ).all()

            slot_data = {
                "id": slot.id,
                "physical_slot": slot.physical_slot,
                "row_number": slot.row_number,
                "column_number": slot.column_number,
                "is_occupied": slot.is_occupied,
                "confidence": slot.confidence,
                "review_items": []
            }

            for item in review_items:
                item_data = {
                    "id": item.id,
                    "reviewed_full_name": item.reviewed_full_name,
                    "reviewed_relation_type": item.reviewed_relation_type,
                    "reviewed_relative_name": item.reviewed_relative_name,
                    "reviewed_house_number": item.reviewed_house_number,
                    "reviewed_age": item.reviewed_age,
                    "reviewed_gender": item.reviewed_gender,
                    "reviewed_caste": item.reviewed_caste,
                    "reviewed_phone_number": item.reviewed_phone_number,
                    "reviewed_epic": item.reviewed_epic,
                    "review_status": item.review_status,
                    "reviewed_by": item.reviewed_by,
                    "reviewed_at": item.reviewed_at,
                    "created_at": item.created_at,
                    "updated_at": item.updated_at,
                    "confirmation_notes": item.confirmation_notes,
                    "exclusion_reason": item.exclusion_reason
                }
                slot_data["review_items"].append(item_data)

            page_data["card_slots"].append(slot_data)

        result["pages"].append(page_data)

    return result


@router.put("/item/{item_id}")
async def update_review_item(
    item_id: int,
    item_update: ReviewItemUpdate,
    db: Session = Depends(get_db)
):
    """
    Update a review item (manual correction).
    """
    review_item = db.query(ReviewItem).filter(ReviewItem.id == item_id).first()
    if not review_item:
        raise HTTPException(status_code=404, detail="Review item not found")

    # Update fields
    update_data = item_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(review_item, field, value)

    review_item.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(review_item)

    return review_item


@router.post("/item/{item_id}/confirm")
async def confirm_review_item(
    item_id: int,
    db: Session = Depends(get_db)
):
    """
    Explicitly confirm a review item.
    """
    review_item = db.query(ReviewItem).filter(ReviewItem.id == item_id).first()
    if not review_item:
        raise HTTPException(status_code=404, detail="Review item not found")

    review_item.review_status = ReviewStatusEnum.CONFIRMED.value
    review_item.reviewed_at = datetime.utcnow()
    review_item.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(review_item)

    return {"message": "Review item confirmed", "review_item": review_item}


@router.post("/item/{item_id}/exclude")
async def exclude_review_item(
    item_id: int,
    exclusion_reason: str,
    db: Session = Depends(get_db)
):
    """
    Exclude a review item from export.
    """
    review_item = db.query(ReviewItem).filter(ReviewItem.id == item_id).first()
    if not review_item:
        raise HTTPException(status_code=404, detail="Review item not found")

    review_item.review_status = ReviewStatusEnum.EXCLUDED.value
    review_item.exclusion_reason = exclusion_reason
    review_item.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(review_item)

    return {"message": "Review item excluded", "review_item": review_item}


@router.get("/document/{document_id}/stats")
async def get_document_review_stats(
    document_id: int,
    db: Session = Depends(get_db)
):
    """
    Get review statistics for a document.
    """
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # Get all review items for this document through joins
    from sqlalchemy import func

    stats = db.query(
        ReviewItem.review_status,
        func.count(ReviewItem.id).label('count')
    ).join(
        CardSlot, ReviewItem.card_slot_id == CardSlot.id
    ).join(
        Page, CardSlot.page_id == Page.id
    ).filter(
        Page.document_id == document_id
    ).group_by(
        ReviewItem.review_status
    ).all()

    # Format results
    stats_dict = {status: 0 for status in ReviewStatusEnum}
    for status, count in stats:
        stats_dict[status] = count

    total_reviewed = sum(stats_dict.values())

    return {
        "document_id": document_id,
        "total_items": total_reviewed,
        "review_stats": stats_dict,
        "progress_percentage": round(
            (stats_dict[ReviewStatusEnum.CONFIRMED.value] +
             stats_dict[ReviewStatusEnum.REVIEWED.value]) /
            max(total_reviewed, 1) * 100, 2
        ) if total_reviewed > 0 else 0
    }


@router.get("/batch/{batch_id}/export-ready")
async def check_batch_export_ready(
    batch_id: int,
    db: Session = Depends(get_db)
):
    """
    Check if all documents in a batch are ready for export.
    """
    batch = db.query(Batch).filter(Batch.id == batch_id).first()
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    # Get all documents in batch
    documents = db.query(Document).filter(Document.batch_id == batch_id).all()

    # Check each document's review status
    not_ready_docs = []
    for doc in documents:
        # Get unreviewed or flagged items for this document
        unreviewed_count = db.query(ReviewItem).join(
            CardSlot, ReviewItem.card_slot_id == CardSlot.id
        ).join(
            Page, CardSlot.page_id == Page.id
        ).filter(
            Page.document_id == doc.id,
            ReviewItem.review_status.in_([
                ReviewStatusEnum.UNREVIEWED.value,
                ReviewStatusEnum.FLAGGED.value
            ])
        ).count()

        if unreviewed_count > 0:
            not_ready_docs.append({
                "document_id": doc.id,
                "filename": doc.original_filename,
                "unreviewed_items": unreviewed_count
            })

    is_ready = len(not_ready_docs) == 0

    return {
        "batch_id": batch_id,
        "batch_name": batch.name,
        "is_export_ready": is_ready,
        "not_ready_documents": not_ready_docs,
        "total_documents": len(documents)
    }