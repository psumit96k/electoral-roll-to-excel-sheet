"""
Export API endpoints for Excel generation.
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
import os

from app.config import settings
from app.models.database import (
    Document, Batch, ReviewItem, CardSlot, Page,
    ReviewStatusEnum
)
from app.database import get_db
from app.export.xlsx_generator import generate_excel_export

router = APIRouter()


class ExportRequest(BaseModel):
    document_ids: Optional[List[int]] = None
    batch_ids: Optional[List[int]] = None
    include_unconfirmed: bool = False  # Whether to include unreviewed items
    export_filename: Optional[str] = None


class ExportResponse(BaseModel):
    message: str
    export_id: int
    filename: str
    record_count: int
    created_at: datetime
    download_url: str


@router.post("/")
async def create_export(
    export_request: ExportRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Create an Excel export from reviewed data.
    """
    # Determine which documents to export
    document_ids = set()

    if export_request.document_ids:
        document_ids.update(export_request.document_ids)

    if export_request.batch_ids:
        batch_docs = db.query(Document.id).filter(
            Document.batch_id.in_(export_request.batch_ids)
        ).all()
        document_ids.update([doc.id for doc in batch_docs])

    # If neither specified, export all documents
    if not document_ids:
        all_docs = db.query(Document.id).all()
        document_ids.update([doc.id for doc in all_docs])

    # Validate that documents exist
    documents = db.query(Document).filter(Document.id.in_(document_ids)).all()
    if len(documents) != len(document_ids):
        raise HTTPException(
            status_code=400,
            detail="One or more specified documents not found"
        )

    # Reject exports from documents still processing or failed
    for doc in documents:
        if doc.processing_status in [ProcessingStatusEnum.PROCESSING.value, ProcessingStatusEnum.FAILED.value]:
            raise HTTPException(
                status_code=400,
                detail=f"Document {doc.original_filename} is {doc.processing_status.lower()} and cannot be exported"
            )

    # Check export readiness unless including unconfirmed
    if not export_request.include_unconfirmed:
        for doc in documents:
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
                raise HTTPException(
                    status_code=400,
                    detail=f"Document {doc.original_filename} has {unreviewed_count} unreviewed items. "
                           "Set include_unconfirmed=True to export anyway, or complete review first."
                )

    # Generate export filename
    if not export_request.export_filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_filename = f"election_export_{timestamp}.xlsx"
    else:
        export_filename = export_request.export_filename
        if not export_filename.endswith('.xlsx'):
            export_filename += '.xlsx'

    # Validate export filename
    # Security: validate filename to prevent path traversal
    if ".." in export_filename or "/" in export_filename or "\\" in export_filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Additional validation: only allow alphanumeric, underscore, hyphen, and dot
    import re
    if not re.match(r'^[a-zA-Z0-9_.\-]+$', export_filename):
        raise HTTPException(status_code=400, detail="Filename contains invalid characters")

    # Ensure export directory exists
    settings.EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    file_path = settings.EXPORT_DIR / export_filename

    # Create export history record (simplified for now)
    # In a full implementation, this would be a database record

    # Generate Excel file in background
    background_tasks.add_task(
        generate_excel_export,
        document_ids=list(document_ids),
        output_path=str(file_path),
        include_unconfirmed=export_request.include_unconfirmed
        # Let the background export create its own database session
    )

    return ExportResponse(
        message="Export generation started",
        export_id=0,  # Would be from database record in full implementation
        filename=export_filename,
        record_count=0,  # Will be updated when generation completes
        created_at=datetime.now(),
        download_url=f"/api/v1/export/download/{export_filename}"
    )


@router.get("/download/{filename}")
async def download_export(
    filename: str,
    db: Session = Depends(get_db)
):
    """
    Download a generated Excel export.
    """
    # Security: validate filename to prevent path traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = settings.EXPORT_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Export file not found")

    # Verify it's an Excel file
    if not filename.endswith('.xlsx'):
        raise HTTPException(status_code=400, detail="Invalid file type")

    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


@router.get("/history")
async def get_export_history(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """
    Get export history.
    """
    # Placeholder - would query export history table in full implementation
    return {
        "message": "Export history not yet implemented",
        "exports": []
    }


@router.delete("/{filename}")
async def delete_export(
    filename: str,
    db: Session = Depends(get_db)
):
    """
    Delete an exported file.
    """
    # Security: validate filename
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = settings.EXPORT_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Export file not found")

    try:
        file_path.unlink()
        return {"message": f"Export {filename} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")