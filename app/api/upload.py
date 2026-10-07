"""
Upload API endpoints for PDF files.
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List, Optional
import os
import shutil
from pathlib import Path
import hashlib
import uuid

from app.config import settings
from app.models.database import Batch, Document
from app.database import get_db

router = APIRouter()


@router.post("/")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    batch_name: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Upload a PDF file for processing.
    """
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # Validate file size
    contents = await file.read()
    file_size = len(contents)

    max_size_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if file_size > max_size_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds maximum allowed size of {settings.MAX_FILE_SIZE_MB}MB"
        )

    # Calculate file hash
    file_hash = hashlib.sha256(contents).hexdigest()

    # Check for duplicate uploads
    existing_doc = db.query(Document).filter(Document.file_hash == file_hash).first()
    if existing_doc:
        return JSONResponse(
            status_code=200,
            content={
                "message": "File already uploaded",
                "document_id": existing_doc.id,
                "filename": existing_doc.filename,
                "is_duplicate": True
            }
        )

    # Create batch if not provided
    if not batch_name:
        batch_name = f"Batch_{uuid.uuid4().hex[:8]}"

    batch = db.query(Batch).filter(Batch.name == batch_name).first()
    if not batch:
        batch = Batch(name=batch_name)
        db.add(batch)
        db.commit()
        db.refresh(batch)

    # Save file
    upload_dir = settings.UPLOAD_DIR
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique filename
    file_extension = Path(file.filename).suffix
    unique_filename = f"{uuid.uuid4().hex}{file_extension}"
    file_path = upload_dir / unique_filename

    # Write file
    with open(file_path, "wb") as buffer:
        buffer.write(contents)

    # Create document record
    document = Document(
        batch_id=batch.id,
        filename=unique_filename,
        original_filename=file.filename,
        file_path=str(file_path),
        file_hash=file_hash,
        file_size=file_size,
        page_count=0  # Will be updated after PDF processing
    )

    db.add(document)
    db.commit()
    db.refresh(document)

    # TODO: Add background task for PDF processing (page count extraction, etc.)
    # background_tasks.add_task(process_pdf_metadata, document.id)

    return {
        "message": "File uploaded successfully",
        "document_id": document.id,
        "filename": document.filename,
        "original_filename": document.original_filename,
        "file_size": document.file_size,
        "batch_id": batch.id,
        "batch_name": batch.name,
        "is_duplicate": False
    }


@router.post("/batch")
async def upload_multiple_pdfs(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    batch_name: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Upload multiple PDF files as a batch.
    """
    if len(files) > settings.MAX_BATCH_PDFS:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {settings.MAX_BATCH_PDFS} files allowed per batch"
        )

    results = []
    for file in files:
        # Process each file (reuse single upload logic)
        result = await upload_pdf(background_tasks, file, batch_name, db)
        results.append(result)

    return {
        "message": f"Uploaded {len(files)} files",
        "results": results
    }


@router.get("/")
async def list_uploads(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    List uploaded documents.
    """
    documents = db.query(Document).offset(skip).limit(limit).all()
    return documents


@router.get("/{document_id}")
async def get_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """
    Get document details by ID.
    """
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@router.delete("/{document_id}")
async def delete_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """
    Delete a document and its associated file.
    """
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # Delete physical file
    try:
        if os.path.exists(document.file_path):
            os.remove(document.file_path)
    except OSError as e:
        # Log error but continue with database deletion
        pass

    # Delete database record (cascading will handle related records)
    db.delete(document)
    db.commit()

    return {"message": "Document deleted successfully"}