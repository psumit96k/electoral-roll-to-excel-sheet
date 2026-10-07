"""
Database models for the election PDF to Excel converter.
"""
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, ForeignKey, Float, Date, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
from datetime import datetime

Base = declarative_base()


class ProcessingStatusEnum(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


class ReviewStatusEnum(str, enum.Enum):
    UNREVIEWED = "unreviewed"
    FLAGGED = "flagged"
    REVIEWED = "reviewed"
    CONFIRMED = "confirmed"
    EXCLUDED = "excluded"


class Batch(Base):
    """Batch of uploaded PDFs."""
    __tablename__ = "batches"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    status = Column(String(50), default="active")

    # Relationships
    documents = relationship("Document", back_populates="batch")


class Document(Base):
    """Uploaded PDF document."""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    batch_id = Column(Integer, ForeignKey("batches.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_hash = Column(String(64), nullable=False, unique=True)  # SHA-256
    file_size = Column(Integer, nullable=False)  # bytes
    page_count = Column(Integer, nullable=False)
    upload_timestamp = Column(DateTime(timezone=True), server_default=func.now())
    processing_status = Column(String(50), default=ProcessingStatusEnum.PENDING.value)

    # Relationships
    batch = relationship("Batch", back_populates="documents")
    pages = relationship("Page", back_populates="document")
    jobs = relationship("Job", back_populates="document")


class Page(Base):
    """Individual page within a document."""
    __tablename__ = "pages"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    page_number = Column(Integer, nullable=False)  # 1-based index
    width = Column(Integer, nullable=False)  # pixels
    height = Column(Integer, nullable=False)  # pixels
    processing_status = Column(String(50), default=ProcessingStatusEnum.PENDING.value)

    # Relationships
    document = relationship("Document", back_populates="pages")
    card_slots = relationship("CardSlot", back_populates="page")


class CardSlot(Base):
    """Physical card slot on a page (based on the 30-slot geometry)."""
    __tablename__ = "card_slots"

    id = Column(Integer, primary_key=True, index=True)
    page_id = Column(Integer, ForeignKey("pages.id"), nullable=False)
    physical_slot = Column(Integer, nullable=False)  # 1-30
    row_number = Column(Integer, nullable=False)  # 1-10
    column_number = Column(Integer, nullable=False)  # 1-3

    # Bounding box coordinates (in original image pixels)
    left = Column(Integer, nullable=False)
    top = Column(Integer, nullable=False)
    right = Column(Integer, nullable=False)
    bottom = Column(Integer, nullable=False)

    # Processing results
    is_occupied = Column(Boolean, default=False)
    confidence = Column(Float, nullable=True)

    # Relationships
    page = relationship("Page", back_populates="card_slots")
    extraction_candidates = relationship("ExtractionCandidate", back_populates="card_slot")
    review_items = relationship("ReviewItem", back_populates="card_slot")


class ExtractionCandidate(Base):
    """Raw OCR extraction results for a card slot."""
    __tablename__ = "extraction_candidates"

    id = Column(Integer, primary_key=True, index=True)
    card_slot_id = Column(Integer, ForeignKey("card_slots.id"), nullable=False)

    # Raw OCR values
    raw_full_name = Column(String(500), nullable=True)
    raw_relation_type = Column(String(100), nullable=True)
    raw_relative_name = Column(String(500), nullable=True)
    raw_house_number = Column(String(100), nullable=True)
    raw_age = Column(String(50), nullable=True)
    raw_gender = Column(String(50), nullable=True)
    raw_epic = Column(String(100), nullable=True)

    # Normalized/parsed values
    normalized_epic = Column(String(100), nullable=True)
    parsed_age = Column(Integer, nullable=True)

    # OCR metadata
    ocr_confidence = Column(Float, nullable=True)
    ocr_model_version = Column(String(100), nullable=True)
    processing_timestamp = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    card_slot = relationship("CardSlot", back_populates="extraction_candidates")
    review_items = relationship("ReviewItem", back_populates="extraction_candidate")


class ReviewItem(Base):
    """Item for manual review and confirmation."""
    __tablename__ = "review_items"

    id = Column(Integer, primary_key=True, index=True)
    card_slot_id = Column(Integer, ForeignKey("card_slots.id"), nullable=False)
    extraction_candidate_id = Column(Integer, ForeignKey("extraction_candidates.id"), nullable=False)

    # Reviewed/confirmed values (what goes into the final Excel)
    reviewed_full_name = Column(String(500), nullable=True)
    reviewed_relation_type = Column(String(100), nullable=True)
    reviewed_relative_name = Column(String(500), nullable=True)
    reviewed_house_number = Column(String(100), nullable=True)
    reviewed_age = Column(Integer, nullable=True)
    reviewed_gender = Column(String(50), nullable=True)  # MALE, FEMALE, OTHER or empty
    reviewed_caste = Column(String(100), nullable=True)
    reviewed_phone_number = Column(String(50), nullable=True)
    reviewed_epic = Column(String(100), nullable=True)

    # Review status and metadata
    review_status = Column(String(50), default=ReviewStatusEnum.UNREVIEWED.value)
    reviewed_by = Column(String(255), nullable=True)
    reviewed_at = Column(DateTime(timezone=True), nullable=True)
    confirmation_notes = Column(Text, nullable=True)
    exclusion_reason = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    card_slot = relationship("CardSlot", back_populates="review_items")
    extraction_candidate = relationship("ExtractionCandidate", back_populates="review_items")


class Person(Base):
    """Stable person record (for deduplication across electoral rolls)."""
    __tablename__ = "people"

    id = Column(Integer, primary_key=True, index=True)
    # Stable identifier that persists across different electoral rolls

    # Effective profile fields (what's currently confirmed)
    effective_full_name = Column(String(500), nullable=True)
    effective_father_name = Column(String(500), nullable=True)
    effective_husband_name = Column(String(500), nullable=True)
    effective_gender = Column(String(50), nullable=True)
    effective_caste = Column(String(100), nullable=True)
    effective_phone_number = Column(String(50), nullable=True)

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    electoral_registrations = relationship("ElectoralRegistration", back_populates="person")


class ElectoralRegistration(Base):
    """Historical electoral roll registration linked to a person."""
    __tablename__ = "electoral_registrations"

    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(Integer, ForeignKey("people.id"), nullable=False)

    # Source document info
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    card_slot_id = Column(Integer, ForeignKey("card_slots.id"), nullable=False)

    # Registration-specific fields (historical values from this specific roll)
    registration_full_name = Column(String(500), nullable=True)
    registration_father_name = Column(String(500), nullable=True)
    registration_husband_name = Column(String(500), nullable=True)
    registration_gender = Column(String(50), nullable=True)
    registration_age = Column(Integer, nullable=True)
    registration_age_reference_date = Column(Date, nullable=True)
    registration_epic = Column(String(100), nullable=True)

    # Location info (from document metadata)
    house_number = Column(String(100), nullable=True)
    constituency_number = Column(String(100), nullable=True)
    constituency_name = Column(String(255), nullable=True)
    part_number = Column(String(50), nullable=True)
    section_number = Column(String(50), nullable=True)
    village = Column(String(255), nullable=True)
    taluka = Column(String(255), nullable=True)
    district = Column(String(255), nullable=True)
    polling_station_number = Column(String(100), nullable=True)
    polling_station_name = Column(String(255), nullable=True)
    polling_station_address = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    person = relationship("Person", back_populates="electoral_registrations")
    document = relationship("Document")
    card_slot = relationship("CardSlot")


class Job(Base):
    """Background processing job tracking."""
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    job_type = Column(String(100), nullable=False)  # e.g., "ocr_extraction", "preprocessing"
    status = Column(String(50), default=ProcessingStatusEnum.PENDING.value)
    progress_percent = Column(Integer, default=0)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    error_message = Column(Text, nullable=True)

    # Relationships
    document = relationship("Document", back_populates="jobs")


class ExportHistory(Base):
    """Track Excel exports for auditing."""
    __tablename__ = "export_history"

    id = Column(Integer, primary_key=True, index=True)
    export_timestamp = Column(DateTime(timezone=True), server_default=func.now())
    exported_by = Column(String(255), nullable=True)
    record_count = Column(Integer, nullable=False)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    document_ids = Column(Text, nullable=True)  # Comma-separated list of document IDs
    export_parameters = Column(Text, nullable=True)  # JSON string of export options

    # Relationship to source documents could be added here


# Create all tables
def create_tables(engine):
    Base.metadata.create_all(bind=engine)