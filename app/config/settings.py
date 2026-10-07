"""
Application configuration settings.
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional

class Settings(BaseSettings):
    # Application settings
    APP_NAME: str = "Election PDF to Excel Converter"
    DEBUG: bool = False
    HOST: str = "127.0.0.1"
    PORT: int = 8000

    # File paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = BASE_DIR / "models"
    EXPORT_DIR: Path = BASE_DIR / "exports"

    # Database
    DATABASE_URL: str = Field(
        default="sqlite:///./data/election.db",
        description="Database connection URL"
    )

    # OCR settings
    OCR_LANG: str = "en"
    OCR_USE_DOC_ORIENTATION_CLASSIFY: bool = False
    OCR_USE_DOC_UNWARPING: bool = False
    OCR_USE_TEXTLINE_ORIENTATION: bool = False
    OCR_ENABLE_MKLDNN: bool = False
    OCR_DEVICE: str = "cpu"  # cpu, gpu, auto

    # Job processing
    MAX_WORKERS: int = 4
    JOB_TIMEOUT_MINUTES: int = 30

    # Upload limits
    MAX_FILE_SIZE_MB: int = 50
    MAX_PAGES_PER_PDF: int = 100
    MAX_BATCH_PDFS: int = 10

    # PaddleOCR model settings
    PADDLEOCR_MODEL_DIR: Optional[Path] = None

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

# Global settings instance
settings = Settings()

# Ensure directories exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
settings.EXPORT_DIR.mkdir(parents=True, exist_ok=True)