# Election PDF to Excel Converter

A local web application for converting electoral roll PDFs to Excel format with manual review capabilities.

## Features

- PDF upload and batch processing
- Background OCR extraction using PaddleOCR
- Manual review and correction interface
- Exact Excel format generation per specification
- Persistent database storage
- REST API for future integration
- Local-only operation (no cloud dependencies)

## Technology Stack

- **Backend**: Python, FastAPI, SQLAlchemy
- **OCR**: PaddleOCR (local inference only)
- **PDF Processing**: PyMuPDF
- **Excel Generation**: openpyxl
- **Frontend**: HTML/CSS/JavaScript (server-rendered with Jinja2 templates)
- **Database**: SQLite (with migration support via Alembic)

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Initialize the database:
   ```bash
   python -c "from app.database import create_tables; create_tables()"
   ```
5. Run the application:
   ```bash
   python app/main.py
   ```
6. Open your browser to `http://127.0.0.1:8000`

## Configuration

Configuration is managed through environment variables or a `.env` file. See `app/config/settings.py` for available options.

## Usage

1. Upload electoral roll PDF files via the web interface
2. Wait for background OCR processing to complete
3. Review and correct extracted data in the review interface
4. Export confirmed data to Excel format
5. Export maintains exact specification including:
   - Proper column headers and order
   - Correct data mapping (Father's/Husband's name)
   - String formatting for phone numbers and EPIC
   - Proper Excel cell types (no formula interpretation)
   - Formatting (headers, widths, filtering, etc.)

## Project Status

See `PROJECT_STATUS.md` for current implementation progress and milestones.

## License

This project is for internal use only. Redistribution requires permission.