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

## OCR Device Selection (OCR_DEVICE)

The OCR_DEVICE setting controls which hardware PaddleOCR uses for inference:

- **cpu**: Default. Explicitly uses CPU for OCR processing.
- **gpu**: Requires CUDA-enabled PaddlePaddle and an available GPU; fails clearly with RuntimeError when unavailable.
- **auto**: Selects GPU when available (CUDA-enabled PaddlePaddle + GPU detected), otherwise falls back to CPU.

**Important Notes**:
- Selection happens at OCR service startup; restart the service after changing OCR_DEVICE.
- Initialization or inference failure does NOT trigger CPU retry or fallback - errors propagate as-is.
- Test isolation: Unit tests mock GPU availability to simulate different environments without requiring actual GPU hardware.

**PowerShell Startup Examples**:
```powershell
# Force CPU usage
$env:OCR_DEVICE="cpu"; .\.venv\Scripts\python.exe -m app.main

# Attempt GPU usage (will fail if GPU/CUDA unavailable)
$env:OCR_DEVICE="gpu"; .\.venv-gpu99\Scripts\python.exe -m app.main

# Auto-select GPU if available, otherwise CPU
$env:OCR_DEVICE="auto"; .\.venv-gpu99\Scripts\python.exe -m app.main
```

**Virtual Environment Notes**:
- `.venv-gpu99` is marked experimental because its cuDNN override conflicts with Paddle's declared dependency pin.
- The GPU examples use the experimentally validated `.venv-gpu99`. Both `.venv-gpu` and `.venv-gpu118` produced cuDNN version mismatch warnings; dependency pins alone do not establish runtime compatibility.

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
