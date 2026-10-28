# Election PDF to Excel Converter

A local web application for converting electoral roll PDFs to Excel format with manual review capabilities.

## Features

- PDF upload and batch processing
- Background OCR extraction using PaddleOCR
- Manual review and correction interface
- Exact Excel format generation per specification
- Persistent database storage
- REST API for future integration
- Local-only operation (no cloud OCR dependencies)
- Automatic Excel export after successful background document processing
- Manual Excel export of extracted voter records

## Technology Stack

- **Backend**: Python, FastAPI, SQLAlchemy
- **OCR**: PaddleOCR (local inference only)
- **PDF Processing**: PyMuPDF
- **Excel Generation**: openpyxl
- **Frontend**: HTML/CSS/JavaScript (server-rendered with Jinja2 templates)
- **Database**: SQLite (with migration support via Alembic)

## Installation

1. Clone the repository.

2. Create a virtual environment using Python 3.12:

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks virtual environment activation, you can run the environment's Python executable directly.

3. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

   **Note:** GPU-enabled PaddlePaddle may require a compatible installation for your CUDA environment. Installing the dependencies alone does not guarantee that GPU inference will work.

4. Initialize the database:

   ```powershell
   New-Item -ItemType Directory -Force -Path .\data
   python -c "from app.database import create_tables; create_tables()"
   ```

5. Run the application:
   ```powershell
   python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
   ```

6. Open your browser to `http://127.0.0.1:8000`.

## Configuration

Configuration is managed through environment variables or a `.env` file. See `app/config/settings.py` for available options.

## OCR Device Selection (OCR_DEVICE)

The OCR_DEVICE setting controls which hardware PaddleOCR uses for inference:

- **cpu**: Uses CPU for OCR processing.
- **gpu**: Requires CUDA-enabled PaddlePaddle and an available GPU; fails clearly with RuntimeError when unavailable.
- **auto**: Selects GPU when available (CUDA-enabled PaddlePaddle and GPU detected), otherwise falls back to CPU according to the application's device-selection logic.

**Important Notes**:
- Selection happens at OCR service startup; restart the service after changing OCR_DEVICE.
- Initialization or inference failure does NOT trigger CPU retry or fallback when GPU initialization or inference fails; errors propagate as-is.
- Test isolation: Unit tests mock GPU availability to simulate different environments without requiring actual GPU hardware.
- GPU support depends on compatible PaddlePaddle, CUDA, and related runtime dependencies.

**PowerShell Startup Examples**:

```powershell
# Force CPU usage
$env:OCR_DEVICE="cpu"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# Attempt GPU usage with the existing GPU environment
$env:OCR_DEVICE="gpu"
.\.venv-gpu\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# Automatically select a supported device
$env:OCR_DEVICE="auto"
.\.venv-gpu\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

**Virtual Environment Notes**:
- The `.venv-gpu` environment is used for the currently working application setup.
- GPU compatibility depends on the installed PaddlePaddle, CUDA, and cuDNN versions.
- Previous GPU environments produced cuDNN version mismatch warnings. Dependency pins alone do not establish runtime compatibility.
- Verify GPU inference in the target environment before relying on it.

## Usage

1. Upload electoral roll PDF files via the web interface.
2. Wait for background OCR processing to complete.
3. Review and correct extracted data in the review interface.
4. Export confirmed data to Excel format using the available export functionality.
5. After successful background document processing, the application also attempts to generate an Excel export automatically in the configured export directory.
6. Excel export is designed to maintain the required specification, including:
   - Proper column headers and order
   - Correct data mapping (Father's/Husband's name)
   - String formatting for phone numbers and EPIC
   - Proper Excel cell types (no formula interpretation)
   - Formatting (headers, widths, filtering, etc.)

Automatic export failures are logged for troubleshooting. The generated Excel file depends on the records available and the application's export configuration.

## Project Status

See `PROJECT_STATUS.md` for current implementation progress and milestones.

## License

This project is for internal use only. Redistribution requires permission.
