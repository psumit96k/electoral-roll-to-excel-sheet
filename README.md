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
- **OCR**: PaddleOCR and PaddlePaddle (local inference only)
- **PDF Processing**: PyMuPDF
- **Excel Generation**: openpyxl
- **Frontend**: HTML/CSS/JavaScript (server-rendered with Jinja2 templates)
- **Database**: SQLite (with migration support via Alembic)

## Installation

### 1. Clone the repository

```powershell
git clone https://github.com/psumit96k/electoral-roll-to-excel-sheet.git
cd electoral-roll-to-excel-sheet
```

### 2. Create a virtual environment

Python 3.12 is recommended.

For a standard CPU environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

For a separate GPU environment:

```powershell
py -3.12 -m venv .venv-gpu
.\.venv-gpu\Scripts\Activate.ps1
```

If PowerShell blocks virtual environment activation, use the environment's Python executable directly.

### 3. Install project dependencies

For the CPU environment:

```powershell
python -m pip install -r requirements.txt
```

For the GPU environment:

```powershell
.\.venv-gpu\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Install PaddlePaddle separately (Required for OCR)

PaddleOCR requires PaddlePaddle as its inference framework.

PaddlePaddle must be installed separately because the appropriate package depends on your operating system, hardware, and CUDA configuration.

**CPU installation**

Install the compatible CPU version of PaddlePaddle using the official installation guide.

**GPU installation**

For NVIDIA GPU acceleration, install the compatible GPU version of PaddlePaddle based on your CUDA and cuDNN configuration.

Official installation guide:

https://www.paddlepaddle.org.cn/install/quick

**Important:**

- Install only one PaddlePaddle version in each virtual environment.
- Do not install CPU and GPU PaddlePaddle packages together in the same environment.
- Installing PaddleOCR through `requirements.txt` does not automatically guarantee GPU support.
- Verify PaddlePaddle and CUDA compatibility before using GPU inference.

### 5. Initialize the database

Create the data directory:

```powershell
New-Item -ItemType Directory -Force -Path .\data
```

Initialize the database:

```powershell
python -c "from app.database import create_tables; create_tables()"
```

### 6. Run the application

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### 7. Open the application

Open your browser and visit:

http://127.0.0.1:8000

## Configuration

Configuration is managed through environment variables or a `.env` file.

See `app/config/settings.py` for available options.

## OCR Device Selection (OCR_DEVICE)

The `OCR_DEVICE` setting controls which hardware PaddleOCR uses for inference.

- **cpu**: Uses CPU for OCR processing.
- **gpu**: Requires CUDA-enabled PaddlePaddle and an available GPU. Raises a `RuntimeError` when GPU support is unavailable.
- **auto**: Selects GPU when available (CUDA-enabled PaddlePaddle and GPU detected), otherwise falls back to CPU according to the application's device-selection logic.

### Important Notes

- Device selection happens at OCR service startup. Restart the service after changing `OCR_DEVICE`.
- Initialization or inference failure does not trigger CPU retry or fallback when GPU initialization or inference fails. Errors propagate as-is.
- Unit tests mock GPU availability to simulate different environments without requiring actual GPU hardware.
- GPU support depends on compatible PaddlePaddle, CUDA, and related runtime dependencies.

### PowerShell Startup Examples

**Force CPU usage**

```powershell
$env:OCR_DEVICE="cpu"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

**Force GPU usage**

```powershell
$env:OCR_DEVICE="gpu"
.\.venv-gpu\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

**Automatically select a supported device**

```powershell
$env:OCR_DEVICE="auto"
.\.venv-gpu\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Virtual Environment Notes

- `.venv` and `.venv-gpu` are local environments and are not included in the repository.
- The `.venv-gpu` environment is used for the currently working application setup.
- GPU compatibility depends on the installed PaddlePaddle, CUDA, and cuDNN versions.
- Previous GPU environments produced cuDNN version mismatch warnings. Dependency pins alone do not establish runtime compatibility.
- Verify GPU inference in the target environment before relying on it.

## Usage

1. Upload electoral roll PDF files through the web interface.
2. Wait for background OCR processing to complete.
3. Review and correct extracted data in the review interface.
4. Export confirmed data to Excel using the available export functionality.
5. After successful background document processing, the application also attempts to generate an Excel export automatically in the configured export directory.

Excel export is designed to maintain the required specification, including:

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