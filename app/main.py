"""
Main FastAPI application entry point.
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from app.api import upload, review, export
from app.api.api import api_router
from app.config.settings import settings

app = FastAPI(
    title="Election PDF to Excel Converter",
    description="A local application for converting electoral roll PDFs to Excel format",
    version="0.1.0"
)

# Include API routers
app.include_router(upload.router, prefix="/api/v1/upload", tags=["upload"])
app.include_router(review.router, prefix="/api/v1/review", tags=["review"])
app.include_router(export.router, prefix="/api/v1/export", tags=["export"])
app.include_router(api_router, prefix="/api/v1", tags=["api"])

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def root():
    """Root endpoint - serve the main interface."""
    return {"message": "Election PDF to Excel Converter API"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )