"""
Main API router that includes all API endpoints.
"""
from fastapi import APIRouter
from app.api import upload, review, export

api_router = APIRouter()

# Include all API routers
api_router.include_router(upload.router)
api_router.include_router(review.router)
api_router.include_router(export.router)