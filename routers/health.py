"""
Health check endpoint.
"""

from datetime import datetime, timezone

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "medivora-api",
        "version": "2.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/")
async def root():
    return {
        "message": "Medivora Medical AI Assistant API",
        "version": "2.0.0",
        "docs": "/docs",
    }
