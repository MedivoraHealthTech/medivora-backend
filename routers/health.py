"""
Health check endpoint.
"""

from datetime import datetime, timezone

import httpx
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


@router.get("/debug/outbound-ip")
async def outbound_ip():
    """Returns the server's outbound IP — used to configure MSG91 whitelist."""
    async with httpx.AsyncClient(timeout=5) as client:
        r = await client.get("https://ifconfig.me/ip")
    return {"outbound_ip": r.text.strip()}
