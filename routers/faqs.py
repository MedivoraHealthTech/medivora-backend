"""
FAQ router — public endpoint, no auth required.
"""

from fastapi import APIRouter, HTTPException
from db import get_db

router = APIRouter(prefix="/faqs", tags=["faqs"])


@router.get("")
async def list_faqs():
    """Return all active FAQs ordered by display_order."""
    db = get_db()
    try:
        result = (
            db.client.table("faqs")
            .select("id, question, points, display_order")
            .eq("is_active", True)
            .order("display_order")
            .execute()
        )
        return result.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
