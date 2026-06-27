"""
Consultations Router — Patient consultation history.
"""

from fastapi import APIRouter, Depends

from auth.dependencies import get_current_user
from database import DatabaseManager

router = APIRouter(prefix="/consultation", tags=["consultations"])


# ── GET /consultation/my ──────────────────────────────────────────────

@router.get("/my")
async def my_consultations(current_user=Depends(get_current_user)):
    """Patient: list own consultation history."""
    db = DatabaseManager()
    sessions = await db.get_patient_consultations(current_user["sub"])
    return {"sessions": sessions}
