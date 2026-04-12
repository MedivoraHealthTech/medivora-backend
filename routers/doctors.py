"""
Doctors Router — Public doctor listing for patients.
"""

from fastapi import APIRouter, Depends

from auth.dependencies import get_current_user
from database import DatabaseManager

router = APIRouter(prefix="/doctors", tags=["doctors"])


# ── GET /doctors ──────────────────────────────────────────────────────

@router.get("")
async def list_doctors(current_user=Depends(get_current_user)):
    """Return all active doctors for the patient-facing directory."""
    db = DatabaseManager()
    doctors = await db.get_all_doctors()
    return {"doctors": doctors}
