"""
Doctors Router — Public doctor listing for patients.
"""

from typing import Optional

from fastapi import APIRouter, Depends

from auth.dependencies import get_current_user_optional
from database import DatabaseManager

router = APIRouter(prefix="/doctors", tags=["doctors"])


# ── GET /doctors ──────────────────────────────────────────────────────

@router.get("")
async def list_doctors(specialty: Optional[str] = None, current_user=Depends(get_current_user_optional)):
    """Return all active doctors for the patient-facing directory. Public endpoint — no auth required."""
    db = DatabaseManager()
    doctors = await db.get_all_doctors(patient_facing=True)
    if specialty:
        spec_lower = specialty.lower().strip()
        filtered = [d for d in doctors if spec_lower in (d.get("specialization") or "").lower()]
        doctors = filtered if filtered else doctors
    return {"doctors": doctors, "count": len(doctors), "filtered_by": specialty or None}
