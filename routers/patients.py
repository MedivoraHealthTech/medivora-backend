"""
Patients Router — Profile and Medical Records.
"""

from fastapi import APIRouter, Depends, HTTPException

from auth.dependencies import get_current_user, require_patient
from db import get_db
from schemas.patients import PatientProfileUpdate, MedicalRecordCreate

router = APIRouter(prefix="/patients", tags=["patients"])


# ── GET /patients/profile ─────────────────────────────────────────────

@router.get("/profile")
async def get_profile(current_user=Depends(get_current_user)):
    """Get the authenticated user's full profile (profile + patient details)."""
    db = get_db()
    profile_id = current_user["sub"]

    profile = db.get_profile_by_id(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found.")

    # Get role-specific data
    patient = None
    doctor = None
    if profile["user_type"] == "patient":
        patient = db.get_patient_by_profile_id(profile_id)
    elif profile["user_type"] == "doctor":
        doctor = db.get_doctor_by_profile_id(profile_id)

    # Remove password hash from response
    profile.pop("password_hash", None)

    return {
        "profile": profile,
        "patient": patient,
        "doctor": doctor,
    }


# ── PUT /patients/profile ─────────────────────────────────────────────

@router.put("/profile")
async def update_profile(
    data: PatientProfileUpdate,
    current_user=Depends(require_patient),
):
    """Update the authenticated patient's profile."""
    db = get_db()
    profile_id = current_user["sub"]

    patient = db.get_patient_by_profile_id(profile_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient record not found.")

    # Convert to dict, excluding None values
    update_data = data.model_dump(exclude_none=True, mode="json")
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update.")

    updated = db.update_patient(patient["id"], update_data)

    return {
        "message": "Profile updated successfully",
        "patient": updated,
    }


# ── GET /patients/medical-records ─────────────────────────────────────

@router.get("/medical-records")
async def get_medical_records(
    limit: int = 50,
    offset: int = 0,
    current_user=Depends(require_patient),
):
    """Get the authenticated patient's medical records."""
    db = get_db()
    profile_id = current_user["sub"]

    patient = db.get_patient_by_profile_id(profile_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient record not found.")

    records = db.get_medical_records(patient["id"], limit=limit, offset=offset)

    return {
        "records": records,
        "count": len(records),
    }


# ── POST /patients/medical-records ────────────────────────────────────

@router.post("/medical-records", status_code=201)
async def create_medical_record(
    data: MedicalRecordCreate,
    current_user=Depends(require_patient),
):
    """Create a new medical record for the authenticated patient."""
    db = get_db()
    profile_id = current_user["sub"]

    patient = db.get_patient_by_profile_id(profile_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient record not found.")

    record_data = data.model_dump(exclude_none=True, mode="json")
    record = db.create_medical_record(patient_id=patient["id"], **record_data)

    return {
        "message": "Medical record created",
        "record": record,
    }


# ── GET /patients/medical-records/{record_id} ─────────────────────────

@router.get("/medical-records/{record_id}")
async def get_medical_record(
    record_id: str,
    current_user=Depends(require_patient),
):
    """Get a specific medical record by ID."""
    db = get_db()
    profile_id = current_user["sub"]

    patient = db.get_patient_by_profile_id(profile_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient record not found.")

    record = db.get_medical_record_by_id(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Medical record not found.")

    # Ensure the record belongs to this patient
    if record["patient_id"] != patient["id"]:
        raise HTTPException(status_code=403, detail="Access denied.")

    return {"record": record}
