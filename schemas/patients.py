"""
Patient Request / Response Schemas
"""

from datetime import date
from typing import List, Optional
from pydantic import BaseModel, Field


class PatientProfileUpdate(BaseModel):
    """Fields a patient can update on their own profile."""
    age: Optional[int] = Field(default=None, ge=0, le=120)
    gender: Optional[str] = None
    date_of_birth: Optional[date] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    postal_code: Optional[str] = None

    blood_group: Optional[str] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None

    emergency_contact_name: Optional[str] = None
    emergency_contact_phone: Optional[str] = None
    emergency_contact_relation: Optional[str] = None

    medical_history: Optional[List[str]] = None
    allergies: Optional[List[str]] = None
    current_medications: Optional[List[str]] = None
    chronic_conditions: Optional[List[str]] = None

    is_smoker: Optional[bool] = None
    is_alcohol_user: Optional[bool] = None
    is_pregnant: Optional[bool] = None
    is_nursing: Optional[bool] = None

    insurance_provider: Optional[str] = None
    insurance_policy_number: Optional[str] = None


class MedicalRecordCreate(BaseModel):
    """Create a new medical record."""
    record_type: str = Field(..., min_length=1, max_length=100)
    title: Optional[str] = Field(default=None, max_length=255)
    description: Optional[str] = None
    diagnosis: Optional[str] = None
    onset_date: Optional[date] = None
    resolution_date: Optional[date] = None
    status: Optional[str] = Field(default="ongoing")
    clinical_notes: Optional[str] = None
    treatment_summary: Optional[str] = None
    medications: Optional[List[str]] = None


class MedicalRecordResponse(BaseModel):
    id: str
    patient_id: str
    record_type: str
    title: Optional[str] = None
    description: Optional[str] = None
    diagnosis: Optional[str] = None
    onset_date: Optional[date] = None
    status: Optional[str] = None
    created_at: str
