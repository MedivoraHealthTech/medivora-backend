from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime
import json

@dataclass
class PatientProfile:
    id: str
    name: str
    age: int
    gender: str
    phone: str
    address: str
    medical_history: List[str]
    allergies: List[str]
    current_medications: List[str]
    emergency_contact: str
    created_at: datetime
    updated_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class Symptom:
    description: str
    duration: str
    severity: int  # 1-10 scale
    location: Optional[str] = None
    triggers: Optional[List[str]] = None
    onset: Optional[str] = "gradual"  # gradual, sudden, intermittent

@dataclass
class Assessment:
    patient_id: str
    symptoms: List[Symptom]
    specialty: str
    risk_level: str
    diagnosis: str
    reasoning: str
    recommendations: List[str]
    prescription: Optional[str]
    follow_up: str
    confidence_score: float
    timestamp: datetime
    # New fields for doctor approval
    requires_doctor_approval: bool = True
    doctor_notes: str = ""
    approval_status: str = "pending"  # pending, approved, rejected, modified
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    approval_id: Optional[str] = None
    modified_prescription: Optional[str] = None
    doctor_feedback: Optional[str] = None

@dataclass
class DoctorSummary:
    patient_profile: PatientProfile
    assessment: Assessment
    formatted_summary: str
    recommendations: List[str]
    # New fields for approval workflow
    approval_priority: int = 3  # 1=highest, 5=lowest
    requires_urgent_review: bool = False
    clinical_notes: str = ""
    suggested_tests: List[str] = None

@dataclass 
class ApprovalRequest:
    approval_id: str
    patient_id: str
    assessment_id: str
    ai_assessment: Assessment
    proposed_prescription: Dict[str, Any]
    priority: int
    status: str  # pending, approved, rejected, modified
    created_at: datetime
    assigned_doctor: Optional[str] = None
    doctor_response: Optional[str] = None
    approved_prescription: Optional[Dict[str, Any]] = None
    rejection_reason: Optional[str] = None
    modified_at: Optional[datetime] = None

@dataclass
class DoctorProfile:
    id: str
    name: str
    phone: str
    email: str
    license_number: str
    specialties: List[str]
    experience_years: int
    status: str  # available, busy, offline
    rating: float
    cases_handled: int
    average_response_time: int  # in minutes
    created_at: datetime
    last_active: datetime

@dataclass
class PrescriptionItem:
    medicine_name: str
    generic_name: str
    dosage: str
    frequency: str
    duration: str
    instructions: str
    contraindications: List[str]
    side_effects: List[str]
    cost_estimate: Optional[float] = None

@dataclass
class DetailedPrescription:
    prescription_id: str
    patient_id: str
    doctor_id: str
    items: List[PrescriptionItem]
    general_instructions: List[str]
    warning_signs: List[str]
    dietary_advice: List[str]
    follow_up_instructions: str
    validity_days: int
    created_at: datetime
    approved_at: datetime
    status: str  # draft, approved, dispensed, expired