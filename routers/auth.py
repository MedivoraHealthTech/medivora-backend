"""
Auth Router — Signup, Login, OTP endpoints.
"""

import re as _re
import random
import string
from datetime import datetime, timezone



def _parse_ts(ts) -> datetime:
    """Parse a Supabase timestamp — handles variable microsecond digits (Python 3.10 compat)."""
    if not isinstance(ts, str):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    ts = _re.sub(r'\.(\d+)', lambda m: '.' + m.group(1)[:6].ljust(6, '0'), ts)
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


import httpx
from fastapi import APIRouter, Depends, Form, HTTPException, Request
from pydantic import BaseModel

from auth.dependencies import get_current_user
from auth.jwt_handler import create_token
from auth.password_handler import hash_password, verify_password, validate_password_strength
from config import settings
from database import _name
from db import get_db
from schemas.auth import (
    SignupRequest,
    LoginRequest,
    SendOTPRequest,
    VerifyOTPRequest,
)

router = APIRouter(prefix="/auth", tags=["auth"])


class _PatientOTPSendRequest(BaseModel):
    phone: str  # E.164 format, e.g. +919876543210


class _PatientOTPVerifyRequest(BaseModel):
    phone: str
    otp: str


def _generate_otp(length: int = 6) -> str:
    """Generate a random numeric OTP."""
    return "".join(random.choices(string.digits, k=length))


async def _send_via_msg91_flow(phone: str, otp: str) -> None:
    """Send OTP via MSG91 Flow (SMS) API. Passes OTP into the DLT-registered SMS template."""
    mobile = phone.lstrip("+")  # MSG91 expects no leading +
    payload = {
        "flow_id": settings.MSG91_OTP_TEMPLATE_ID,
        "sender":  settings.MSG91_SENDER_ID,
        "mobiles": mobile,
        "OTP":     otp,
    }
    if settings.MSG91_DLT_TEMPLATE_ID:
        payload["DLT_TE_ID"] = settings.MSG91_DLT_TEMPLATE_ID
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            "https://control.msg91.com/api/v5/flow/",
            json=payload,
            headers={"authkey": settings.MSG91_AUTH_KEY, "Content-Type": "application/json"},
        )
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"MSG91 error {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    if data.get("type") == "error":
        raise HTTPException(status_code=502, detail=f"MSG91: {data.get('message', 'Unknown error')}")
    return data


# ── POST /auth/send-patient-otp ──────────────────────────────────────
# Generates OTP, stores in DB, sends via MSG91 Flow (SMS) API.

OTP_SEND_COOLDOWN_SECONDS = 30

@router.post("/send-patient-otp")
async def send_patient_otp(req: _PatientOTPSendRequest):
    """Send a 6-digit OTP to a patient phone via MSG91."""
    db = get_db()

    # Cooldown: prevent MSG91 error 311 (duplicate SMS within 10s) and abuse
    latest = db.get_latest_otp(req.phone)
    if latest:
        created_at = _parse_ts(latest["created_at"])
        elapsed = (datetime.now(timezone.utc) - created_at).total_seconds()
        if elapsed < OTP_SEND_COOLDOWN_SECONDS:
            wait = int(OTP_SEND_COOLDOWN_SECONDS - elapsed)
            raise HTTPException(
                status_code=429,
                detail=f"Please wait {wait} seconds before requesting a new OTP.",
            )

    otp = _generate_otp()
    db.create_otp(phone=req.phone, otp_code=otp, ttl_minutes=settings.OTP_TTL_MINUTES)

    if settings.OTP_MOCK_MODE:
        return {"message": "OTP sent successfully (mock)", "otp": otp}

    if not settings.MSG91_AUTH_KEY or not settings.MSG91_OTP_TEMPLATE_ID:
        raise HTTPException(status_code=500, detail="MSG91 not configured on server.")
    await _send_via_msg91_flow(req.phone, otp)
    return {"message": "OTP sent successfully"}


# ── POST /auth/verify-patient-otp ────────────────────────────────────
# Verifies OTP against local DB (OTP was stored on send).

@router.post("/verify-patient-otp")
async def verify_patient_otp(req: _PatientOTPVerifyRequest):
    """Verify patient OTP and return a custom JWT. Creates profile on first login."""
    db = get_db()

    if not db.verify_otp(phone=req.phone, otp_code=req.otp):
        raise HTTPException(status_code=400, detail="Invalid or expired OTP.")

    existing = db.get_profile_by_phone(req.phone)
    is_new_user = existing is None

    if is_new_user:
        placeholder_hash = hash_password("otp_verified_no_password")
        profile = db.create_profile(
            phone=req.phone,
            first_name="",
            last_name="",
            password_hash=placeholder_hash,
            user_type="patient",
        )
        db.create_patient(profile_id=profile["id"])
    else:
        profile = existing

    if profile.get("status", "active") != "active":
        raise HTTPException(status_code=403, detail=f"Account is {profile['status']}.")

    db.update_profile(profile["id"], {"phone_verified": True})
    db.update_last_login(profile["id"])

    token = create_token(user_id=str(profile["id"]), role="patient")
    return {
        "token":      token,
        "user_id":    str(profile["id"]),
        "user_type":  "patient",
        "full_name":  _name(profile.get("first_name"), profile.get("last_name")),
        "is_new_user": is_new_user,
    }


# ── POST /auth/verify-dual-otp ───────────────────────────────────────
# New endpoint for dual-role accounts (e.g. a doctor who also uses the
# patient flow). Verifies OTP and always returns a patient-scoped JWT,
# creating a patients row if one doesn't exist yet for this profile.

@router.post("/verify-dual-otp")
async def verify_dual_otp(req: _PatientOTPVerifyRequest):
    """Verify OTP for a dual-role account and return a patient JWT.
    Works for any profile (patient or doctor). Does NOT touch verify-patient-otp."""
    db = get_db()

    if not db.verify_otp(phone=req.phone, otp_code=req.otp):
        raise HTTPException(status_code=400, detail="Invalid or expired OTP.")

    profile = db.get_profile_by_phone(req.phone)
    if not profile:
        raise HTTPException(status_code=404, detail="No account found for this phone number.")

    if profile.get("status", "active") != "active":
        raise HTTPException(status_code=403, detail=f"Account is {profile['status']}.")

    # Ensure a patients row exists (creates one if this is a doctor-only account)
    patient_row = db.get_patient_by_profile_id(profile["id"])
    if patient_row is None:
        db.create_patient(profile_id=profile["id"])

    db.update_profile(profile["id"], {"phone_verified": True})
    db.update_last_login(profile["id"])

    token = create_token(user_id=str(profile["id"]), role="patient")
    return {
        "token":     token,
        "user_id":   str(profile["id"]),
        "user_type": "patient",
        "full_name": _name(profile.get("first_name"), profile.get("last_name")),
        "is_new_user": False,
    }


def _get_client_ip(request: Request) -> str:
    """Extract client IP from request."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else ""


# ── POST /auth/signup ─────────────────────────────────────────────────
# Patient signup is handled by Supabase Phone OTP + handle_new_user DB trigger.
# This endpoint remains for doctor onboarding only.

@router.post("/signup")
async def signup(req: SignupRequest):
    """Register a new patient or doctor."""
    db = get_db()

    # Check if phone already exists
    existing = db.get_profile_by_phone(req.phone)
    if existing:
        raise HTTPException(status_code=409, detail="Phone number already registered.")

    # Validate password strength
    is_valid, err = validate_password_strength(req.password, req.user_type)
    if not is_valid:
        raise HTTPException(status_code=422, detail=err)

    # Create profile
    hashed = hash_password(req.password)
    _name_parts = (req.name or "").strip().split(" ", 1)
    profile = db.create_profile(
        phone=req.phone,
        first_name=_name_parts[0],
        last_name=_name_parts[1] if len(_name_parts) > 1 else "",
        password_hash=hashed,
        user_type=req.user_type,
        email=req.email,
    )

    profile_id = profile["id"]

    # Create role-specific record
    if req.user_type == "patient":
        db.create_patient(profile_id=profile_id)
    elif req.user_type == "doctor":
        doctor_data = {}
        if req.nmc_number:
            doctor_data["nmc_number"] = req.nmc_number
        if req.specialties:
            doctor_data["specialties"] = req.specialties
        if req.experience_years is not None:
            doctor_data["experience_years"] = req.experience_years
        db.create_doctor(profile_id=profile_id, **doctor_data)

    # Generate JWT
    token = create_token(user_id=str(profile_id), role=req.user_type)

    return {
        "message": "Signup successful",
        "user_id": str(profile_id),
        "token": token,
        "user_type": req.user_type,
        "is_new_user": True,
    }


# ── POST /auth/login ─────────────────────────────────────────────────

@router.post("/login")
async def login(req: LoginRequest, request: Request):
    """Login with phone + password — doctors and admins only.
    Patient auth is handled entirely by Supabase Phone OTP."""
    db = get_db()
    ip = _get_client_ip(request)

    profile = db.get_profile_by_phone(req.phone)
    if not profile or profile.get("user_type") not in ("doctor", "admin"):
        db.record_login_attempt(req.phone, "unknown", False, ip)
        raise HTTPException(status_code=401, detail="Invalid phone or password.")

    # Check lockout
    failed_count = db.count_failed_attempts(req.phone, settings.LOCKOUT_WINDOW_MINUTES)
    if failed_count >= settings.MAX_LOGIN_ATTEMPTS:
        raise HTTPException(
            status_code=429,
            detail=f"Too many failed attempts. Try again in {settings.LOCKOUT_WINDOW_MINUTES} minutes.",
        )

    if not verify_password(req.password, profile["password_hash"]):
        db.record_login_attempt(req.phone, profile["user_type"], False, ip)
        raise HTTPException(status_code=401, detail="Invalid phone or password.")

    if profile["status"] != "active":
        raise HTTPException(status_code=403, detail=f"Account is {profile['status']}.")

    db.record_login_attempt(req.phone, profile["user_type"], True, ip)
    db.update_last_login(profile["id"])

    token = create_token(user_id=str(profile["id"]), role=profile["user_type"])

    return {
        "message": "Login successful",
        "user_id": str(profile["id"]),
        "token": token,
        "user_type": profile["user_type"],
        "full_name": _name(profile.get("first_name"), profile.get("last_name")),
    }


# ── POST /auth/send-otp ──────────────────────────────────────────────

@router.post("/send-otp")
async def send_otp(req: SendOTPRequest):
    """Send a 6-digit OTP to the given phone number via MSG91."""
    db = get_db()

    # Cooldown: prevent abuse and MSG91 duplicate-SMS errors
    latest = db.get_latest_otp(req.phone)
    if latest:
        created_at = _parse_ts(latest["created_at"])
        elapsed = (datetime.now(timezone.utc) - created_at).total_seconds()
        if elapsed < OTP_SEND_COOLDOWN_SECONDS:
            wait = int(OTP_SEND_COOLDOWN_SECONDS - elapsed)
            raise HTTPException(
                status_code=429,
                detail=f"Please wait {wait} seconds before requesting a new OTP.",
            )

    otp = _generate_otp()
    db.create_otp(phone=req.phone, otp_code=otp, ttl_minutes=settings.OTP_TTL_MINUTES)

    # Mock mode — return OTP directly (no SMS sent)
    if settings.OTP_MOCK_MODE:
        return {
            "message": "OTP sent successfully",
            "phone": req.phone,
            "otp_for_testing": otp,
            "note": "Mock mode — OTP returned in response for development",
        }

    # Production — send via MSG91
    if not settings.MSG91_AUTH_KEY or not settings.MSG91_OTP_TEMPLATE_ID:
        raise HTTPException(status_code=500, detail="MSG91 not configured on server.")
    await _send_via_msg91_flow(req.phone, otp)

    return {"message": "OTP sent successfully", "phone": req.phone}


# ── POST /auth/verify-otp ────────────────────────────────────────────

@router.post("/verify-otp")
async def verify_otp(req: VerifyOTPRequest):
    """Doctor OTP verify. Auto-creates a doctor profile + doctors row on first login."""
    db = get_db()

    if settings.OTP_MOCK_MODE:
        is_valid = True
    else:
        is_valid = db.verify_otp(phone=req.phone, otp_code=req.otp)

    if not is_valid:
        raise HTTPException(status_code=400, detail="Invalid or expired OTP.")

    existing = db.get_profile_by_phone(req.phone)
    is_new_user = existing is None

    if is_new_user:
        placeholder_hash = hash_password("otp_verified_no_password")
        name = (req.name or "").strip()
        name_parts = name.split(" ", 1) if name else ["", ""]
        profile = db.create_profile(
            phone=req.phone,
            first_name=name_parts[0],
            last_name=name_parts[1] if len(name_parts) > 1 else "",
            password_hash=placeholder_hash,
            user_type="doctor",
        )
        doctor_row = db.create_doctor(profile_id=profile["id"], available_status="inactive")
        # Auto-create a pending join request so admin sees this doctor immediately
        try:
            db.client.table("doctor_join_requests").insert({
                "doctor_id":        doctor_row["id"] if doctor_row else None,
                "first_name":       name_parts[0],
                "last_name":        name_parts[1] if len(name_parts) > 1 else "",
                "phone":            req.phone,
                "email":            "",
                "specialties":      "general_medicine",
                "experience_years": 0,
                "medical_college":  "",
                "nmc_number":       "",
                "clinic_name":      "",
                "clinic_address":   "",
                "status":           "draft",
            }).execute()
        except Exception:
            pass  # join request creation is non-blocking
    else:
        profile = existing

    if profile.get("status", "active") != "active":
        raise HTTPException(status_code=403, detail=f"Account is {profile['status']}.")

    db.update_profile(profile["id"], {"phone_verified": True})
    db.update_last_login(profile["id"])

    token = create_token(user_id=str(profile["id"]), role=profile["user_type"])

    return {
        "message": "OTP verified successfully",
        "user_id": str(profile["id"]),
        "token": token,
        "user_type": profile["user_type"],
        "full_name": _name(profile.get("first_name"), profile.get("last_name")),
        "is_new_user": is_new_user,
    }


def _merge_user(profile: dict, patient: dict | None) -> dict:
    """Merge profile + patient rows into a single user dict.
    Normalises field names so the frontend always receives consistent keys
    (e.g. emergency_contact_phone → emergency_contact).
    """
    merged = {**(profile or {}), **(patient or {})}
    # Map DB column name → frontend field name
    if "emergency_contact_phone" in merged and "emergency_contact" not in merged:
        merged["emergency_contact"] = merged["emergency_contact_phone"]
    return merged


# ── GET /auth/user/{user_id} ─────────────────────────────────────────

@router.get("/user/{user_id}")
async def get_user_profile(
    user_id: str,
    current_user=Depends(get_current_user),
):
    """Get patient profile by ID (own record only)."""
    if current_user["sub"] != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    db = get_db()

    # profiles.id == Supabase auth UID after migration — direct lookup only.
    profile = db.get_profile_by_id(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="User not found")

    profile.pop("password_hash", None)
    patient = db.get_patient_by_profile_id(profile["id"])

    return {"user": _merge_user(profile, patient)}


# ── PUT /auth/user/{user_id} ─────────────────────────────────────────

@router.put("/user/{user_id}")
async def update_user_profile(
    user_id: str,
    current_user=Depends(get_current_user),
    full_name: str = Form(default=None),
    first_name: str = Form(default=None),
    last_name: str = Form(default=None),
    email: str = Form(default=None),
    phone: str = Form(default=None),
    age: str = Form(default=None),
    gender: str = Form(default=None),
    blood_group: str = Form(default=None),
    height_cm: str = Form(default=None),
    weight_kg: str = Form(default=None),
    address: str = Form(default=None),
    emergency_contact: str = Form(default=None),
    emergency_contact_name: str = Form(default=None),
    emergency_contact_phone: str = Form(default=None),
    emergency_contact_relation: str = Form(default=None),
    medical_history: str = Form(default=None),
    current_medications: str = Form(default=None),
    allergies: str = Form(default=None),
    chronic_conditions: str = Form(default=None),
    is_smoker: str = Form(default=None),
    is_alcohol_user: str = Form(default=None),
    is_pregnant: str = Form(default=None),
    is_nursing: str = Form(default=None),
):
    """Update patient profile and medical details (own record only)."""
    if current_user["sub"] != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    db = get_db()
    profile_data = {}
    patient_data = {}

    if full_name is not None and full_name.strip():
        _fn_parts = full_name.strip().split(" ", 1)
        profile_data["first_name"] = _fn_parts[0]
        profile_data["last_name"] = _fn_parts[1] if len(_fn_parts) > 1 else ""
    if first_name is not None and first_name.strip():
        profile_data["first_name"] = first_name.strip()
    if last_name is not None and last_name.strip():
        profile_data["last_name"] = last_name.strip()
    if email is not None and email.strip():
        profile_data["email"] = email.strip()

    # Allow phone update only for email-signup users (no phone claim in JWT)
    if phone is not None and phone.strip():
        cleaned_phone = phone.strip()
        if not cleaned_phone.startswith("+"):
            cleaned_phone = "+" + cleaned_phone
        jwt_phone = (current_user.get("phone") or "").strip()
        if not jwt_phone:
            profile_data["phone"] = cleaned_phone

    if age is not None and age.strip():
        try:
            patient_data["age"] = int(age)
        except ValueError:
            pass
    if gender is not None and gender.strip():
        patient_data["gender"] = gender.strip()
    if blood_group is not None and blood_group.strip():
        patient_data["blood_group"] = blood_group.strip()
    if height_cm is not None and height_cm.strip():
        try:
            patient_data["height_cm"] = float(height_cm)
        except ValueError:
            pass
    if weight_kg is not None and weight_kg.strip():
        try:
            patient_data["weight_kg"] = float(weight_kg)
        except ValueError:
            pass
    if address is not None and address.strip():
        patient_data["address"] = address.strip()
    if emergency_contact is not None and emergency_contact.strip():
        patient_data["emergency_contact_phone"] = emergency_contact.strip()
    if emergency_contact_name is not None and emergency_contact_name.strip():
        patient_data["emergency_contact_name"] = emergency_contact_name.strip()
    if emergency_contact_phone is not None and emergency_contact_phone.strip():
        patient_data["emergency_contact_phone"] = emergency_contact_phone.strip()
    if emergency_contact_relation is not None and emergency_contact_relation.strip():
        patient_data["emergency_contact_relation"] = emergency_contact_relation.strip()
    if medical_history is not None and medical_history.strip():
        patient_data["medical_history"] = medical_history.strip()
    if current_medications is not None and current_medications.strip():
        patient_data["current_medications"] = current_medications.strip()
    if allergies is not None and allergies.strip():
        patient_data["allergies"] = allergies.strip()
    if chronic_conditions is not None and chronic_conditions.strip():
        patient_data["chronic_conditions"] = chronic_conditions.strip()
    for flag_key, flag_val in [
        ("is_smoker", is_smoker), ("is_alcohol_user", is_alcohol_user),
        ("is_pregnant", is_pregnant), ("is_nursing", is_nursing),
    ]:
        if flag_val is not None:
            patient_data[flag_key] = flag_val.lower() == "true"

    if not profile_data and not patient_data:
        raise HTTPException(status_code=400, detail="No fields to update")

    # profiles.id == Supabase auth UID after migration — direct lookup only.
    actual_profile_id = user_id
    existing_profile = db.get_profile_by_id(user_id)
    if not existing_profile:
        raise HTTPException(status_code=404, detail="Profile not found.")

    if profile_data:
        # If email is taken by a ghost placeholder profile, clear it first.
        if "email" in profile_data:
            conflict = (
                db.client.table("profiles")
                .select("id,phone")
                .eq("email", profile_data["email"])
                .neq("id", actual_profile_id)
                .limit(1)
                .execute()
            )
            if conflict.data:
                holder = conflict.data[0]
                if (holder.get("phone") or "").startswith("supabase_"):
                    db.client.table("profiles").update({"email": None}).eq("id", holder["id"]).execute()
                else:
                    profile_data.pop("email")
        db.update_profile(actual_profile_id, profile_data)

    if patient_data:
        patient = db.get_patient_by_profile_id(actual_profile_id)
        if patient:
            db.client.table("patients").update(patient_data).eq("id", patient["id"]).execute()
        else:
            patient_data["profile_id"] = actual_profile_id
            db.client.table("patients").insert(patient_data).execute()

    updated_profile = db.get_profile_by_id(actual_profile_id)
    if updated_profile:
        updated_profile.pop("password_hash", None)
    updated_patient = db.get_patient_by_profile_id(actual_profile_id)

    return {
        "message": "Profile updated",
        "user": _merge_user(updated_profile, updated_patient),
    }
