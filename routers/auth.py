"""
Auth Router — Signup, Login, OTP endpoints.
"""

import random
import string

from fastapi import APIRouter, Depends, Form, HTTPException, Request

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


def _generate_otp(length: int = 6) -> str:
    """Generate a random numeric OTP."""
    return "".join(random.choices(string.digits, k=length))


def _get_client_ip(request: Request) -> str:
    """Extract client IP from request."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else ""


# ── POST /auth/signup ─────────────────────────────────────────────────

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
    """Login with phone + password."""
    db = get_db()
    ip = _get_client_ip(request)

    # Check lockout
    failed_count = db.count_failed_attempts(
        req.phone, settings.LOCKOUT_WINDOW_MINUTES
    )
    if failed_count >= settings.MAX_LOGIN_ATTEMPTS:
        raise HTTPException(
            status_code=429,
            detail=f"Too many failed attempts. Try again in {settings.LOCKOUT_WINDOW_MINUTES} minutes.",
        )

    # Find user
    profile = db.get_profile_by_phone(req.phone)
    if not profile:
        db.record_login_attempt(req.phone, "unknown", False, ip)
        raise HTTPException(status_code=401, detail="Invalid phone or password.")

    # Verify password
    if not verify_password(req.password, profile["password_hash"]):
        db.record_login_attempt(req.phone, profile["user_type"], False, ip)
        raise HTTPException(status_code=401, detail="Invalid phone or password.")

    # Check account status
    if profile["status"] != "active":
        raise HTTPException(
            status_code=403, detail=f"Account is {profile['status']}."
        )

    # Success
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
    """Send a 6-digit OTP to the given phone number."""
    db = get_db()

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

    # Production — send via Twilio
    if not settings.TWILIO_ACCOUNT_SID or not settings.TWILIO_AUTH_TOKEN or not settings.TWILIO_FROM_NUMBER:
        raise HTTPException(
            status_code=500,
            detail="SMS service not configured. Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_FROM_NUMBER.",
        )

    try:
        from twilio.rest import Client as TwilioClient
        client = TwilioClient(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
        client.messages.create(
            body=f"Your Medivora OTP is: {otp}. Valid for {settings.OTP_TTL_MINUTES} minutes. Do not share this code.",
            from_=settings.TWILIO_FROM_NUMBER,
            to=req.phone,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send OTP via SMS: {str(e)}")

    return {"message": "OTP sent successfully", "phone": req.phone}


# ── POST /auth/verify-otp ────────────────────────────────────────────

@router.post("/verify-otp")
async def verify_otp(req: VerifyOTPRequest):
    """Verify OTP and auto-create/login the user."""
    db = get_db()

    # In mock mode, accept any OTP
    if settings.OTP_MOCK_MODE:
        is_valid = True
    else:
        is_valid = db.verify_otp(phone=req.phone, otp_code=req.otp)

    if not is_valid:
        raise HTTPException(status_code=400, detail="Invalid or expired OTP.")

    # Check if user already exists
    existing = db.get_profile_by_phone(req.phone)
    is_new_user = existing is None

    if is_new_user:
        # New user — create profile + patient record
        name = (req.name or "").strip()
        _otp_parts = name.split(" ", 1) if name else ["", ""]
        # Create with a placeholder password (user can set real one later)
        placeholder_hash = hash_password("otp_verified_no_password")
        profile = db.create_profile(
            phone=req.phone,
            first_name=_otp_parts[0],
            last_name=_otp_parts[1] if len(_otp_parts) > 1 else "",
            password_hash=placeholder_hash,
            user_type="patient",
        )
        db.create_patient(profile_id=profile["id"])
    else:
        profile = existing

    # Update phone_verified
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

    # Resolve profile: Supabase UID may differ from our profiles.id
    profile = db.get_profile_by_id(user_id)
    if not profile:
        jwt_phone = (current_user.get("phone") or "").strip()
        if jwt_phone:
            if not jwt_phone.startswith("+"):
                jwt_phone = "+" + jwt_phone
            profile = db.get_profile_by_phone(jwt_phone)
        if not profile:
            # Auto-create minimal profile for first-time Supabase auth users
            import secrets, bcrypt  # noqa: E401
            pw_hash = bcrypt.hashpw(secrets.token_hex(16).encode(), bcrypt.gensalt()).decode()
            email = (current_user.get("email") or "").strip()
            try:
                result = db.client.table("profiles").insert({
                    "id":            user_id,
                    "first_name":    "",
                    "last_name":     "",
                    "phone":         jwt_phone or f"supabase_{user_id[:8]}",
                    "email":         email or None,
                    "password_hash": pw_hash,
                    "user_type":     "patient",
                    "role":          "patient",
                }).execute()
                profile = result.data[0] if result.data else None
            except Exception:
                pass

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
    email: str = Form(default=None),
    phone: str = Form(default=None),
    age: str = Form(default=None),
    gender: str = Form(default=None),
    blood_group: str = Form(default=None),
    address: str = Form(default=None),
    emergency_contact: str = Form(default=None),
    medical_history: str = Form(default=None),
    current_medications: str = Form(default=None),
    allergies: str = Form(default=None),
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
    if address is not None and address.strip():
        patient_data["address"] = address.strip()
    if emergency_contact is not None and emergency_contact.strip():
        patient_data["emergency_contact_phone"] = emergency_contact.strip()
    if medical_history is not None and medical_history.strip():
        patient_data["medical_history"] = medical_history.strip()
    if current_medications is not None and current_medications.strip():
        patient_data["current_medications"] = current_medications.strip()
    if allergies is not None and allergies.strip():
        patient_data["allergies"] = allergies.strip()

    if not profile_data and not patient_data:
        raise HTTPException(status_code=400, detail="No fields to update")

    # Resolve actual profiles row. Supabase OTP users have a different UID than
    # the UUID in our profiles table — fall back to phone lookup when needed.
    actual_profile_id = user_id
    existing_profile = db.get_profile_by_id(user_id)
    if not existing_profile:
        jwt_phone = (current_user.get("phone") or "").strip()
        if jwt_phone:
            if not jwt_phone.startswith("+"):
                jwt_phone = "+" + jwt_phone
            phone_profile = db.get_profile_by_phone(jwt_phone)
            if phone_profile:
                actual_profile_id = phone_profile["id"]
                existing_profile = phone_profile
        if not existing_profile:
            jwt_phone = jwt_phone or f"+placeholder_{user_id[:8]}"
            _placeholder_parts = ((full_name or "").strip()).split(" ", 1) if (full_name or "").strip() else ["", ""]
            db.client.table("profiles").insert({
                "id": user_id,
                "phone": jwt_phone,
                "first_name": _placeholder_parts[0],
                "last_name": _placeholder_parts[1] if len(_placeholder_parts) > 1 else "",
                "password_hash": "supabase_managed",
                "user_type": "patient",
                "role": "patient",
                "status": "active",
            }).execute()
            profile_data.pop("first_name", None)
            profile_data.pop("last_name", None)

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
