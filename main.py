"""
# v2
Medivora API — Main Entry Point
FastAPI backend with Supabase (Postgres) database.

Run: uvicorn main:app --reload --port 8000
"""

from contextlib import asynccontextmanager
import logging
import uuid as _uuid

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from config import settings
from routers import auth, patients, health, consultations, doctors, faqs
from auth.dependencies import get_current_user
from db import get_db

# ── Logging ───────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("medivora")


# ── Lifespan ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Medivora API v2.0.0 (Supabase)")
    # Test DB connection on startup
    try:
        from db import get_db
        db = get_db()
        logger.info("Supabase connection established")
    except Exception as e:
        logger.error(f"Supabase connection failed: {e}")
    yield
    logger.info("Shutting down Medivora API")


# ── FastAPI App ───────────────────────────────────────────────────────

app = FastAPI(
    title="Medivora Medical AI Assistant API",
    version="2.0.0",
    description="Healthcare triage platform powered by Supabase",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────

app.include_router(health.router)
app.include_router(auth.router)
app.include_router(patients.router)
app.include_router(consultations.router)
app.include_router(doctors.router)
app.include_router(faqs.router)

# ── Rate-limit handler (covers legacy routes that use @limiter.limit) ─────────
# Without this, RateLimitExceeded escapes past CORSMiddleware → 500 without
# CORS headers → browser reports it as a "CORS error".

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse({"detail": "Rate limit exceeded. Please try again later."}, status_code=429)


# ── POST /payment/dev-confirm ─────────────────────────────────────────────────

@app.post("/payment/dev-confirm")
async def dev_confirm_booking(
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """Dev-only: create a consultation without going through Razorpay payment.
    Ensures profile + patient rows exist for the Supabase auth user first."""
    if not settings.DEBUG:
        raise HTTPException(status_code=403, detail="Not available in production.")

    body = await request.json()
    doctor_id         = body.get("doctor_id", "")
    specialty         = body.get("specialty", "general_medicine")
    patient_note      = body.get("patient_note", "")
    scheduled_at      = body.get("scheduled_at") or None
    consultation_type = body.get("consultation_type", "in_person")

    user_id = current_user["sub"]
    db = get_db()

    # ── 1. Ensure profile row exists ─────────────────────────────────────
    profile = db.get_profile_by_id(user_id)
    if not profile:
        # Try by phone (Supabase OTP users)
        jwt_phone = (current_user.get("phone") or "").strip()
        if jwt_phone and not jwt_phone.startswith("+"):
            jwt_phone = "+" + jwt_phone
        if jwt_phone:
            profile = db.get_profile_by_phone(jwt_phone)
        if not profile:
            # Auto-create a minimal profile row keyed by the Supabase UID
            import secrets, bcrypt as _bcrypt
            pw_hash = _bcrypt.hashpw(secrets.token_hex(16).encode(), _bcrypt.gensalt()).decode()
            try:
                result = db.client.table("profiles").insert({
                    "id":            user_id,
                    "full_name":     "",
                    "first_name":    "",
                    "last_name":     "",
                    "phone":         jwt_phone or f"supabase_{user_id[:8]}",
                    "email":         (current_user.get("email") or None),
                    "password_hash": pw_hash,
                    "user_type":     "patient",
                    "role":          "patient",
                    "status":        "active",
                }).execute()
                profile = result.data[0] if result.data else None
            except Exception as e:
                logger.warning(f"dev_confirm: profile auto-create failed: {e}")
        if not profile:
            raise HTTPException(status_code=500, detail="Could not resolve patient profile.")

    actual_profile_id = profile["id"]

    # ── 2. Ensure patient row exists ──────────────────────────────────────
    patient = db.get_patient_by_profile_id(actual_profile_id)
    if not patient:
        try:
            patient = db.create_patient(profile_id=actual_profile_id)
        except Exception as e:
            logger.warning(f"dev_confirm: patient auto-create failed: {e}")
            raise HTTPException(status_code=500, detail="Could not resolve patient record.")

    patient_id = patient["id"]

    # ── 3. Resolve doctor_id → doctors.id if needed ───────────────────────
    resolved_doctor_id = doctor_id or None
    if doctor_id:
        # Accept both doctors.id directly and profiles.id (convert to doctors.id)
        doc_row = db.client.table("doctors").select("id").eq("id", doctor_id).limit(1).execute()
        if not doc_row.data:
            # Try treating it as a profile_id
            doc_row = db.client.table("doctors").select("id").eq("profile_id", doctor_id).limit(1).execute()
        resolved_doctor_id = doc_row.data[0]["id"] if doc_row.data else None

    # ── 4. Create consultation ────────────────────────────────────────────
    session_id = str(_uuid.uuid4())
    record = {
        "id":                session_id,
        "patient_id":        patient_id,
        "doctor_id":         resolved_doctor_id,
        "specialty":         specialty,
        "patient_note":      patient_note,
        "status":            "scheduled",
        "scheduled_at":      scheduled_at,
        "payment_id":        "dev_skip",
        "consultation_type": consultation_type,
    }
    record = {k: v for k, v in record.items() if v is not None}
    try:
        result = db.client.table("consultations").insert(record).execute()
        if not result.data:
            raise HTTPException(status_code=500, detail="Consultation insert returned no data.")
    except Exception as e:
        logger.error(f"dev_confirm: consultation insert failed: {e}")
        raise HTTPException(status_code=500, detail=f"Could not create consultation: {e}")

    logger.info(f"dev_confirm: created consultation {session_id} for patient {patient_id}")

    # Notify patient
    try:
        specialty_label = specialty.replace("_", " ").title() if specialty else "General Medicine"
        doctor_display = "A doctor"
        if resolved_doctor_id:
            try:
                doc_res = db.client.table("doctors") \
                    .select("*, profiles(first_name, last_name)") \
                    .eq("id", resolved_doctor_id).limit(1).execute()
                if doc_res.data:
                    prof = doc_res.data[0].get("profiles") or {}
                    fn = (prof.get("first_name") or "").strip()
                    ln = (prof.get("last_name") or "").strip()
                    full = f"{fn} {ln}".strip()
                    if full:
                        doctor_display = f"Dr. {full}"
            except Exception:
                pass
        db.client.table("notifications").insert({
            "user_id":           actual_profile_id,
            "notification_type": "consultation",
            "title":             "Consultation Booked",
            "message":           f"Your {specialty_label} consultation is confirmed. {doctor_display} has been assigned.",
            "is_read":           False,
        }).execute()
    except Exception as e:
        logger.warning(f"dev_confirm: notification insert failed: {e}")

    return {"status": "ok", "session_id": session_id}


# ── Legacy routes (api.py) — mounted to cover all not-yet-migrated endpoints ──
# Routes defined in the new routers above take precedence.
from api import app as _legacy_app  # noqa: E402
for route in _legacy_app.routes:
    # Skip routes already registered by the new routers to avoid duplicates.
    # Dedup by (path, methods) so GET and PUT on the same path are both kept.
    existing = {(r.path, frozenset(getattr(r, "methods", None) or [])) for r in app.routes}
    if hasattr(route, "path"):
        key = (route.path, frozenset(getattr(route, "methods", None) or []))
        if key not in existing:
            app.routes.append(route)


# ── DELETE /account ───────────────────────────────────────────────────

@app.delete("/account")
async def delete_account(current_user: dict = Depends(get_current_user)):
    """Permanently delete the authenticated user's account."""
    user_id = current_user["sub"]
    db = get_db()

    # Anonymise profiles row
    try:
        db.update_profile(user_id, {
            "first_name":    "Deleted",
            "last_name":     "Account",
            "phone":         f"deleted_{user_id[:8]}",
            "email":         "",
            "password_hash": "",
            "status":        "deleted",
        })
    except Exception:
        pass

    # Anonymise patients row
    try:
        patient = db.get_patient_by_profile_id(user_id)
        if patient:
            db.client.table("patients").update({
                "medical_history":          None,
                "allergies":                None,
                "current_medications":      None,
                "address":                  None,
                "emergency_contact_name":   None,
                "emergency_contact_phone":  None,
            }).eq("profile_id", user_id).execute()
    except Exception:
        pass

    # Attempt Supabase Auth hard-delete (requires service role key)
    try:
        from supabase import create_client
        service_key = getattr(settings, "SUPABASE_SERVICE_ROLE_KEY", settings.SUPABASE_KEY)
        admin_client = create_client(settings.SUPABASE_URL, service_key)
        admin_client.auth.admin.delete_user(user_id)
        logger.info(f"Supabase auth user {user_id} deleted")
    except Exception as e:
        logger.warning(f"Could not delete Supabase auth user {user_id}: {e}")

    logger.info(f"Account deleted: {user_id}")
    return {"message": "Account deleted successfully"}


# ── Run ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=settings.PORT, reload=settings.DEBUG)

