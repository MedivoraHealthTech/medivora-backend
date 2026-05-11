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

# ── Private Network Access (PNA) middleware ────────────────────────────
# Chrome 94+ enforces Private Network Access (PNA): when a page on localhost
# requests a loopback address, Chrome requires Access-Control-Allow-Private-Network
# in preflight responses. Starlette's CORSMiddleware doesn't support this and
# returns 400 when it sees Access-Control-Request-Private-Network: true.
# This middleware handles both cases: explicit PNA preflights and regular
# preflights where Chrome still expects the header.

@app.middleware("http")
async def private_network_access_middleware(request: Request, call_next):
    if request.method == "OPTIONS":
        origin = request.headers.get("origin", "")
        from fastapi.responses import Response as FResponse
        response = FResponse(status_code=200)
        response.headers["Access-Control-Allow-Private-Network"] = "true"
        response.headers["Access-Control-Allow-Origin"] = origin or "*"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = request.headers.get(
            "access-control-request-headers", "authorization, content-type"
        )
        response.headers["Access-Control-Max-Age"] = "600"
        return response
    response = await call_next(request)
    return response

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

    # ── 1. Resolve profile (profiles.id == Supabase auth UID after migration) ──
    profile = db.get_profile_by_id(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Patient profile not found. Please log in again.")

    # ── 2. Resolve patient row ─────────────────────────────────────────────
    patient = db.get_patient_by_profile_id(user_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient record not found.")

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
            "user_id":           user_id,
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

