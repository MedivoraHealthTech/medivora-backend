"""
Medivora API — Main Entry Point
FastAPI backend with Supabase (Postgres) database.

Run: uvicorn main:app --reload --port 8000
"""

from contextlib import asynccontextmanager
import logging

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
