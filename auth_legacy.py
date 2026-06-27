"""
Medivora Authentication Utilities — Production Grade
- JWT token creation & validation with role-specific expiry
- bcrypt password hashing with strength enforcement
- FastAPI auth dependencies for patient, doctor, and admin
- Account lockout support (see database for attempt tracking)
"""

import os
import re
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict

import bcrypt
import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# ---------- Config ----------
JWT_SECRET = os.getenv("JWT_SECRET", "CHANGE-ME-in-production-use-openssl-rand-hex-32")
JWT_ALGORITHM = "HS256"

# Role-specific expiry: admin sessions are shorter for security
EXPIRY_BY_ROLE = {
    "admin":   int(os.getenv("ADMIN_JWT_EXPIRY_HOURS",   "4")),   # 4 hours
    "doctor":  int(os.getenv("DOCTOR_JWT_EXPIRY_HOURS", "12")),   # 12 hours
    "patient": int(os.getenv("PATIENT_JWT_EXPIRY_HOURS", "24")),  # 24 hours
}

# Legacy fallback
JWT_EXPIRY_HOURS = int(os.getenv("JWT_EXPIRY_HOURS", "24"))

security = HTTPBearer(auto_error=False)

# ---------- Password validation ----------

PASSWORD_POLICY = {
    "admin":   {"min_length": 10, "require_upper": True, "require_digit": True, "require_special": True},
    "doctor":  {"min_length": 8,  "require_upper": False, "require_digit": True, "require_special": False},
    "patient": {"min_length": 6,  "require_upper": False, "require_digit": False, "require_special": False},
}


def validate_password_strength(password: str, role: str = "patient") -> tuple[bool, str]:
    """
    Validate password meets policy for the given role.
    Returns (is_valid, error_message). error_message is empty if valid.
    """
    policy = PASSWORD_POLICY.get(role, PASSWORD_POLICY["patient"])
    if len(password) < policy["min_length"]:
        return False, f"Password must be at least {policy['min_length']} characters"
    if policy["require_upper"] and not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter"
    if policy["require_digit"] and not re.search(r"\d", password):
        return False, "Password must contain at least one number"
    if policy["require_special"] and not re.search(r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?]", password):
        return False, "Password must contain at least one special character (!@#$%^&*...)"
    return True, ""


# ---------- Password hashing (bcrypt) ----------

def hash_password(password: str) -> str:
    """Hash password with bcrypt. Returns string with embedded salt."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    """Verify a plaintext password against a bcrypt hash."""
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


def verify_legacy_sha256(password: str, salt: str, stored_hash: str) -> bool:
    """Verify against the old SHA-256 scheme for migration."""
    return hashlib.sha256((password + salt).encode()).hexdigest() == stored_hash


# ---------- JWT ----------

def create_token(user_id: str, role: str = "patient", extra: Dict = None) -> str:
    """
    Create a JWT with user_id, role, and optional extra claims.
    Token expiry is determined by role (admin < doctor < patient).
    """
    now = datetime.now(timezone.utc)
    expiry_hours = EXPIRY_BY_ROLE.get(role, JWT_EXPIRY_HOURS)
    payload = {
        "sub": user_id,
        "role": role,
        "iat": now,
        "exp": now + timedelta(hours=expiry_hours),
    }
    if extra:
        payload.update(extra)
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> Dict:
    """Decode and validate a JWT. Raises HTTPException on failure."""
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired. Please log in again.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token. Please log in again.")


# ---------- FastAPI Dependencies ----------

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Dict:
    """Dependency: extract and validate JWT from Authorization header.
    Returns decoded payload with at least 'sub' and 'role'."""
    if credentials is None:
        raise HTTPException(status_code=401, detail="Not authenticated. Please provide a Bearer token.")
    return decode_token(credentials.credentials)


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[Dict]:
    """Like get_current_user but returns None for unauthenticated requests."""
    if credentials is None:
        return None
    try:
        return decode_token(credentials.credentials)
    except HTTPException:
        return None


async def require_doctor(
    current_user: Dict = Depends(get_current_user),
) -> Dict:
    """Dependency: requires the caller to have doctor role."""
    if current_user.get("role") != "doctor":
        raise HTTPException(status_code=403, detail="Doctor access required. Please log in as a doctor.")
    return current_user


async def require_admin(
    current_user: Dict = Depends(get_current_user),
) -> Dict:
    """Dependency: requires the caller to have admin role."""
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required. Unauthorized.")
    return current_user


async def require_patient(
    current_user: Dict = Depends(get_current_user),
) -> Dict:
    """Dependency: requires the caller to have patient role."""
    if current_user.get("role") != "patient":
        raise HTTPException(status_code=403, detail="Patient access required.")
    return current_user
