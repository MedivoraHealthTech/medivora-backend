"""
JWT Token Creation & Validation for Medivora.

Handles three token types:
  1. Custom HS256 tokens  — issued by this backend for doctors / admins
  2. Supabase HS256 tokens — legacy, used by older Supabase projects
  3. Supabase ES256 tokens — current default (ECC P-256), validated via JWKS
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import httpx
from jose import JWTError, jwk, jwt
from jose.exceptions import JWKError

from config import settings

logger = logging.getLogger(__name__)

# Role → expiry hours mapping
_EXPIRY_MAP = {
    "patient": settings.JWT_EXPIRY_HOURS_PATIENT,
    "doctor": settings.JWT_EXPIRY_HOURS_DOCTOR,
    "admin": settings.JWT_EXPIRY_HOURS_ADMIN,
}

# Module-level JWKS cache — populated on first Supabase ES256 verification attempt
_jwks_cache: Optional[List[Dict]] = None


def _get_supabase_jwks() -> List[Dict]:
    """Fetch and cache Supabase's public JWKS (called at most once per process)."""
    global _jwks_cache
    if _jwks_cache is not None:
        return _jwks_cache

    supabase_url = getattr(settings, "SUPABASE_URL", "").rstrip("/")
    if not supabase_url:
        return []

    try:
        url = f"{supabase_url}/auth/v1/.well-known/jwks.json"
        resp = httpx.get(url, timeout=10)
        resp.raise_for_status()
        _jwks_cache = resp.json().get("keys", [])
        logger.info(f"Supabase JWKS loaded: {len(_jwks_cache)} key(s)")
    except Exception as exc:
        logger.warning(f"Failed to fetch Supabase JWKS from {supabase_url}: {exc}")
        _jwks_cache = []

    return _jwks_cache


def _decode_supabase_token(token: str) -> Dict:
    """Try to verify a Supabase-issued token.

    Attempts in order:
      1. Legacy HS256 shared secret (SUPABASE_JWT_SECRET env var)
      2. Current ECC P-256 via JWKS (fetched from Supabase's well-known endpoint)

    Raises JWTError if neither method succeeds.
    """
    # ── 1. Legacy HS256 secret ─────────────────────────────────────
    supabase_secret = getattr(settings, "SUPABASE_JWT_SECRET", "")
    if supabase_secret:
        try:
            return jwt.decode(
                token,
                supabase_secret,
                algorithms=["HS256"],
                options={"verify_aud": False},
            )
        except JWTError:
            pass  # not HS256 or wrong secret — fall through to JWKS

    # ── 2. Current ECC P-256 via JWKS ─────────────────────────────
    jwks_keys = _get_supabase_jwks()
    if not jwks_keys:
        raise JWTError("No Supabase JWKS keys available and no SUPABASE_JWT_SECRET set.")

    # Identify the right key using the `kid` in the token header
    try:
        header = jwt.get_unverified_header(token)
    except JWTError:
        raise JWTError("Cannot parse JWT header.")

    kid = header.get("kid")
    alg = header.get("alg", "ES256")

    last_exc: Exception = JWTError("No matching JWKS key found.")
    for key_data in jwks_keys:
        # Skip keys that don't match kid (if present)
        if kid and key_data.get("kid") and key_data["kid"] != kid:
            continue
        try:
            public_key = jwk.construct(key_data, algorithm=alg)
            return jwt.decode(
                token,
                public_key,
                algorithms=[alg],
                options={"verify_aud": False},
            )
        except (JWTError, JWKError) as exc:
            last_exc = exc
            continue

    raise last_exc


def _normalise_supabase_payload(payload: Dict) -> Dict:
    """Map Supabase role 'authenticated' → role from user_metadata (or 'patient') and surface user metadata."""
    meta = payload.get("user_metadata") or {}
    meta_role = meta.get("role", "")

    if payload.get("role") == "authenticated":
        payload = dict(payload)  # don't mutate the original
        # Use role from user_metadata if explicitly set, otherwise default to patient
        if meta_role in ("patient", "doctor", "admin"):
            payload["role"] = meta_role
        else:
            payload["role"] = "patient"

    # Hoist user_metadata fields to top level for easy access in route handlers
    if not payload.get("name"):
        payload["name"] = (
            meta.get("full_name")
            or meta.get("name")
            or payload.get("email", "")
        )
    if not payload.get("email"):
        payload["email"] = meta.get("email", "")

    return payload


def create_token(
    user_id: str,
    role: str = "patient",
    extra: Optional[Dict] = None,
) -> str:
    """Create a signed JWT with user_id, role, and optional extra claims."""
    now = datetime.now(timezone.utc)
    expiry_hours = _EXPIRY_MAP.get(role, 24)
    payload = {
        "sub": user_id,
        "role": role,
        "iat": now,
        "exp": now + timedelta(hours=expiry_hours),
    }
    if extra:
        payload.update(extra)
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str) -> Dict:
    """Decode and validate a JWT.

    Priority:
      1. Custom HS256 secret (doctor / admin tokens issued by this backend)
      2. Supabase token via _decode_supabase_token (HS256 legacy or ES256 JWKS)

    Raises JWTError if all methods fail.
    """
    # ── 1. Custom backend token ────────────────────────────────────
    try:
        return jwt.decode(
            token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM]
        )
    except (JWTError, JWKError):
        pass

    # ── 2. Supabase token (HS256 legacy or ES256 JWKS) ─────────────
    # Only attempt if Supabase is configured at all
    supabase_url = getattr(settings, "SUPABASE_URL", "")
    supabase_secret = getattr(settings, "SUPABASE_JWT_SECRET", "")
    if not supabase_url and not supabase_secret:
        raise JWTError("Invalid or expired token.")

    try:
        payload = _decode_supabase_token(token)
    except (JWTError, JWKError, Exception):
        raise JWTError("Invalid or expired token.")
    return _normalise_supabase_payload(payload)
