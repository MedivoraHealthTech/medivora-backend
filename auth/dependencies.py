"""
FastAPI Auth Dependencies — Inject current user into route handlers.
"""

from typing import Dict, Optional

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from jose.exceptions import JWKError

from auth.jwt_handler import decode_token

security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Dict:
    """Extract and validate JWT from Authorization header.
    Returns decoded payload with 'sub' (user_id) and 'role'."""
    if credentials is None:
        raise HTTPException(
            status_code=401, detail="Not authenticated. Provide a Bearer token."
        )
    try:
        payload = decode_token(credentials.credentials)
    except (JWTError, JWKError):
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
    return payload


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[Dict]:
    """Like get_current_user but returns None for unauthenticated requests."""
    if credentials is None:
        return None
    try:
        return decode_token(credentials.credentials)
    except (JWTError, JWKError):
        return None


async def require_patient(
    current_user: Dict = Depends(get_current_user),
) -> Dict:
    if current_user.get("role") != "patient":
        raise HTTPException(status_code=403, detail="Patient access required.")
    return current_user


async def require_doctor(
    current_user: Dict = Depends(get_current_user),
) -> Dict:
    if current_user.get("role") != "doctor":
        raise HTTPException(status_code=403, detail="Doctor access required.")
    return current_user


async def require_admin(
    current_user: Dict = Depends(get_current_user),
) -> Dict:
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required.")
    return current_user
