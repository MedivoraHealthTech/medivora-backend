"""
Auth package — re-exports for both main.py (modular) and api.py (legacy) imports.
"""

from auth.jwt_handler import create_token, decode_token
from auth.password_handler import hash_password, verify_password, validate_password_strength
from auth.dependencies import (
    get_current_user,
    get_current_user_optional,
    require_doctor,
    require_admin,
    require_patient,
)

__all__ = [
    "create_token",
    "decode_token",
    "hash_password",
    "verify_password",
    "validate_password_strength",
    "get_current_user",
    "get_current_user_optional",
    "require_doctor",
    "require_admin",
    "require_patient",
]
