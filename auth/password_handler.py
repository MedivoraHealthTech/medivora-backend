"""
Password Hashing & Validation for Medivora.
Uses bcrypt for secure password storage.
"""

import re

import bcrypt


def hash_password(password: str) -> str:
    """Hash a plaintext password with bcrypt. Returns string with embedded salt."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    """Verify a plaintext password against a bcrypt hash."""
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


def validate_password_strength(
    password: str, role: str = "patient"
) -> tuple[bool, str]:
    """
    Validate password meets minimum strength for the given role.
    Returns (is_valid, error_message).
    """
    policies = {
        "admin": {"min": 10, "upper": True, "digit": True, "special": True},
        "doctor": {"min": 8, "upper": False, "digit": True, "special": False},
        "patient": {"min": 6, "upper": False, "digit": False, "special": False},
    }
    policy = policies.get(role, policies["patient"])

    if len(password) < policy["min"]:
        return False, f"Password must be at least {policy['min']} characters"
    if policy["upper"] and not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter"
    if policy["digit"] and not re.search(r"\d", password):
        return False, "Password must contain at least one number"
    if policy["special"] and not re.search(
        r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?]", password
    ):
        return False, "Password must contain at least one special character"

    return True, ""
