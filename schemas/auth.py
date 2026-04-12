"""
Auth Request / Response Schemas
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator
import re


class SignupRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    phone: str = Field(..., min_length=10, max_length=20)
    password: str = Field(..., min_length=6, max_length=100)
    user_type: str = Field(default="patient")
    email: Optional[str] = None

    # Doctor-specific (optional, filled during doctor signup)
    nmc_number: Optional[str] = None
    specialties: Optional[list] = None
    experience_years: Optional[int] = None

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, v: str) -> str:
        # Normalize: strip spaces, ensure starts with +91 or is 10 digits
        v = v.strip().replace(" ", "").replace("-", "")
        if v.startswith("+91"):
            v = v  # keep as-is
        elif v.startswith("91") and len(v) == 12:
            v = "+" + v
        elif len(v) == 10 and v.isdigit():
            v = "+91" + v
        # Validate format
        if not re.match(r"^\+91\d{10}$", v):
            raise ValueError("Phone must be a valid Indian number (+91XXXXXXXXXX)")
        return v

    @field_validator("user_type")
    @classmethod
    def validate_user_type(cls, v: str) -> str:
        if v not in ("patient", "doctor"):
            raise ValueError("user_type must be 'patient' or 'doctor'")
        return v


class LoginRequest(BaseModel):
    phone: str = Field(..., min_length=10, max_length=20)
    password: str = Field(..., min_length=1, max_length=100)

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, v: str) -> str:
        v = v.strip().replace(" ", "").replace("-", "")
        if v.startswith("+91"):
            pass
        elif v.startswith("91") and len(v) == 12:
            v = "+" + v
        elif len(v) == 10 and v.isdigit():
            v = "+91" + v
        if not re.match(r"^\+91\d{10}$", v):
            raise ValueError("Phone must be a valid Indian number (+91XXXXXXXXXX)")
        return v


class SendOTPRequest(BaseModel):
    phone: str = Field(..., min_length=10, max_length=20)

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, v: str) -> str:
        v = v.strip().replace(" ", "").replace("-", "")
        if v.startswith("+91"):
            pass
        elif v.startswith("91") and len(v) == 12:
            v = "+" + v
        elif len(v) == 10 and v.isdigit():
            v = "+91" + v
        if not re.match(r"^\+91\d{10}$", v):
            raise ValueError("Phone must be a valid Indian number (+91XXXXXXXXXX)")
        return v


class VerifyOTPRequest(BaseModel):
    phone: str = Field(..., min_length=10, max_length=20)
    otp: str = Field(..., min_length=4, max_length=10)
    name: Optional[str] = Field(default=None, max_length=255)

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, v: str) -> str:
        v = v.strip().replace(" ", "").replace("-", "")
        if v.startswith("+91"):
            pass
        elif v.startswith("91") and len(v) == 12:
            v = "+" + v
        elif len(v) == 10 and v.isdigit():
            v = "+91" + v
        if not re.match(r"^\+91\d{10}$", v):
            raise ValueError("Phone must be a valid Indian number (+91XXXXXXXXXX)")
        return v


class AuthResponse(BaseModel):
    message: str
    user_id: str
    token: str
    user_type: str
    is_new_user: bool = False
