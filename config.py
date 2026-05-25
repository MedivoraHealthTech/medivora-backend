"""
Medivora Backend Configuration
Loads settings from environment variables.
Compatible with both main.py (modular) and api.py (chat agent).
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Supabase
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")  # service_role key
    SUPABASE_ANON_KEY: str = os.getenv("SUPABASE_ANON_KEY", "")

    # JWT
    JWT_SECRET: str = os.getenv("JWT_SECRET", "CHANGE-ME-use-openssl-rand-hex-32")
    JWT_ALGORITHM: str = "HS256"
    # Optional — only needed if your Supabase project still uses Legacy HS256 signing.
    # For ECC (P-256) projects (the current default) the backend auto-fetches the
    # public JWKS from Supabase using SUPABASE_URL, so no secret is required.
    SUPABASE_JWT_SECRET: str = os.getenv("SUPABASE_JWT_SECRET", "")
    JWT_EXPIRY_HOURS_PATIENT: int = int(os.getenv("JWT_EXPIRY_HOURS_PATIENT", "24"))
    JWT_EXPIRY_HOURS_DOCTOR: int = int(os.getenv("JWT_EXPIRY_HOURS_DOCTOR", "12"))
    JWT_EXPIRY_HOURS_ADMIN: int = int(os.getenv("JWT_EXPIRY_HOURS_ADMIN", "4"))

    # Server
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    ALLOWED_ORIGINS: list = [
        o.strip()
        for o in os.getenv(
            "ALLOWED_ORIGINS",
            "http://localhost:5173,http://localhost:5174,http://localhost:3000",
        ).split(",")
    ]

    # Security
    MAX_LOGIN_ATTEMPTS: int = int(os.getenv("MAX_LOGIN_ATTEMPTS", "5"))
    LOCKOUT_WINDOW_MINUTES: int = int(os.getenv("LOCKOUT_WINDOW_MINUTES", "15"))

    # OTP
    OTP_TTL_MINUTES: int = int(os.getenv("OTP_TTL_MINUTES", "1"))
    OTP_MOCK_MODE: bool = os.getenv("OTP_MOCK_MODE", "true").lower() == "true"

    # MSG91 (required when OTP_MOCK_MODE=false)
    MSG91_AUTH_KEY: str = os.getenv("MSG91_AUTH_KEY", "")
    MSG91_SENDER_ID: str = os.getenv("MSG91_SENDER_ID", "MEDVRA")
    MSG91_OTP_TEMPLATE_ID: str = os.getenv("MSG91_OTP_TEMPLATE_ID", "")
    MSG91_ALERT_TEMPLATE_ID: str = os.getenv("MSG91_ALERT_TEMPLATE_ID", "")

    # Razorpay
    RAZORPAY_KEY_ID: str = os.getenv("RAZORPAY_KEY_ID", "")
    RAZORPAY_KEY_SECRET: str = os.getenv("RAZORPAY_KEY_SECRET", "")

    # ── api.py (chat agent) compatibility fields ──────────────────
    # These use lowercase names to match what api.py expects

    @property
    def port(self):
        return self.PORT

    @property
    def debug_mode(self):
        return self.DEBUG

    @property
    def max_message_length(self):
        return int(os.getenv("MAX_MESSAGE_LENGTH", "5000"))

    @property
    def max_file_size_mb(self):
        return int(os.getenv("MAX_FILE_SIZE_MB", "10"))

    @property
    def verification_base_url(self):
        return os.getenv("VERIFICATION_BASE_URL", "https://your-api.onrender.com/verify")

    @property
    def backend_base_url(self):
        return os.getenv("BACKEND_BASE_URL", "https://medivora-backend-production.up.railway.app")


# Alias so `from config import Config` works (api.py imports Config)
Config = Settings

settings = Settings()
