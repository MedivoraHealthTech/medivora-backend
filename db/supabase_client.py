"""
Supabase Database Client — All DB operations for Medivora.
Uses the service_role key for full access (RLS bypassed).
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from supabase import create_client, Client

from config import settings


class SupabaseDB:
    """Wrapper around supabase-py with typed methods for each table."""

    def __init__(self):
        if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_KEY must be set in environment"
            )
        self.client: Client = create_client(
            settings.SUPABASE_URL, settings.SUPABASE_KEY
        )

    # ── PROFILES ──────────────────────────────────────────────────────

    def create_profile(
        self,
        phone: str,
        password_hash: str,
        user_type: str = "patient",
        email: Optional[str] = None,
        full_name: str = "",
        first_name: str = "",
        last_name: str = "",
    ) -> Dict[str, Any]:
        # Support both legacy full_name and new first_name/last_name params
        if full_name and not first_name and not last_name:
            _parts = full_name.strip().split(" ", 1)
            first_name = _parts[0]
            last_name = _parts[1] if len(_parts) > 1 else ""
        data = {
            "phone": phone,
            "first_name": first_name,
            "last_name": last_name,
            "password_hash": password_hash,
            "user_type": user_type,
            "role": user_type,
        }
        if email:
            data["email"] = email
        result = self.client.table("profiles").insert(data).execute()
        return result.data[0]

    def get_profile_by_phone(self, phone: str) -> Optional[Dict[str, Any]]:
        result = (
            self.client.table("profiles")
            .select("*")
            .eq("phone", phone)
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None

    def get_profile_by_id(self, profile_id: str) -> Optional[Dict[str, Any]]:
        result = (
            self.client.table("profiles")
            .select("*")
            .eq("id", profile_id)
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None

    def update_profile(self, profile_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        result = (
            self.client.table("profiles").update(data).eq("id", profile_id).execute()
        )
        return result.data[0] if result.data else {}

    def update_last_login(self, profile_id: str):
        self.client.table("profiles").update(
            {"last_login": datetime.now(timezone.utc).isoformat()}
        ).eq("id", profile_id).execute()

    # ── PATIENTS ──────────────────────────────────────────────────────

    def create_patient(self, profile_id: str, **kwargs) -> Dict[str, Any]:
        data = {"profile_id": profile_id, **kwargs}
        result = self.client.table("patients").insert(data).execute()
        return result.data[0]

    def get_patient_by_profile_id(self, profile_id: str) -> Optional[Dict[str, Any]]:
        result = (
            self.client.table("patients")
            .select("*")
            .eq("profile_id", profile_id)
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None

    def update_patient(self, patient_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        # Remove None values so we don't overwrite with nulls
        clean = {k: v for k, v in data.items() if v is not None}
        if not clean:
            return {}
        result = (
            self.client.table("patients").update(clean).eq("id", patient_id).execute()
        )
        return result.data[0] if result.data else {}

    # ── DOCTORS ───────────────────────────────────────────────────────

    def create_doctor(self, profile_id: str, **kwargs) -> Dict[str, Any]:
        data = {"profile_id": profile_id, **kwargs}
        result = self.client.table("doctors").insert(data).execute()
        return result.data[0]

    def get_doctor_by_profile_id(self, profile_id: str) -> Optional[Dict[str, Any]]:
        result = (
            self.client.table("doctors")
            .select("*")
            .eq("profile_id", profile_id)
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None

    # ── MEDICAL RECORDS ───────────────────────────────────────────────

    def create_medical_record(
        self, patient_id: str, **kwargs
    ) -> Dict[str, Any]:
        data = {"patient_id": patient_id, **kwargs}
        result = self.client.table("medical_records").insert(data).execute()
        return result.data[0]

    def get_medical_records(
        self, patient_id: str, limit: int = 50, offset: int = 0
    ) -> List[Dict[str, Any]]:
        result = (
            self.client.table("medical_records")
            .select("*")
            .eq("patient_id", patient_id)
            .order("created_at", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )
        return result.data

    def get_medical_record_by_id(self, record_id: str) -> Optional[Dict[str, Any]]:
        result = (
            self.client.table("medical_records")
            .select("*")
            .eq("id", record_id)
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None

    # ── OTP TOKENS ────────────────────────────────────────────────────

    def create_otp(self, phone: str, otp_code: str, ttl_minutes: int = 10) -> Dict:
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=ttl_minutes)
        data = {
            "phone": phone,
            "otp_code": otp_code,
            "expires_at": expires_at.isoformat(),
        }
        result = self.client.table("otp_tokens").insert(data).execute()
        return result.data[0]

    def verify_otp(self, phone: str, otp_code: str) -> bool:
        """Check OTP is valid and not expired, then mark as used."""
        now = datetime.now(timezone.utc).isoformat()
        result = (
            self.client.table("otp_tokens")
            .select("*")
            .eq("phone", phone)
            .eq("otp_code", otp_code)
            .eq("is_used", False)
            .gte("expires_at", now)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if not result.data:
            return False

        # Mark as used
        self.client.table("otp_tokens").update({"is_used": True}).eq(
            "id", result.data[0]["id"]
        ).execute()
        return True

    # ── LOGIN ATTEMPTS ────────────────────────────────────────────────

    def record_login_attempt(
        self,
        phone: str,
        user_type: str,
        success: bool,
        ip_address: str = "",
    ):
        self.client.table("login_attempts").insert(
            {
                "phone": phone,
                "user_type": user_type,
                "success": success,
                "ip_address": ip_address,
            }
        ).execute()

    def count_failed_attempts(
        self, phone: str, window_minutes: int = 15
    ) -> int:
        since = (
            datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
        ).isoformat()
        result = (
            self.client.table("login_attempts")
            .select("id", count="exact")
            .eq("phone", phone)
            .eq("success", False)
            .gte("created_at", since)
            .execute()
        )
        return result.count or 0


# ── Singleton ─────────────────────────────────────────────────────────

_db_instance: Optional[SupabaseDB] = None


def get_db() -> SupabaseDB:
    global _db_instance
    if _db_instance is None:
        _db_instance = SupabaseDB()
    return _db_instance
