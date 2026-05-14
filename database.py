"""
DatabaseManager — Supabase-backed database operations for api.py.
Wraps the Supabase client to provide the interface api.py expects.
"""

import os
import logging
import random
import string
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from supabase import create_client, Client
from config import settings

logger = logging.getLogger("medivora.database")


def _name(first, last):
    """Compute display name from first_name and last_name."""
    return f"{(first or '').strip()} {(last or '').strip()}".strip()


# Singleton Supabase client
_supabase: Optional[Client] = None


def _get_client() -> Client:
    global _supabase
    if _supabase is None:
        _supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    return _supabase


class DatabaseManager:
    """Async-compatible database manager using Supabase Postgres."""

    def __init__(self):
        self.client = _get_client()

    # ── Helper ─────────────────────────────────────────────────────────

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # ── USER / PROFILE ─────────────────────────────────────────────────

    async def register_user(self, name: str, phone: str, password: str) -> Dict:
        """Register a new patient user."""
        import bcrypt
        pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        phone = phone.strip() if phone else phone
        if phone and not phone.startswith("+"):
            phone = "+" + phone
        name_parts = (name or "").strip().split(" ", 1)
        data = {
            "phone": phone,
            "first_name": name_parts[0],
            "last_name": name_parts[1] if len(name_parts) > 1 else "",
            "password_hash": pw_hash,
            "user_type": "patient",
            "role": "patient",
        }
        result = self.client.table("profiles").insert(data).execute()
        return result.data[0] if result.data else {}

    async def login_user(self, phone: str, password: str) -> Optional[Dict]:
        """Verify patient credentials. Returns user dict or None."""
        import bcrypt
        result = (
            self.client.table("profiles")
            .select("*")
            .eq("phone", phone)
            .eq("user_type", "patient")
            .limit(1)
            .execute()
        )
        if not result.data:
            return None
        user = result.data[0]
        if bcrypt.checkpw(password.encode(), user["password_hash"].encode()):
            self.client.table("profiles").update(
                {"last_login": self._now()}
            ).eq("id", user["id"]).execute()
            return user
        return None

    async def get_user_by_phone(self, phone: str) -> Optional[Dict]:
        result = (
            self.client.table("profiles")
            .select("*")
            .eq("phone", phone)
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None

    async def get_user_by_id(self, user_id: str) -> Optional[Dict]:
        result = (
            self.client.table("profiles")
            .select("*")
            .eq("id", user_id)
            .limit(1)
            .execute()
        )
        if not result.data:
            return None
        profile = result.data[0]
        # Merge patient data if it exists
        try:
            patient_result = (
                self.client.table("patients")
                .select("*")
                .eq("profile_id", user_id)
                .limit(1)
                .execute()
            )
            if patient_result.data:
                patient = patient_result.data[0]
                profile["age"] = patient.get("age")
                profile["gender"] = patient.get("gender")
                profile["blood_group"] = patient.get("blood_group")
                profile["address"] = patient.get("address")
                profile["height_cm"] = patient.get("height_cm")
                profile["weight_kg"] = patient.get("weight_kg")
                profile["emergency_contact"] = patient.get("emergency_contact_phone")
                profile["emergency_contact_name"] = patient.get("emergency_contact_name")
                profile["emergency_contact_phone"] = patient.get("emergency_contact_phone")
                profile["emergency_contact_relation"] = patient.get("emergency_contact_relation")
                profile["medical_history"] = patient.get("medical_history")
                profile["current_medications"] = patient.get("current_medications")
                profile["allergies"] = patient.get("allergies")
                profile["chronic_conditions"] = patient.get("chronic_conditions")
                profile["is_smoker"] = patient.get("is_smoker", False)
                profile["is_alcohol_user"] = patient.get("is_alcohol_user", False)
                profile["is_pregnant"] = patient.get("is_pregnant", False)
                profile["is_nursing"] = patient.get("is_nursing", False)
                profile["patient_id"] = patient.get("id")
        except Exception as e:
            logger.warning(f"get_user_by_id: failed to fetch patient data: {e}")
        return profile

    async def get_all_users(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        result = (
            self.client.table("profiles")
            .select("*")
            .eq("user_type", "patient")
            .range(offset, offset + limit - 1)
            .execute()
        )
        users = result.data or []
        if not users:
            return users
        profile_ids = [u["id"] for u in users]

        # Resolve profile_id → patients.id for all users (one query)
        patients_result = (
            self.client.table("patients")
            .select("id, profile_id")
            .in_("profile_id", profile_ids)
            .execute()
        )
        profile_to_patient: dict = {}
        patient_ids_list = []
        for row in patients_result.data or []:
            profile_to_patient[row["profile_id"]] = row["id"]
            patient_ids_list.append(row["id"])

        # Session counts keyed by patients.id (chat_sessions.patient_id references patients.id)
        session_count_map: dict = {}
        if patient_ids_list:
            sessions_result = (
                self.client.table("chat_sessions")
                .select("patient_id")
                .in_("patient_id", patient_ids_list)
                .execute()
            )
            for s in sessions_result.data or []:
                pid = s["patient_id"]
                session_count_map[pid] = session_count_map.get(pid, 0) + 1

        # Family member counts keyed by patients.id
        family_count_map: dict = {}
        if patient_ids_list:
            family_result = (
                self.client.table("family_members")
                .select("patient_id")
                .in_("patient_id", patient_ids_list)
                .execute()
            )
            for f in family_result.data or []:
                pid = f["patient_id"]
                family_count_map[pid] = family_count_map.get(pid, 0) + 1

        for u in users:
            patient_id = profile_to_patient.get(u["id"])
            u["patient_id"]         = patient_id
            u["session_count"]      = session_count_map.get(patient_id, 0) if patient_id else 0
            u["family_member_count"] = family_count_map.get(patient_id, 0) if patient_id else 0
        return users

    # ── OTP ────────────────────────────────────────────────────────────

    async def create_otp(self, phone: str, otp_code: str, ttl_minutes: int = 1) -> Dict:
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=ttl_minutes)
        data = {
            "phone": phone,
            "otp_code": otp_code,
            "expires_at": expires_at.isoformat(),
        }
        result = self.client.table("otp_tokens").insert(data).execute()
        return result.data[0] if result.data else {}

    async def verify_otp(self, phone: str, otp_code: str) -> bool:
        now = self._now()
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
        self.client.table("otp_tokens").update({"is_used": True}).eq(
            "id", result.data[0]["id"]
        ).execute()
        return True

    # ── LOGIN ATTEMPTS ─────────────────────────────────────────────────

    async def record_login_attempt(self, identifier: str, user_type: str, success: bool, ip: str = ""):
        self.client.table("login_attempts").insert({
            "phone": identifier,
            "user_type": user_type,
            "success": success,
            "ip_address": ip,
        }).execute()

    async def count_failed_attempts(self, identifier: str, user_type: str, window_minutes: int = 15) -> int:
        since = (datetime.now(timezone.utc) - timedelta(minutes=window_minutes)).isoformat()
        result = (
            self.client.table("login_attempts")
            .select("id", count="exact")
            .eq("phone", identifier)
            .eq("success", False)
            .gte("created_at", since)
            .execute()
        )
        return result.count or 0

    async def reset_failed_attempts(self, identifier: str, user_type: str):
        """Delete failed login attempts for this identifier."""
        self.client.table("login_attempts").delete().eq(
            "phone", identifier
        ).eq("success", False).execute()

    # ── PATIENT ID RESOLUTION ─────────────────────────────────────────

    async def _resolve_patient_id(self, profile_id: str) -> Optional[str]:
        """Resolve a profiles.id to the corresponding patients.id.
        If no patient row exists, try to create one automatically."""
        try:
            result = (
                self.client.table("patients")
                .select("id")
                .eq("profile_id", profile_id)
                .limit(1)
                .execute()
            )
            if result.data:
                return result.data[0]["id"]
            # Auto-create patient row for this profile
            insert_result = (
                self.client.table("patients")
                .insert({"profile_id": profile_id})
                .execute()
            )
            if insert_result.data:
                return insert_result.data[0]["id"]
        except Exception as e:
            logger.warning(f"_resolve_patient_id failed for {profile_id}: {e}")
        return None

    # ── CHAT SESSIONS & MESSAGES ───────────────────────────────────────

    async def create_chat_session(self, session_id: str, user_id: str, title: str):
        patient_id = await self._resolve_patient_id(user_id)
        if not patient_id:
            logger.warning(f"create_chat_session: could not resolve patient_id for profile {user_id}")
            patient_id = user_id  # fallback — will fail on FK but at least we tried
        data = {
            "id": session_id,
            "patient_id": patient_id,
            "title": title,
            "status": "active",
            "session_type": "triage",
            "message_count": 0,
        }
        try:
            self.client.table("chat_sessions").insert(data).execute()
        except Exception as e:
            logger.warning(f"create_chat_session failed (may already exist): {e}")

    async def save_chat_message(self, session_id: str, user_id: str, role: str, content: str):
        # Map role names to schema's sender_type values
        sender_type_map = {"user": "patient", "ai": "ai", "assistant": "ai", "doctor": "doctor", "patient": "patient"}
        sender_type = sender_type_map.get(role, "patient")
        data = {
            "session_id": session_id,
            "sender_type": sender_type,
            "sender_id": user_id if sender_type == "patient" else None,
            "message_text": content,
            "message_type": "text",
        }
        try:
            self.client.table("chat_messages").insert(data).execute()
        except Exception as e:
            logger.warning(f"save_chat_message failed: {e}")

    async def get_chat_session(self, session_id: str) -> Optional[Dict]:
        """Get a single chat session by ID."""
        try:
            result = (
                self.client.table("chat_sessions")
                .select("*")
                .eq("id", session_id)
                .limit(1)
                .execute()
            )
            return result.data[0] if result.data else None
        except Exception as e:
            logger.warning(f"get_chat_session failed: {e}")
            return None

    async def update_chat_session_title(self, session_id: str, title: str):
        """Update the title of a chat session."""
        try:
            self.client.table("chat_sessions").update(
                {"title": title}
            ).eq("id", session_id).execute()
        except Exception as e:
            logger.warning(f"update_chat_session_title failed: {e}")

    async def touch_chat_session(self, session_id: str):
        try:
            self.client.table("chat_sessions").update(
                {"last_activity": self._now()}
            ).eq("id", session_id).execute()
        except Exception as e:
            logger.warning(f"touch_chat_session failed: {e}")

    async def get_user_chat_sessions(self, user_id: str) -> List[Dict]:
        patient_id = await self._resolve_patient_id(user_id)
        lookup_id = patient_id or user_id
        result = (
            self.client.table("chat_sessions")
            .select("*")
            .eq("patient_id", lookup_id)
            .neq("status", "archived")
            .order("last_activity", desc=True)
            .execute()
        )
        return result.data

    async def get_chat_conversation(self, session_id: str) -> List[Dict]:
        result = (
            self.client.table("chat_messages")
            .select("*")
            .eq("session_id", session_id)
            .order("created_at", desc=False)
            .execute()
        )
        # Normalize to frontend-friendly format
        messages = []
        for msg in result.data:
            messages.append({
                "id": msg.get("id"),
                "session_id": msg.get("session_id"),
                "sender": "ai" if msg.get("sender_type") == "ai" else "user",
                "content": msg.get("message_text", ""),
                "created_at": msg.get("created_at"),
            })
        return messages

    async def delete_chat_session(self, session_id: str, user_id: str) -> bool:
        try:
            patient_id = await self._resolve_patient_id(user_id)
            lookup_id = patient_id or user_id
            self.client.table("chat_sessions").update(
                {"status": "archived"}
            ).eq("id", session_id).eq("patient_id", lookup_id).execute()
            return True
        except Exception:
            return False

    # ── DOCTORS ────────────────────────────────────────────────────────

    async def get_all_doctors(self, patient_facing: bool = False) -> List[Dict]:
        """Return all doctors, enriched with first_name/last_name/email from profiles.
        patient_facing=True shows available + inactive doctors (inactive = pending approval,
        visible to patients but not bookable). Suspended doctors are always excluded from
        patient-facing views."""
        query = self.client.table("doctors").select("*, profiles(first_name, last_name, email, phone)")
        if patient_facing:
            # Show available (bookable) and inactive (pending approval, visible but not bookable)
            query = query.in_("available_status", ["available", "inactive"])
        result = query.order("sort_order", desc=False).execute()
        doctors = []
        for row in result.data or []:
            profile = row.pop("profiles", None) or {}
            row["first_name"] = (profile.get("first_name") or "").strip()
            row["last_name"] = (profile.get("last_name") or "").strip()
            row["email"] = profile.get("email", "")
            row["phone"] = profile.get("phone", "")
            # Flatten specialties JSONB array → first element as 'specialization'
            specs = row.get("specialties") or []
            row["specialization"] = specs[0] if specs else "General Physician"
            # clinic_address doubles as city for the directory listing
            row["city"] = row.get("clinic_address") or ""
            # Patients can only book available doctors
            row["is_bookable"] = row.get("available_status") == "available"
            doctors.append(row)
        return doctors

    async def get_doctor_by_id(self, doctor_id: str) -> Optional[Dict]:
        result = (
            self.client.table("doctors")
            .select("*")
            .eq("id", doctor_id)
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None

    async def get_doctor_by_phone(self, phone: str) -> Optional[Dict]:
        # Look up the doctor's profile row first (works for both registration paths)
        profile_result = (
            self.client.table("profiles")
            .select("*")
            .eq("phone", phone)
            .eq("user_type", "doctor")
            .limit(1)
            .execute()
        )
        if not profile_result.data:
            return None
        profile = profile_result.data[0]

        # Fetch extra doctor-specific fields (profile_id FK)
        try:
            doctor_result = (
                self.client.table("doctors")
                .select("*")
                .eq("profile_id", profile["id"])
                .limit(1)
                .execute()
            )
            doctor_extra = doctor_result.data[0] if doctor_result.data else {}
        except Exception:
            doctor_extra = {}

        computed_name = _name(profile.get("first_name"), profile.get("last_name"))
        return {
            "id":            profile["id"],
            "name":          computed_name,
            "full_name":     computed_name,
            "phone":         profile.get("phone", ""),
            "email":         profile.get("email", ""),
            "password_hash": profile.get("password_hash", ""),
            **{k: v for k, v in doctor_extra.items() if k != "id"},
        }

    async def save_doctor(self, doctor_data: Dict) -> bool:
        try:
            self.client.table("doctors").upsert(doctor_data).execute()
            return True
        except Exception as e:
            logger.error(f"save_doctor failed: {e}")
            return False

    async def update_doctor_status(self, doctor_id: str, status: str) -> bool:
        try:
            self.client.table("doctors").update(
                {"available_status": status}
            ).eq("id", doctor_id).execute()
            return True
        except Exception as e:
            logger.error(f"update_doctor_status failed (id={doctor_id}, status={status}): {e}")
            return False

    async def login_doctor(self, phone: str, password: str) -> Optional[Dict]:
        import bcrypt
        doctor = await self.get_doctor_by_phone(phone)
        if not doctor:
            return None
        pw_hash = doctor.get("password_hash", "")
        if pw_hash and bcrypt.checkpw(password.encode(), pw_hash.encode()):
            return doctor
        return None

    # ── ADMIN ──────────────────────────────────────────────────────────

    async def create_admin(self, username: str, email: str, password: str, full_name: str) -> Dict:
        import bcrypt
        pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        username = username.strip() if username else username
        if username and not username.startswith("+"):
            username = "+" + username
        name_parts = (full_name or "").strip().split(" ", 1)
        data = {
            "phone": username,
            "email": email,
            "first_name": name_parts[0],
            "last_name": name_parts[1] if len(name_parts) > 1 else "",
            "password_hash": pw_hash,
            "user_type": "admin",
            "role": "admin",
        }
        result = self.client.table("profiles").insert(data).execute()
        return result.data[0] if result.data else {}

    async def login_admin(self, username: str, password: str) -> Optional[Dict]:
        import bcrypt
        # Username is stored with a '+' prefix in the phone column
        lookup = username if username.startswith("+") else "+" + username
        result = (
            self.client.table("profiles")
            .select("*")
            .eq("phone", lookup)
            .eq("user_type", "admin")
            .limit(1)
            .execute()
        )
        if not result.data:
            return None
        admin = result.data[0]
        if bcrypt.checkpw(password.encode(), admin["password_hash"].encode()):
            return admin
        return None

    # ── FAMILY MEMBERS ─────────────────────────────────────────────────

    async def get_family_members(self, patient_id: str) -> List[Dict]:
        result = (
            self.client.table("family_members")
            .select("*")
            .eq("patient_id", patient_id)
            .order("created_at")
            .execute()
        )
        return result.data or []

    async def create_family_member(self, patient_id: str, data: Dict) -> Optional[Dict]:
        row = {
            "patient_id":          patient_id,
            "name":                data.get("name", ""),
            "age":                 data.get("age"),
            "gender":              data.get("gender"),
            "relationship":        data.get("relationship"),
            "blood_group":         data.get("blood_group"),
            "medical_history":     data.get("medical_history"),
            "allergies":           data.get("allergies"),
            "current_medications": data.get("current_medications"),
        }
        result = self.client.table("family_members").insert(row).execute()
        return result.data[0] if result.data else None

    async def update_family_member(self, member_id: str, patient_id: str, data: Dict) -> Optional[Dict]:
        updates = {}
        for field in ("name", "age", "gender", "relationship", "blood_group",
                      "medical_history", "allergies", "current_medications"):
            if field in data:
                updates[field] = data[field]
        if not updates:
            return None
        result = (
            self.client.table("family_members")
            .update(updates)
            .eq("id", member_id)
            .eq("patient_id", patient_id)   # ownership check
            .execute()
        )
        return result.data[0] if result.data else None

    async def delete_family_member(self, member_id: str, patient_id: str) -> bool:
        self.client.table("family_members").delete().eq("id", member_id).eq("patient_id", patient_id).execute()
        return True

    async def get_system_stats(self) -> Dict:
        patients = self.client.table("profiles").select("id", count="exact").eq("user_type", "patient").execute()
        doctors = self.client.table("doctors").select("id", count="exact").execute()
        sessions = self.client.table("chat_sessions").select("id", count="exact").execute()
        return {
            "total_patients": patients.count or 0,
            "total_doctors": doctors.count or 0,
            "total_sessions": sessions.count or 0,
        }

    # ── PATIENTS ───────────────────────────────────────────────────────

    async def get_patient_by_phone(self, phone: str) -> Optional[Dict]:
        result = (
            self.client.table("profiles")
            .select("*")
            .eq("phone", phone)
            .eq("user_type", "patient")
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None

    async def get_patient_consultations(self, patient_id: str, phone: str = "") -> List[Dict]:
        """Return all consultations for a patient.

        profiles.id == Supabase auth UID (enforced by handle_new_user trigger +
        one-time migration), so a direct patients.profile_id lookup is all that's needed.
        """
        resolved_id = None
        try:
            patient_row = (
                self.client.table("patients")
                .select("id")
                .eq("profile_id", patient_id)
                .limit(1)
                .execute()
            )
            if patient_row.data:
                resolved_id = patient_row.data[0]["id"]
        except Exception:
            pass

        if not resolved_id:
            resolved_id = patient_id  # fallback; query will return []

        # Query consultations; enrich with doctor name via profiles join
        result = (
            self.client.table("consultations")
            .select("*, doctors(id, specialties, consultation_fee, clinic_address, profiles(first_name, last_name))")
            .eq("patient_id", resolved_id)
            .order("created_at", desc=True)
            .execute()
        )
        rows = result.data or []

        # 3. Fetch prescription IDs for these consultations in one query
        consultation_ids = [r["id"] for r in rows if r.get("id")]
        prescription_map: dict = {}
        if consultation_ids:
            try:
                rx_result = (
                    self.client.table("prescriptions")
                    .select("id, consultation_id")
                    .in_("consultation_id", consultation_ids)
                    .execute()
                )
                for rx in (rx_result.data or []):
                    cid = rx.get("consultation_id")
                    if cid and cid not in prescription_map:
                        prescription_map[cid] = rx["id"]
            except Exception:
                pass

        for row in rows:
            doctor = row.pop("doctors", None) or {}
            profile = doctor.pop("profiles", None) or {}
            row["doctor_name"]      = _name(profile.get("first_name"), profile.get("last_name"))
            row["consultation_fee"] = doctor.get("consultation_fee")
            row["doctor_specialty"] = (doctor.get("specialties") or [""])[0]
            row["clinic_address"]   = doctor.get("clinic_address") or ""
            row["prescription_id"]  = prescription_map.get(row["id"])
        return rows

    def _ensure_profile_and_patient(self, user_id: str, name: str = "", email: str = "") -> str:
        """Ensure a profiles row and patients row exist for a Supabase auth user.
        Returns the patients.id (not the auth UID).

        Supabase users sign up via Supabase Auth (not our /register endpoint), so
        they may have no row in our custom profiles table. We upsert a minimal row
        so the patients FK constraint is satisfied.
        """
        import uuid as _uuid

        # 1. Check / create profiles row
        try:
            profile_check = (
                self.client.table("profiles")
                .select("id")
                .eq("id", user_id)
                .limit(1)
                .execute()
            )
            if not profile_check.data:
                placeholder_phone = f"+91{user_id.replace('-','')[:10]}"
                _display = name or email or "Patient"
                _parts = _display.strip().split(" ", 1)
                self.client.table("profiles").insert({
                    "id":            user_id,
                    "user_type":     "patient",
                    "phone":         placeholder_phone,
                    "email":         email or "",
                    "first_name":    _parts[0],
                    "last_name":     _parts[1] if len(_parts) > 1 else "",
                    "password_hash": "supabase_auth",   # auth handled by Supabase
                    "role":          "patient",
                    "status":        "active",
                    "email_verified": True,
                }).execute()
                logger.info(f"_ensure_profile_and_patient: created profiles row for {user_id}")
        except Exception as e:
            logger.warning(f"_ensure_profile_and_patient: profiles upsert failed (may already exist): {e}")

        # 2. Check / create patients row
        try:
            patient_row = (
                self.client.table("patients")
                .select("id")
                .eq("profile_id", user_id)
                .limit(1)
                .execute()
            )
            if patient_row.data:
                return patient_row.data[0]["id"]
            else:
                patient_id = str(_uuid.uuid4())
                self.client.table("patients").insert({
                    "id":         patient_id,
                    "profile_id": user_id,
                }).execute()
                logger.info(f"_ensure_profile_and_patient: created patients row {patient_id} for {user_id}")
                return patient_id
        except Exception as e:
            logger.error(f"_ensure_profile_and_patient: patients insert failed: {e}")
            raise

    def create_consultation_from_chat_sync(self, data: dict) -> dict:
        """Sync version — called directly from ADK tool functions (no event loop needed).
        Resolves patient_id from user_id (Supabase auth UID) via profiles/patients.
        """
        user_id = data.get("user_id") or data.get("patient_id")

        # Ensure profiles + patients rows exist (Supabase users bypass /register)
        try:
            patient_id = self._ensure_profile_and_patient(
                user_id,
                name  = data.get("patient_name", ""),
                email = data.get("patient_email", ""),
            )
        except Exception as e:
            logger.error(f"create_consultation_from_chat_sync: patient resolve failed: {e}")
            raise

        record = {
            "id":                data["id"],
            "patient_id":        patient_id,
            "doctor_id":         data.get("doctor_id") or None,
            "specialty":         data.get("specialty", "general_medicine"),
            "patient_note":      data.get("patient_note", ""),
            "consultation_type": data.get("consultation_type", "in_person"),
            "status":            "requested",
        }
        record = {k: v for k, v in record.items() if v is not None}

        result = self.client.table("consultations").insert(record).execute()
        logger.info(f"create_consultation_from_chat_sync: inserted consultation {data['id']}")

        # Notify patient (sync)
        try:
            specialty_label = (data.get("specialty") or "General Medicine").replace("_", " ").title()
            self.client.table("notifications").insert({
                "user_id":           user_id,
                "notification_type": "consultation",
                "title":             "Consultation Booked",
                "message":           f"Your consultation has been booked. Specialty: {specialty_label}. A doctor will be assigned shortly.",
                "is_read":           False,
            }).execute()
        except Exception as e:
            logger.warning(f"create_consultation_from_chat_sync: notification failed: {e}")

        return result.data[0] if result.data else record

    async def create_consultation_from_chat(self, data: dict) -> dict:
        """Save an AI-chat-initiated consultation request to the consultations table.
        Resolves patient_id from user_id (Supabase auth UID) via the profiles/patients relationship.
        """
        user_id = data.get("user_id") or data.get("patient_id")

        # Ensure profiles + patients rows exist (Supabase users bypass /register)
        try:
            patient_id = self._ensure_profile_and_patient(user_id)
        except Exception as e:
            logger.warning(f"create_consultation_from_chat: patient resolve failed: {e}")
            patient_id = user_id  # last-resort fallback

        record = {
            "id":                data["id"],
            "patient_id":        patient_id,
            "doctor_id":         data.get("doctor_id") or None,
            "specialty":         data.get("specialty", "general_medicine"),
            "patient_note":      data.get("patient_note", ""),
            "consultation_type": data.get("consultation_type", "in_person"),
            "status":            "requested",
        }
        record = {k: v for k, v in record.items() if v is not None}

        result = (
            self.client.table("consultations")
            .insert(record)
            .execute()
        )

        # Notify patient (async)
        try:
            specialty_label = (data.get("specialty") or "General Medicine").replace("_", " ").title()
            self.client.table("notifications").insert({
                "user_id":           user_id,
                "notification_type": "consultation",
                "title":             "Consultation Booked",
                "message":           f"Your consultation has been booked. Specialty: {specialty_label}. A doctor will be assigned shortly.",
                "is_read":           False,
            }).execute()
        except Exception as e:
            logger.warning(f"create_consultation_from_chat: notification failed: {e}")

        return result.data[0] if result.data else record

    # ── PATIENT PRESCRIPTIONS (prescriptions table) ───────────────────

    async def get_patient_prescriptions_full(self, profile_id: str) -> List[Dict]:
        """Return prescriptions for a patient with their medicine items and doctor name.
        profile_id = Supabase auth UUID; we resolve to patients.id first."""
        try:
            patient_row = (
                self.client.table("patients")
                .select("id")
                .eq("profile_id", profile_id)
                .limit(1)
                .execute()
            )
            patient_id = patient_row.data[0]["id"] if patient_row.data else None
        except Exception:
            patient_id = None

        if not patient_id:
            return []

        result = (
            self.client.table("prescriptions")
            .select("*, prescription_items(*), doctors(profile_id)")
            .eq("patient_id", patient_id)
            .order("prescribed_at", desc=True)
            .execute()
        )

        prescriptions = []
        for rx in result.data or []:
            # Enrich with doctor full_name
            doctor_obj = rx.pop("doctors", None) or {}
            doctor_profile_id = doctor_obj.get("profile_id")
            doctor_name = ""
            if doctor_profile_id:
                try:
                    prof = (
                        self.client.table("profiles")
                        .select("first_name, last_name")
                        .eq("id", doctor_profile_id)
                        .limit(1)
                        .execute()
                    )
                    if prof.data:
                        doctor_name = _name(prof.data[0].get("first_name"), prof.data[0].get("last_name"))
                except Exception:
                    pass
            rx["doctor_name"] = doctor_name
            prescriptions.append(rx)
        return prescriptions

    # ── APPROVALS / PRESCRIPTIONS ──────────────────────────────────────

    async def get_all_approvals(self) -> List[Dict]:
        result = (
            self.client.table("approval_requests")
            .select("*")
            .order("created_at", desc=True)
            .execute()
        )
        return result.data

    async def get_approvals_by_user(self, user_id: str) -> List[Dict]:
        result = (
            self.client.table("approval_requests")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .execute()
        )
        return result.data

    async def get_approvals_by_patient(self, patient_id: str) -> List[Dict]:
        result = (
            self.client.table("approval_requests")
            .select("*")
            .eq("patient_id", patient_id)
            .order("created_at", desc=True)
            .execute()
        )
        return result.data

    async def create_approval(self, data: Dict) -> Dict:
        result = self.client.table("approval_requests").insert(data).execute()
        return result.data[0] if result.data else {}

    async def update_approval(self, approval_id: str, data: Dict) -> Dict:
        result = (
            self.client.table("approval_requests")
            .update(data)
            .eq("id", approval_id)
            .execute()
        )
        return result.data[0] if result.data else {}

    async def get_approval_by_id(self, approval_id: str) -> Optional[Dict]:
        result = (
            self.client.table("approval_requests")
            .select("*")
            .eq("id", approval_id)
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None

    async def get_prescription_document(self, approval_id: str) -> Optional[Dict]:
        result = (
            self.client.table("prescription_documents")
            .select("*")
            .eq("approval_id", approval_id)
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None

    async def save_prescription_document(self, data: Dict) -> Dict:
        result = self.client.table("prescription_documents").upsert(data).execute()
        return result.data[0] if result.data else {}

    # ── DRUG BLACKLIST ─────────────────────────────────────────────────

    async def get_blacklisted_drugs(self) -> List[Dict]:
        result = self.client.table("drug_blacklist").select("*").execute()
        return result.data

    async def add_blacklisted_drug(self, data: Dict) -> Dict:
        result = self.client.table("drug_blacklist").insert(data).execute()
        return result.data[0] if result.data else {}

    # ── PATIENT SAVE (used by medivora_agent/tools.py) ──────────────

    async def save_patient(self, patient_data: Dict) -> Dict:
        """Upsert a patient record (used by agent tools)."""
        try:
            result = self.client.table("profiles").upsert(patient_data).execute()
            return result.data[0] if result.data else {}
        except Exception as e:
            logger.warning(f"save_patient failed: {e}")
            return {}

    # ── SAFETY VIOLATIONS ──────────────────────────────────────────────

    async def log_safety_violation(self, **kwargs) -> Dict:
        """Log a drug safety violation event."""
        try:
            result = self.client.table("safety_violations").insert(kwargs).execute()
            return result.data[0] if result.data else {}
        except Exception as e:
            # Table may not exist yet — log but don't crash
            logger.warning(f"log_safety_violation failed (table may not exist): {e}")
            return {}

    # ── APPROVAL REQUESTS (agent tools) ────────────────────────────────

    async def save_approval_request(self, data: Dict) -> Dict:
        try:
            result = self.client.table("approval_requests").upsert(data).execute()
            return result.data[0] if result.data else {}
        except Exception as e:
            logger.warning(f"save_approval_request failed: {e}")
            return {}

    async def assign_doctor_to_approval(self, approval_id: str, doctor_id: str) -> bool:
        try:
            self.client.table("approval_requests").update(
                {"assigned_doctor_id": doctor_id}
            ).eq("id", approval_id).execute()
            return True
        except Exception:
            return False

    # ── DOCTORS (agent tools) ──────────────────────────────────────────

    async def get_available_doctors(self, specialty: str = "") -> List[Dict]:
        """Get doctors, optionally filtered by specialty."""
        try:
            query = self.client.table("doctors").select("*")
            if specialty:
                query = query.eq("specialization", specialty)
            result = query.execute()
            return result.data
        except Exception as e:
            logger.warning(f"get_available_doctors failed: {e}")
            return []

    # ── NOTIFICATIONS ──────────────────────────────────────────────────

    async def save_notification(self, doctor_id: str, approval_id: str, message: str, priority: int = 3) -> Dict:
        try:
            data = {
                "doctor_id": doctor_id,
                "approval_id": approval_id,
                "message": message,
                "priority": priority,
            }
            result = self.client.table("notifications").insert(data).execute()
            return result.data[0] if result.data else {}
        except Exception as e:
            logger.warning(f"save_notification failed (table may not exist): {e}")
            return {}

    # ── VERIFICATION ───────────────────────────────────────────────────

    async def get_verification_by_token(self, token: str) -> Optional[Dict]:
        result = (
            self.client.table("prescription_documents")
            .select("*")
            .eq("verification_token", token)
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None

    # ── DOCTOR PROFILE (extended) ──────────────────────────────────────

    async def update_doctor_profile(self, doctor_id: str, **kwargs) -> bool:
        """Update any fields on a doctor row (by doctors.id or profile_id)."""
        try:
            fields = {k: v for k, v in kwargs.items() if v is not None and v != ""}
            if not fields:
                return True
            # Try doctors.id first
            result = self.client.table("doctors").update(fields).eq("id", doctor_id).execute()
            if result.data:
                return True
            # Fallback: match by profile_id (Supabase auth UID)
            result = self.client.table("doctors").update(fields).eq("profile_id", doctor_id).execute()
            return bool(result.data)
        except Exception as e:
            logger.error(f"update_doctor_profile failed: {e}")
            return False

    async def get_doctor_full_profile(self, doctor_id: str) -> Optional[Dict]:
        """Get a doctor by doctors.id or profile_id, joined with profiles data."""
        try:
            result = (
                self.client.table("doctors")
                .select("*, profiles(first_name, last_name, email, phone, status)")
                .eq("id", doctor_id)
                .limit(1)
                .execute()
            )
            if not result.data:
                result = (
                    self.client.table("doctors")
                    .select("*, profiles(first_name, last_name, email, phone, status)")
                    .eq("profile_id", doctor_id)
                    .limit(1)
                    .execute()
                )
            if not result.data:
                return None
            row = result.data[0]
            profile = row.pop("profiles", None) or {}
            row["first_name"] = profile.get("first_name", "")
            row["last_name"]  = profile.get("last_name", "")
            row["full_name"]  = _name(profile.get("first_name"), profile.get("last_name"))
            row["email"]      = row.get("email") or profile.get("email", "")
            row["phone"]      = row.get("phone") or profile.get("phone", "")
            return row
        except Exception as e:
            logger.error(f"get_doctor_full_profile failed: {e}")
            return None

    # ── CONSULTATIONS (missing methods used by api.py) ─────────────────

    async def create_consultation(self, data: dict) -> Optional[Dict]:
        """Create a consultation record (from consultation/request endpoint)."""
        try:
            import uuid as _uuid
            # Resolve patient_id: always ensure a valid patients row exists
            user_id = data.get("patient_id")
            if user_id:
                try:
                    patient_id = self._ensure_profile_and_patient(
                        user_id,
                        name=data.get("patient_name", ""),
                        email=data.get("patient_email", ""),
                    )
                except Exception:
                    patient_id = user_id  # last-resort fallback
            else:
                patient_id = user_id

            record = {
                "id":                data.get("id", str(_uuid.uuid4())),
                "patient_id":        patient_id,
                "doctor_id":         data.get("doctor_id"),
                "specialty":         data.get("specialty", "general_medicine"),
                "patient_note":      data.get("patient_note", ""),
                "consultation_type": data.get("consultation_type", "in_person"),
                "status":            data.get("status", "requested"),
                "scheduled_at":      data.get("scheduled_at"),
                "payment_id":        data.get("payment_id"),
                "payment_order_id":  data.get("payment_order_id"),
                "room_name":         data.get("room_name"),
                "room_url":          data.get("room_url"),
                "patient_token":     data.get("patient_token"),
                "doctor_token":      data.get("doctor_token"),
            }
            record = {k: v for k, v in record.items() if v is not None}
            result = self.client.table("consultations").insert(record).execute()
            return result.data[0] if result.data else record
        except Exception as e:
            logger.error(f"create_consultation failed: {e}")
            return None

    async def get_consultation_by_id(self, session_id: str) -> Optional[Dict]:
        """Get a single consultation by id."""
        try:
            result = (
                self.client.table("consultations")
                .select("*")
                .eq("id", session_id)
                .limit(1)
                .execute()
            )
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"get_consultation_by_id failed: {e}")
            return None

    async def update_consultation(self, session_id: str, updates: Dict) -> bool:
        """Update fields on a consultation row."""
        try:
            updates["updated_at"] = self._now()
            self.client.table("consultations").update(updates).eq("id", session_id).execute()
            return True
        except Exception as e:
            logger.error(f"update_consultation failed: {e}")
            return False

    async def get_pending_consultations(self) -> List[Dict]:
        """Return all consultations with status 'requested' (waiting for a doctor)."""
        try:
            result = (
                self.client.table("consultations")
                .select("*, patients(id, profiles(first_name, last_name, phone))")
                .in_("status", ["requested", "waiting"])
                .order("created_at", desc=False)
                .execute()
            )
            rows = result.data or []
            for row in rows:
                patient = row.pop("patients", None) or {}
                profile = patient.pop("profiles", None) or {}
                row["patient_name"]  = _name(profile.get("first_name"), profile.get("last_name")) or "Patient"
                row["patient_phone"] = profile.get("phone", "")
            return rows
        except Exception as e:
            logger.error(f"get_pending_consultations failed: {e}")
            return []

    async def resolve_doctor_id(self, profile_id: str) -> str:
        """Resolve a Supabase auth UID (profile_id) to doctors.id. Returns profile_id unchanged if not found."""
        try:
            result = (
                self.client.table("doctors")
                .select("id")
                .eq("profile_id", profile_id)
                .limit(1)
                .execute()
            )
            if result.data:
                return result.data[0]["id"]
        except Exception:
            pass
        return profile_id

    async def get_doctor_consultations(self, doctor_id: str) -> List[Dict]:
        """Return all consultations assigned to a doctor (by doctors.id)."""
        try:
            result = (
                self.client.table("consultations")
                .select("*, patients(id, profiles(first_name, last_name, phone))")
                .eq("doctor_id", doctor_id)
                .order("created_at", desc=True)
                .execute()
            )
            rows = result.data or []

            # Collect IDs of completed consultations to check for prescriptions
            completed_ids = [r["id"] for r in rows if r.get("status") == "completed"]
            prescribed_ids: set = set()
            if completed_ids:
                try:
                    rx_result = (
                        self.client.table("prescriptions")
                        .select("consultation_id")
                        .in_("consultation_id", completed_ids)
                        .execute()
                    )
                    prescribed_ids = {r["consultation_id"] for r in (rx_result.data or [])}
                except Exception:
                    pass  # non-fatal — migration may not be run yet

            for row in rows:
                patient = row.pop("patients", None) or {}
                profile = patient.pop("profiles", None) or {}
                row["patient_name"]      = _name(profile.get("first_name"), profile.get("last_name")) or "Patient"
                row["patient_phone"]     = profile.get("phone", "")
                row["has_prescription"]  = row["id"] in prescribed_ids
            return rows
        except Exception as e:
            logger.error(f"get_doctor_consultations failed: {e}")
            return []

    async def get_doctor_total_income(self, doctor_id: str) -> float:
        """Total income = paid consultations * doctor's consultation_fee."""
        try:
            # Get doctor's consultation fee
            doc_result = (
                self.client.table("doctors")
                .select("consultation_fee")
                .eq("id", doctor_id)
                .limit(1)
                .execute()
            )
            fee = float((doc_result.data or [{}])[0].get("consultation_fee") or 0)
            if fee == 0:
                return 0.0

            # Count all scheduled/completed consultations for this doctor
            # (every booking goes through payment — dev_skip or real Razorpay)
            consult_result = (
                self.client.table("consultations")
                .select("id")
                .eq("doctor_id", doctor_id)
                .in_("status", ["scheduled", "completed"])
                .execute()
            )
            count = len(consult_result.data or [])
            return round(fee * count, 2)
        except Exception as e:
            logger.error(f"get_doctor_total_income failed: {e}")
            return 0.0

    # ── APPROVAL REQUESTS (missing methods used by api.py) ─────────────

    async def get_pending_approvals(self, doctor_id: str) -> List[Dict]:
        """Return pending approval_requests assigned to this doctor or unassigned."""
        try:
            result = (
                self.client.table("approval_requests")
                .select("*")
                .eq("status", "pending")
                .order("created_at", desc=True)
                .execute()
            )
            return result.data or []
        except Exception as e:
            logger.error(f"get_pending_approvals failed: {e}")
            return []

    async def update_approval_status(
        self,
        approval_id: str,
        status: str,
        doctor_id: str,
        doctor_notes: str = "",
        approved_prescription: str = "",
    ) -> bool:
        """Update the status of an approval request."""
        try:
            updates: Dict = {
                "status":              status,
                "assigned_doctor_id":  doctor_id,
                "doctor_notes":        doctor_notes,
                "responded_at":        self._now(),
            }
            if approved_prescription:
                updates["modified_prescription"] = approved_prescription
            self.client.table("approval_requests").update(updates).eq("id", approval_id).execute()
            return True
        except Exception as e:
            logger.error(f"update_approval_status failed: {e}")
            return False

    async def store_original_prescription(self, approval_id: str) -> bool:
        """Snapshot the proposed_prescription into a separate column before doctor edits."""
        try:
            row = await self.get_approval_by_id(approval_id)
            if not row:
                return False
            proposed = row.get("proposed_prescription")
            if proposed is None:
                return True  # nothing to snapshot
            self.client.table("approval_requests").update(
                {"original_prescription": proposed}
            ).eq("id", approval_id).execute()
            return True
        except Exception as e:
            logger.warning(f"store_original_prescription failed (column may not exist): {e}")
            return False

    async def store_signature(
        self,
        approval_id: str,
        sig_hash: str,
        nmc_number: str = "",
        clinic_address: str = "",
    ) -> bool:
        """Store the doctor's digital signature and NMC details on an approval."""
        try:
            updates: Dict = {"signature_hash": sig_hash}
            if nmc_number:
                updates["nmc_number"] = nmc_number
            self.client.table("approval_requests").update(updates).eq("id", approval_id).execute()
            return True
        except Exception as e:
            logger.error(f"store_signature failed: {e}")
            return False

    async def save_edit_log(
        self,
        approval_id: str,
        doctor_id: str,
        field: str,
        old_value: str,
        new_value: str,
        change_type: str = "modified",
    ) -> bool:
        """Append an entry to the prescription edit log."""
        try:
            data = {
                "approval_id":  approval_id,
                "doctor_id":    doctor_id,
                "field_name":   field,
                "old_value":    str(old_value)[:2000],
                "new_value":    str(new_value)[:2000],
                "change_type":  change_type,
                "created_at":   self._now(),
            }
            self.client.table("prescription_edit_log").insert(data).execute()
            return True
        except Exception as e:
            logger.warning(f"save_edit_log failed (table may not exist): {e}")
            return False

    async def get_notifications(self, doctor_id: str, unread_only: bool = False) -> List[Dict]:
        """Fetch notifications for a doctor."""
        try:
            query = self.client.table("notifications").select("*").eq("doctor_id", doctor_id)
            if unread_only:
                query = query.eq("is_read", False)
            result = query.order("created_at", desc=True).limit(50).execute()
            return result.data or []
        except Exception as e:
            logger.warning(f"get_notifications failed: {e}")
            return []

    async def create_user_notification(
        self,
        user_id: str,
        notification_type: str,
        title: str,
        message: str,
        priority: int = 3,  # kept for API compat but not stored (not in schema)
    ) -> Dict:
        """Create a notification for a patient user."""
        try:
            data = {
                "user_id": user_id,
                "notification_type": notification_type,
                "title": title,
                "message": message,
                "is_read": False,
            }
            result = self.client.table("notifications").insert(data).execute()
            return result.data[0] if result.data else {}
        except Exception as e:
            logger.warning(f"create_user_notification failed: {e}")
            return {}

    async def get_user_notifications(self, user_id: str, unread_only: bool = False) -> List[Dict]:
        """Fetch notifications for a patient user."""
        try:
            query = self.client.table("notifications").select("*").eq("user_id", user_id)
            if unread_only:
                query = query.eq("is_read", False)
            result = query.order("created_at", desc=True).limit(50).execute()
            return result.data or []
        except Exception as e:
            logger.warning(f"get_user_notifications failed: {e}")
            return []

    async def mark_notification_read(self, notification_id: str, user_id: str = None) -> bool:
        """Mark a notification as read. Optionally restrict to a specific user."""
        try:
            query = self.client.table("notifications").update({"is_read": True}).eq("id", notification_id)
            if user_id:
                query = query.eq("user_id", user_id)
            result = query.execute()
            return bool(result.data)
        except Exception as e:
            logger.warning(f"mark_notification_read failed: {e}")
            return False

    async def mark_all_notifications_read(self, user_id: str) -> bool:
        """Mark all notifications as read for a user."""
        try:
            self.client.table("notifications").update({"is_read": True}).eq("user_id", user_id).execute()
            return True
        except Exception as e:
            logger.warning(f"mark_all_notifications_read failed: {e}")
            return False

    # ── CONSULTATION PRESCRIPTIONS ─────────────────────────────────────

    async def get_consultation_with_patient(self, consultation_id: str) -> Optional[Dict]:
        """Return a consultation row enriched with patient demographics and profile."""
        try:
            result = (
                self.client.table("consultations")
                .select("*, patients(id, age, gender, medical_history, allergies, chronic_conditions, profile_id)")
                .eq("id", consultation_id)
                .limit(1)
                .execute()
            )
            if not result.data:
                return None
            row = result.data[0]
            patient = row.pop("patients", None) or {}
            row["patient_db_id"] = patient.get("id", "")
            row["patient_age"] = patient.get("age")
            row["patient_gender"] = patient.get("gender", "")
            row["patient_medical_history"] = patient.get("medical_history", [])
            row["patient_allergies"] = patient.get("allergies", [])
            row["patient_chronic_conditions"] = patient.get("chronic_conditions", [])
            row["patient_profile_id"] = patient.get("profile_id", "")

            # Enrich with patient full name from profiles
            if row["patient_profile_id"]:
                try:
                    prof = (
                        self.client.table("profiles")
                        .select("first_name, last_name")
                        .eq("id", row["patient_profile_id"])
                        .limit(1)
                        .execute()
                    )
                    if prof.data:
                        row["patient_name"] = _name(prof.data[0].get("first_name"), prof.data[0].get("last_name")) or "Patient"
                    else:
                        row["patient_name"] = "Patient"
                except Exception:
                    row["patient_name"] = "Patient"
            return row
        except Exception as e:
            logger.error(f"get_consultation_with_patient failed: {e}")
            return None

    async def get_prescription_by_consultation(self, consultation_id: str) -> Optional[Dict]:
        """Return prescription linked to a consultation, or None."""
        try:
            result = (
                self.client.table("prescriptions")
                .select("id, status")
                .eq("consultation_id", consultation_id)
                .limit(1)
                .execute()
            )
            return result.data[0] if result.data else None
        except Exception as e:
            logger.warning(f"get_prescription_by_consultation failed: {e}")
            return None

    async def create_prescription_with_items(
        self,
        rx_data: Dict,
        items: List[Dict],
    ) -> Dict:
        """Insert a prescription row + all prescription_items, return the prescription."""
        import uuid as _uuid
        import json

        rx_id = str(_uuid.uuid4())
        rx_data["id"] = rx_id

        # Ensure JSONB fields are serialised as strings for Supabase client
        for field in ("general_instructions", "dietary_advice", "warning_signs"):
            val = rx_data.get(field, [])
            if isinstance(val, list):
                rx_data[field] = json.dumps(val)

        result = self.client.table("prescriptions").insert(rx_data).execute()
        saved_rx = result.data[0] if result.data else rx_data

        # Insert prescription items
        for item in items:
            item["id"] = str(_uuid.uuid4())
            item["prescription_id"] = rx_id
            for jfield in ("contraindications", "side_effects"):
                if isinstance(item.get(jfield), list):
                    item[jfield] = json.dumps(item[jfield])
            try:
                self.client.table("prescription_items").insert(item).execute()
            except Exception as e:
                logger.warning(f"create_prescription_with_items: item insert failed: {e}")

        return saved_rx

    # ─── Doctor Join Requests ──────────────────────────────────────────────────

    async def create_doctor_join_request(self, data: Dict) -> Dict:
        result = self.client.table("doctor_join_requests").insert(data).execute()
        return result.data[0] if result.data else {}

    async def get_doctor_join_requests(self, status: Optional[str] = None) -> List[Dict]:
        query = self.client.table("doctor_join_requests").select("*").order("created_at", desc=True)
        if status:
            query = query.eq("status", status)
        result = query.execute()
        return result.data or []

    async def get_doctor_join_request(self, request_id: str) -> Optional[Dict]:
        result = self.client.table("doctor_join_requests").select("*").eq("id", request_id).limit(1).execute()
        return result.data[0] if result.data else None

    async def update_doctor_join_request_status(self, request_id: str, status: str, reviewed_by: str) -> bool:
        from datetime import datetime, timezone
        try:
            self.client.table("doctor_join_requests").update({
                "status": status,
                "reviewed_at": datetime.now(timezone.utc).isoformat(),
                "reviewed_by": reviewed_by,
            }).eq("id", request_id).execute()
            return True
        except Exception as e:
            logger.error(f"update_doctor_join_request_status failed: {e}")
            return False

    # ─── Doctor Waitlist ───────────────────────────────────────────────────────

    async def add_to_doctor_waitlist(self, name: str, phone: str) -> Dict:
        """Insert a doctor into the waitlist table."""
        result = self.client.table("doctor_waitlist").insert({
            "name": name,
            "phone": phone,
        }).execute()
        return result.data[0] if result.data else {}
