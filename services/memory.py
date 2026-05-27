"""
Medivora Memory Service — Phase 1 + 4

Responsibilities:
  - Store structured patient facts (conditions, allergies, meds, preferences)
  - Generate end-of-session summaries via Gemini
  - Retrieve context before each new conversation
  - Phase 4: semantic similarity search via pgvector embeddings
"""

import os
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_GEMINI_MODEL = "gemini-2.5-flash"
_EMBED_MODEL  = "text-embedding-004"


class MemoryService:
    """
    Persistent patient memory across chat sessions.

    Usage:
        svc = MemoryService(db)                            # db = DatabaseManager()
        ctx = await svc.get_patient_context(patient_id)
        note = svc.format_context_for_prompt(ctx)
        # inject `note` into adapted_message before calling Gemini

        # After session ends (background task):
        await svc.close_session(patient_id, session_id, user_msg, ai_response, outcome)
    """

    def __init__(self, db):
        self.db = db  # DatabaseManager instance

    # ── Patient ID resolution ──────────────────────────────────────────────

    async def get_patient_id(self, profile_id: str) -> Optional[str]:
        """Resolve patients.id from profiles.id (JWT sub)."""
        try:
            res = self.db.client.table("patients") \
                .select("id") \
                .eq("profile_id", profile_id) \
                .limit(1) \
                .execute()
            if res.data:
                return res.data[0]["id"]
        except Exception as e:
            logger.warning(f"get_patient_id failed for profile {profile_id}: {e}")
        return None

    # ── Read ──────────────────────────────────────────────────────────────

    async def get_patient_context(
        self, patient_id: str, query_text: Optional[str] = None
    ) -> dict:
        """
        Fetch all structured memory for a patient.
        Phase 4: if query_text is provided, uses semantic similarity search
                 to retrieve the most relevant past sessions instead of
                 just the most recent ones.
        Returns: { facts: {type: [{key, value}]}, recent_summaries: [...] }
        """
        facts: dict = {}
        summaries: list = []

        try:
            rows = self.db.client.table("patient_memory") \
                .select("memory_type, key, value") \
                .eq("patient_id", patient_id) \
                .execute()
            for row in (rows.data or []):
                mtype = row["memory_type"]
                if mtype not in facts:
                    facts[mtype] = []
                facts[mtype].append({"key": row["key"], "value": row["value"]})
        except Exception as e:
            logger.warning(f"patient_memory fetch failed ({patient_id}): {e}")

        # Phase 4: semantic search if query provided, else recency fallback
        if query_text:
            summaries = await self.find_similar_sessions(patient_id, query_text, limit=3)
        else:
            try:
                rows = self.db.client.table("conversation_summaries") \
                    .select("summary_text, chief_complaint, outcome, created_at") \
                    .eq("patient_id", patient_id) \
                    .order("created_at", desc=True) \
                    .limit(3) \
                    .execute()
                summaries = rows.data or []
            except Exception as e:
                logger.warning(f"conversation_summaries fetch failed ({patient_id}): {e}")

        return {"facts": facts, "recent_summaries": summaries}

    def format_context_for_prompt(self, context: dict) -> str:
        """
        Convert patient context into a system note injected before the user message.
        Returns empty string if the patient has no memory yet.
        """
        facts = context.get("facts", {})
        summaries = context.get("recent_summaries", [])

        if not facts and not summaries:
            return ""

        lines = ["[PATIENT MEMORY — personalise your response using this context]"]

        # Identity — name/age/gender from previous sessions
        prefs = {f["key"]: f["value"] for f in facts.get("preference", [])}
        identity_parts = []
        if prefs.get("name"):
            identity_parts.append(f"Name: {prefs['name']}")
        if prefs.get("age"):
            identity_parts.append(f"Age: {prefs['age']}")
        if prefs.get("gender"):
            identity_parts.append(f"Gender: {prefs['gender']}")
        if identity_parts:
            lines.append("Patient identity: " + ", ".join(identity_parts))

        # Safety-critical first
        if facts.get("allergy"):
            allergies = ", ".join(f["value"] for f in facts["allergy"])
            lines.append(f"⚠️ KNOWN ALLERGIES: {allergies} — NEVER recommend these drugs or their class")

        if facts.get("condition"):
            conditions = ", ".join(f["value"] for f in facts["condition"])
            lines.append(f"Known medical conditions: {conditions}")

        if facts.get("medication"):
            meds = ", ".join(f["value"] for f in facts["medication"])
            lines.append(f"Current medications: {meds} — check for interactions")

        # Other preferences (excluding identity already shown)
        other_prefs = {k: v for k, v in prefs.items() if k not in ("name", "age", "gender")}
        if other_prefs:
            pref_str = "; ".join(f"{k}: {v}" for k, v in other_prefs.items())
            lines.append(f"Patient preferences: {pref_str}")

        emotional = facts.get("emotional_state", [])
        if emotional:
            state = emotional[0]["value"]
            if state and state != "neutral":
                lines.append(f"Emotional state from last visit: {state} — acknowledge warmly before anything clinical")

        if summaries:
            lines.append("\nPrevious visit history:")
            for s in summaries[:2]:
                date = (s.get("created_at") or "")[:10]
                complaint = s.get("chief_complaint", "—")
                outcome = s.get("outcome", "—")
                snippet = (s.get("summary_text") or "")[:180]
                lines.append(f"  [{date}] {complaint} → {outcome}: {snippet}…")

        lines.append("[END PATIENT MEMORY]")
        return "\n".join(lines)

    # ── Write ─────────────────────────────────────────────────────────────

    async def upsert_fact(
        self,
        patient_id: str,
        memory_type: str,
        key: str,
        value: str,
        session_id: Optional[str] = None,
    ):
        """Insert or update a single structured patient fact."""
        try:
            from datetime import datetime
            self.db.client.table("patient_memory").upsert(
                {
                    "patient_id":    patient_id,
                    "memory_type":   memory_type,
                    "key":           key.lower()[:255],
                    "value":         value[:1000],
                    "source_session": session_id,
                    "updated_at":    datetime.utcnow().isoformat(),
                },
                on_conflict="patient_id,memory_type,key",
            ).execute()
        except Exception as e:
            logger.warning(f"upsert_fact failed ({memory_type}/{key}): {e}")

    async def extract_and_store_facts(
        self, patient_id: str, session_id: str, conversation_text: str
    ):
        """
        Use Gemini to extract structured facts from a conversation snippet,
        then persist them to patient_memory.
        Called as a background task after each AI response.
        """
        try:
            from google import genai
            client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
            prompt = (
                "Extract durable facts from the following conversation for patient memory storage.\n"
                "Return a JSON object — include only keys that have confirmed data:\n"
                "{\n"
                '  "name": "Rishi",\n'
                '  "age": "32",\n'
                '  "gender": "male",\n'
                '  "conditions": ["diabetes", "hypertension"],\n'
                '  "allergies": ["penicillin"],\n'
                '  "medications": ["metformin 500mg OD"],\n'
                '  "emotional_state": "anxious",\n'
                '  "preferences": {"language": "hinglish"}\n'
                "}\n"
                "Rules:\n"
                "- Only include confirmed/stated facts, not speculation.\n"
                "- emotional_state: one of neutral/anxious/grieving/in_crisis.\n"
                "- name/age/gender: extract if the patient introduces themselves.\n"
                "- Return ONLY valid JSON, no markdown.\n\n"
                f"Conversation:\n{conversation_text[:3000]}"
            )

            resp = client.models.generate_content(model=_GEMINI_MODEL, contents=prompt)
            text = resp.text.strip()
            # Strip markdown fences if present
            if text.startswith("```"):
                parts = text.split("```")
                text = parts[1][4:] if parts[1].startswith("json") else parts[1]

            data = json.loads(text)

            # Identity facts (name/age/gender)
            if data.get("name"):
                await self.upsert_fact(patient_id, "preference", "name", data["name"], session_id)
            if data.get("age"):
                await self.upsert_fact(patient_id, "preference", "age", str(data["age"]), session_id)
            if data.get("gender"):
                await self.upsert_fact(patient_id, "preference", "gender", data["gender"], session_id)

            for item in data.get("conditions", []):
                await self.upsert_fact(patient_id, "condition", item.lower(), item, session_id)
            for item in data.get("allergies", []):
                await self.upsert_fact(patient_id, "allergy", item.lower(), item, session_id)
            for item in data.get("medications", []):
                await self.upsert_fact(patient_id, "medication", item.lower(), item, session_id)
            state = data.get("emotional_state", "")
            if state:
                await self.upsert_fact(patient_id, "emotional_state", "current", state, session_id)
            for k, v in data.get("preferences", {}).items():
                await self.upsert_fact(patient_id, "preference", k, str(v), session_id)

            logger.info(f"Facts extracted for patient {patient_id}: {list(data.keys())}")
        except Exception as e:
            logger.warning(f"extract_and_store_facts failed ({patient_id}): {e}")

    async def save_session_summary(
        self,
        patient_id: str,
        session_id: str,
        conversation_text: str,
        outcome: str = "unknown",
    ):
        """
        Generate a session summary via Gemini and store it.
        Called as a background task when a triage report is completed.
        Phase 4: will also generate + store embedding.
        """
        try:
            from google import genai
            client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
            prompt = (
                "Summarize this medical chat session in 2–3 clinical sentences.\n"
                "Include: chief complaint, key findings, and recommendation.\n"
                "Do NOT include patient name, phone, or any PII.\n\n"
                "Return ONLY a JSON object:\n"
                "{\n"
                '  "summary": "Patient presented with ...",\n'
                '  "chief_complaint": "headache and fever",\n'
                '  "outcome": "home_care"\n'
                "}\n"
                'outcome must be one of: home_care, prescription_pending, consultation_booked, emergency\n\n'
                f"Conversation:\n{conversation_text[:4000]}"
            )

            resp = client.models.generate_content(model=_GEMINI_MODEL, contents=prompt)
            text = resp.text.strip()
            if text.startswith("```"):
                parts = text.split("```")
                text = parts[1][4:] if parts[1].startswith("json") else parts[1]

            data = json.loads(text)
            summary_text    = data.get("summary", "").strip()
            chief_complaint = data.get("chief_complaint", "").strip()
            detected_outcome = data.get("outcome", outcome)

            if not summary_text:
                return

            record: dict = {
                "patient_id":     patient_id,
                "session_id":     session_id,
                "summary_text":   summary_text,
                "chief_complaint": chief_complaint,
                "outcome":        detected_outcome,
            }

            # Phase 4: generate embedding if pgvector column exists
            embedding = await self._generate_embedding(summary_text)
            if embedding:
                record["embedding"] = embedding

            self.db.client.table("conversation_summaries").insert(record).execute()
            logger.info(f"Session summary saved: patient={patient_id}, session={session_id}")
        except Exception as e:
            logger.warning(f"save_session_summary failed ({patient_id}): {e}")

    # ── Phase 4: Embeddings ────────────────────────────────────────────────

    async def _generate_embedding(self, text: str) -> Optional[list]:
        """
        Generate text embedding via text-embedding-004 (768 dimensions).
        Phase 4 is live — pgvector + embedding column enabled on Supabase.
        Returns None only on API failure.
        """
        try:
            from google import genai
            from google.genai import types as _gtypes
            client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
            result = client.models.embed_content(
                model=_EMBED_MODEL,
                contents=[text],
                config=_gtypes.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            return list(result.embeddings[0].values)
        except Exception as e:
            logger.warning(f"_generate_embedding failed: {e}")
            return None

    async def find_similar_sessions(
        self, patient_id: str, query_text: str, limit: int = 3
    ) -> list:
        """
        Phase 4: Semantic search over past session summaries.
        Falls back to recency-based retrieval until pgvector is ready.
        """
        try:
            embedding = await self._generate_embedding(query_text)
            if not embedding:
                raise ValueError("embedding unavailable")

            result = self.db.client.rpc(
                "match_conversation_summaries",
                {
                    "query_embedding":  embedding,
                    "match_patient_id": patient_id,
                    "match_count":      limit,
                },
            ).execute()
            return result.data or []
        except Exception:
            # Fallback: most recent summaries
            rows = self.db.client.table("conversation_summaries") \
                .select("summary_text, chief_complaint, outcome, created_at") \
                .eq("patient_id", patient_id) \
                .order("created_at", desc=True) \
                .limit(limit) \
                .execute()
            return rows.data or []

    # ── Convenience: close session ─────────────────────────────────────────

    async def close_session(
        self,
        patient_id: str,
        session_id: str,
        user_message: str,
        ai_response: str,
        outcome: str = "unknown",
    ):
        """
        Full session-end workflow (run as background task):
          1. Extract + store structured facts from this exchange
          2. Generate + store session summary
        """
        conversation_text = f"Patient: {user_message}\n\nMedivora AI: {ai_response}"
        await self.extract_and_store_facts(patient_id, session_id, conversation_text)
        await self.save_session_summary(patient_id, session_id, conversation_text, outcome)
