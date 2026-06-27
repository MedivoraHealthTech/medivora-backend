"""
Medivora Vision Service — Image Analysis Layer

Responsibilities:
  - Accept an uploaded image (JPEG/PNG/WEBP/GIF)
  - Call Gemini 2.5-flash multimodal to analyse the medical content
  - Return a structured analysis: image_type, description, medical_context,
    suggested_questions
  - Optionally persist the analysis to patient_memory and chat_image_analyses
  - Provide a formatted note that can be injected into the chat prompt so the
    next Gemini call is already aware of what the patient uploaded

Supported image categories detected:
  wound, skin_condition, rash, prescription, medicine_label, medical_report,
  xray, eye_condition, other
"""

import os
import base64
import logging
import mimetypes
from typing import Optional

logger = logging.getLogger(__name__)

_GEMINI_MODEL = "gemini-2.5-flash"

# Maximum image bytes we'll send to Gemini (10 MB — same as upload limit)
_MAX_IMAGE_BYTES = 10 * 1024 * 1024

# ── Prompt used for the vision analysis ──────────────────────────────────────
_VISION_SYSTEM_PROMPT = """You are a warm, caring doctor-friend on Medivora, a telehealth app. A patient just shared a photo with you.

Respond like a real concerned friend who happens to be a doctor — natural, empathetic, conversational. NOT clinical, NOT robotic.

Examples of the tone you should use:
- For a wound: "Oh that looks pretty nasty! That's an open wound on your palm — the skin's been torn back. Is there any bleeding right now? When did this happen? And when was your last tetanus shot?"
- For a rash: "Ouch, that rash looks really irritated and angry! How long has it been like this? Is it itchy or does it burn? Did anything new come in contact with that area recently?"
- For a skin condition: "That looks quite inflamed. Has this spread recently or has it been this size for a while? Is it painful to touch?"
- For a prescription: "Got it, I can see your prescription! Looks like [drug name] was prescribed. Are you currently taking this, or is this a new prescription?"
- For a medicine: "I can see this is [medicine name]. Are you taking this for something specific, or did someone recommend it to you?"

Respond with a JSON object containing EXACTLY these fields:

{
  "image_type": "<one of: wound, skin_condition, rash, prescription, medicine_label, medical_report, xray, eye_condition, dental, other>",
  "description": "<your natural, warm, conversational reaction to what you see — like texting a friend who's a doctor. 2-4 sentences. Express concern if needed. DO NOT use clinical jargon.>",
  "medical_context": "<1-2 follow-up questions you'd naturally ask next, in the same warm tone. Keep it short.>",
  "suggested_questions": ["<short natural follow-up 1>", "<short natural follow-up 2>", "<short natural follow-up 3>"],
  "urgency_flag": "<none | monitor | seek_care | urgent>"
}

Rules:
- NEVER use clinical jargon like 'dermis', 'serous fluid', 'epidermal edges', 'purulent discharge', 'partial-thickness injury' etc.
- Speak like a caring doctor friend, not a medical textbook.
- Express genuine concern or reassurance depending on what you see.
- DO NOT make definitive diagnoses — but you CAN say "this looks like it could be..." or "this reminds me of..."
- For prescriptions / medicine labels: read visible text and ask naturally about it.
- urgency_flag = "urgent" ONLY if the image shows severe trauma, signs of emergency, or needs immediate attention.
- Respond with valid JSON only — no markdown fences, no extra text.
"""


class VisionService:
    """Analyses medical images using Gemini 2.5-flash multimodal."""

    def __init__(self):
        self._client = None  # Lazy-initialise

    def _get_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        return self._client

    async def analyse_image(
        self,
        image_bytes: bytes,
        mime_type: str,
        session_id: Optional[str] = None,
        patient_id: Optional[str] = None,
    ) -> dict:
        """
        Analyse a medical image and return structured findings.

        Returns a dict with keys:
            image_type, description, medical_context, suggested_questions,
            urgency_flag, prompt_note (formatted for injection into chat prompt)

        On any error, returns a safe fallback dict — never raises.
        """
        if len(image_bytes) > _MAX_IMAGE_BYTES:
            return self._fallback("Image is too large to analyse (max 10 MB).")

        if not mime_type or mime_type not in (
            "image/jpeg", "image/png", "image/webp", "image/gif",
            "image/heic", "image/heif",
        ):
            # Try to recover a sensible MIME type
            mime_type = "image/jpeg"

        try:
            client = self._get_client()
            b64_data = base64.standard_b64encode(image_bytes).decode("utf-8")

            from google.genai import types as genai_types

            response = client.models.generate_content(
                model=_GEMINI_MODEL,
                contents=[
                    genai_types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    genai_types.Part.from_text(text=_VISION_SYSTEM_PROMPT),
                ],
            )

            raw_text = (response.text or "").strip()

            # Strip accidental markdown fences
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
                raw_text = raw_text.strip()

            import json
            analysis = json.loads(raw_text)

            # Validate expected keys; fill any missing with safe defaults
            analysis.setdefault("image_type", "other")
            analysis.setdefault("description", "Image received and analysed.")
            analysis.setdefault("medical_context", "")
            analysis.setdefault("suggested_questions", [])
            analysis.setdefault("urgency_flag", "none")

            # Build a prompt-injection note for the next chat turn
            analysis["prompt_note"] = self._build_prompt_note(analysis)

            logger.info(
                f"Vision analysis complete: type={analysis['image_type']}, "
                f"urgency={analysis['urgency_flag']}, session={session_id}"
            )
            return analysis

        except Exception as exc:
            logger.error(f"Vision analysis error: {exc}", exc_info=True)
            return self._fallback(f"Vision analysis encountered an error: {exc}")

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_prompt_note(analysis: dict) -> str:
        """
        Build a concise [IMAGE CONTEXT] block to inject before the next user
        message so the AI has full visibility of what was uploaded.
        """
        lines = [
            "[IMAGE CONTEXT]",
            f"The patient just shared a photo. Image type: {analysis['image_type']}.",
            f"What was seen: {analysis['description']}",
        ]
        if analysis.get("medical_context"):
            lines.append(f"Follow-up context: {analysis['medical_context']}")
        urgency = analysis.get("urgency_flag", "none")
        if urgency not in ("none", ""):
            lines.append(f"Urgency level: {urgency}")
        lines.append(
            "Continue the conversation naturally, like a caring doctor-friend. "
            "Build on the image analysis — ask relevant follow-up questions in a warm, human tone. "
            "Do NOT repeat the full description back. Do NOT use clinical jargon."
        )
        return "\n".join(lines)

    @staticmethod
    def _fallback(reason: str) -> dict:
        return {
            "image_type": "other",
            "description": "I received your image but couldn't analyse it fully. Please describe what you're seeing.",
            "medical_context": reason,
            "suggested_questions": [
                "Can you describe what you see in the image?",
                "How long have you had this condition?",
                "Are you experiencing any pain or discomfort?",
            ],
            "urgency_flag": "none",
            "prompt_note": (
                "[IMAGE CONTEXT]\n"
                "The patient uploaded an image that could not be fully analysed. "
                "Ask them to describe what they see."
            ),
        }

    # ── Memory persistence ───────────────────────────────────────────────────

    async def persist_to_memory(
        self,
        db,
        patient_id: str,
        session_id: str,
        analysis: dict,
        filename: str,
    ) -> None:
        """
        Store image analysis in:
          1. patient_memory  — lightweight key/value fact (memory_type='image_upload')
          2. chat_image_analyses — full audit record
        Both writes are best-effort; failures are logged, never raised.
        """
        try:
            import json as _json
            # ── patient_memory fact ──
            db.client.table("patient_memory").upsert({
                "patient_id":    patient_id,
                "memory_type":   "image_upload",
                "key":           f"image_{session_id[:8]}",
                "value":         (
                    f"{analysis['image_type']}: {analysis['description']}"
                )[:500],
                "confidence":    0.9,
                "source_session": session_id,
            }, on_conflict="patient_id,memory_type,key").execute()

            # ── full audit record ──
            db.client.table("chat_image_analyses").insert({
                "patient_id":       patient_id,
                "session_id":       session_id,
                "filename":         filename or "upload",
                "image_type":       analysis["image_type"],
                "description":      analysis["description"],
                "medical_context":  analysis.get("medical_context", ""),
                "urgency_flag":     analysis.get("urgency_flag", "none"),
                "full_analysis":    _json.dumps(analysis),
            }).execute()

        except Exception as exc:
            logger.warning(f"Vision memory persist failed (non-fatal): {exc}")


# Module-level singleton (lazy-initialised once per worker process)
_vision_service: Optional[VisionService] = None


def get_vision_service() -> VisionService:
    global _vision_service
    if _vision_service is None:
        _vision_service = VisionService()
    return _vision_service
