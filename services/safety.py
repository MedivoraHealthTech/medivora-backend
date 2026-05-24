"""
Medivora Safety Validator — Phase 5

Post-processes every Gemini response before it reaches the user.
Checks:
  1. drug_mention       — AI named a specific drug outside approved context
  2. allergy_conflict   — AI mentions a drug the patient is known to be allergic to
  3. emergency_flag     — Response severity requires escalation UI
  4. pii_echo           — Response echoes PII unnecessarily (phone, Aadhaar, etc.)

Each check either:
  - Logs an 'info' event and passes through
  - Logs a 'warning' event and annotates the response
  - Logs a 'blocked' event and replaces the offending content
  - Logs a 'critical' event and overrides the full response
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Drug name patterns ─────────────────────────────────────────────────────────
# Covers common generic names and brand names. Not exhaustive — catches the most
# dangerous ones (anticoagulants, NSAIDs, opioids, antibiotics, psychotropics).
_DRUG_PATTERN = re.compile(
    r"\b("
    # NSAIDs
    r"ibuprofen|naproxen|diclofenac|aspirin|mefenamic|celecoxib|indomethacin|"
    # Opioids
    r"morphine|oxycodone|codeine|tramadol|fentanyl|hydrocodone|buprenorphine|"
    # Antibiotics
    r"amoxicillin|azithromycin|ciprofloxacin|metronidazole|doxycycline|"
    r"levofloxacin|cephalexin|clindamycin|nitrofurantoin|trimethoprim|"
    # Anticoagulants
    r"warfarin|heparin|rivaroxaban|apixaban|dabigatran|clopidogrel|"
    # Psychotropics
    r"sertraline|fluoxetine|escitalopram|paroxetine|venlafaxine|alprazolam|"
    r"clonazepam|diazepam|lorazepam|quetiapine|olanzapine|risperidone|lithium|"
    # Cardiovascular
    r"atenolol|metoprolol|amlodipine|lisinopril|ramipril|enalapril|losartan|"
    r"furosemide|spironolactone|digoxin|nitroglycerin|isosorbide|"
    # Diabetes
    r"metformin|glipizide|glimepiride|glyburide|insulin|sitagliptin|empagliflozin|"
    # Steroids
    r"prednisolone|prednisone|dexamethasone|betamethasone|hydrocortisone|"
    # Others high-risk
    r"methotrexate|hydroxychloroquine|colchicine|allopurinol|isotretinoin"
    r")\b",
    re.IGNORECASE,
)

# ── Emergency severity markers ────────────────────────────────────────────────
_EMERGENCY_MARKERS = re.compile(
    r"\b(EMERGENCY|108|call ambulance|VERY_SEVERE|🆘|🚨)\b",
    re.IGNORECASE,
)

# ── PII patterns ──────────────────────────────────────────────────────────────
_PHONE_PATTERN  = re.compile(r"\b[6-9]\d{9}\b")
_AADHAAR_PATTERN = re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")
_EMAIL_PATTERN  = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")


class SafetyResult:
    __slots__ = (
        "response", "modified", "events", "has_emergency", "is_blocked"
    )

    def __init__(self, response: str):
        self.response    = response
        self.modified    = False
        self.events: list = []
        self.has_emergency = False
        self.is_blocked    = False

    def add_event(self, event_type: str, severity: str, details: dict):
        self.events.append({
            "event_type": event_type,
            "severity":   severity,
            "details":    details,
        })
        if severity in ("blocked", "critical"):
            self.is_blocked = True


class SafetyValidator:
    """
    Validates Gemini response before delivery to the patient.

    Usage:
        result = SafetyValidator().validate(response, patient_context)
        # Use result.response (may be sanitized)
        # Log result.events to safety_events table (background)
    """

    def validate(
        self,
        response: str,
        patient_context: Optional[dict] = None,
        session_id: Optional[str] = None,
        patient_id: Optional[str] = None,
    ) -> SafetyResult:
        result = SafetyResult(response)

        self._check_allergy_conflict(result, patient_context)
        self._check_pii_echo(result)
        self._check_emergency_flag(result)
        # Drug mention check is info-level only (Gemini is allowed to name drugs
        # in approved context — e.g. "avoid ibuprofen in pregnancy")
        self._check_drug_mention(result)

        if result.modified:
            logger.info(
                f"Safety validator modified response for session={session_id}, "
                f"events={[e['event_type'] for e in result.events]}"
            )

        return result

    # ── Individual checks ──────────────────────────────────────────────────────

    def _check_allergy_conflict(self, result: SafetyResult, context: Optional[dict]):
        """
        Block if AI response recommends a drug the patient is allergic to.
        """
        if not context:
            return
        allergies = context.get("facts", {}).get("allergy", [])
        if not allergies:
            return

        response_lower = result.response.lower()
        for allergy_fact in allergies:
            drug = allergy_fact["value"].lower()
            # Check if the drug name appears in the response in a prescriptive context
            if re.search(rf"\b{re.escape(drug)}\b", response_lower):
                # Check if it's a "don't take" or warning context — that's OK
                context_window = self._get_context_window(response_lower, drug, chars=80)
                if self._is_prohibitive_context(context_window):
                    # AI is correctly warning against it
                    continue

                # AI appears to be recommending a drug the patient is allergic to
                warning = (
                    f"\n\n⚠️ **Important:** Please inform your doctor about your allergy "
                    f"before taking any medication mentioned above."
                )
                result.response = result.response + warning
                result.modified  = True
                result.add_event(
                    event_type="allergy_conflict",
                    severity="critical",
                    details={"drug": drug, "allergy": allergy_fact["value"]},
                )
                logger.warning(
                    f"ALLERGY CONFLICT: AI response mentions '{drug}' "
                    f"which patient is allergic to."
                )

    # Known helpline numbers that must NEVER be redacted
    _HELPLINE_WHITELIST = {
        "9152987821",  # iCall mental health
        "9820466627",  # Vandrevala
        "9999666555",  # iCall alternate
    }

    def _check_pii_echo(self, result: SafetyResult):
        """
        Strip patient PII (mobile numbers, Aadhaar) from AI responses.
        Whitelists known helpline numbers so they are never redacted.
        """
        text = result.response
        modified = False

        def _redact_if_not_helpline(m: re.Match) -> str:
            num = m.group(0).replace(" ", "").replace("-", "")
            if num in self._HELPLINE_WHITELIST:
                return m.group(0)
            return "[phone redacted]"

        if _PHONE_PATTERN.search(text):
            cleaned = _PHONE_PATTERN.sub(_redact_if_not_helpline, text)
            if cleaned != text:
                text = cleaned
                modified = True
                result.add_event("pii_echo", "warning", {"type": "phone"})

        if _AADHAAR_PATTERN.search(text):
            text = _AADHAAR_PATTERN.sub("[ID redacted]", text)
            modified = True
            result.add_event("pii_echo", "blocked", {"type": "aadhaar"})

        if modified:
            result.response = text
            result.modified  = True

    def _check_emergency_flag(self, result: SafetyResult):
        """Flag responses that contain emergency severity markers."""
        if _EMERGENCY_MARKERS.search(result.response):
            result.has_emergency = True
            result.add_event(
                "emergency_flag",
                "info",
                {"marker": "EMERGENCY/108 found in response"},
            )

    def _check_drug_mention(self, result: SafetyResult):
        """Info-level log when AI mentions a specific drug name."""
        matches = _DRUG_PATTERN.findall(result.response)
        if matches:
            unique = list(set(m.lower() for m in matches))
            result.add_event(
                "drug_mention",
                "info",
                {"drugs": unique},
            )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_context_window(self, text: str, keyword: str, chars: int = 80) -> str:
        idx = text.find(keyword)
        if idx == -1:
            return ""
        start = max(0, idx - chars)
        end   = min(len(text), idx + len(keyword) + chars)
        return text[start:end]

    def _is_prohibitive_context(self, window: str) -> bool:
        """Return True if the surrounding text says 'avoid', 'don't take', etc."""
        prohibit_words = [
            "avoid", "don't", "do not", "contraindicated", "allerg",
            "never", "stop", "not recommended", "na len", "mat len",
        ]
        window_lower = window.lower()
        return any(w in window_lower for w in prohibit_words)


# ── Async DB logger ───────────────────────────────────────────────────────────

async def log_safety_events(
    db,
    result: SafetyResult,
    session_id: Optional[str],
    patient_id: Optional[str],
    raw_response: str,
):
    """Persist safety events to the safety_events table. Run as background task."""
    if not result.events:
        return
    try:
        for event in result.events:
            db.client.table("safety_events").insert({
                "session_id":         session_id,
                "patient_id":         patient_id,
                "event_type":         event["event_type"],
                "severity":           event["severity"],
                "raw_response":       raw_response[:5000] if event["severity"] != "info" else None,
                "sanitized_response": result.response[:5000] if result.modified else None,
                "details":            event["details"],
            }).execute()
    except Exception as e:
        logger.warning(f"log_safety_events failed: {e}")
