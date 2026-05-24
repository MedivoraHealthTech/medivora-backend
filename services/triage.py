"""
Medivora Triage Engine — Phase 3

Deterministic, pre-Gemini risk scoring.
Runs BEFORE the AI so that:
  1. Hard emergencies get an instant response (no LLM latency)
  2. Triage result is injected as structured context — Gemini doesn't re-derive it
  3. Routing decision is set deterministically, not probabilistically

Routing decisions:
  call_108            — life-threatening emergency, immediate 108 call
  book_now            — SEVERE/URGENT, book consultation immediately
  async_prescription  — MODERATE, AI prescription + doctor review
  home_care           — LOW/MILD, self-care advice

Risk score: 0–100
  90–100 → EMERGENCY
  65–89  → URGENT
  35–64  → MODERATE
  0–34   → LOW
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Emergency patterns (score 90–100) ────────────────────────────────────────
_EMERGENCY_HARD = re.compile(
    r"\b("
    # Cardiac — only clear-cut emergencies, not generic "chest pain"
    r"heart attack|cardiac arrest|crushing chest|"
    r"dil ka dauraa|"
    # Breathing — only severe
    r"can't breathe|cannot breathe|stopped breathing|"
    r"saans nahi aa rahi|saans ruk gayi|"
    # Neurological
    r"stroke|face drooping|slurred speech|sudden paralysis|"
    r"laqwa|"
    # Obstetric emergencies
    r"heavy bleeding|eclampsia|convulsion|"
    r"bahut zyada bleeding|"
    # Trauma / severe
    r"unconscious|not responding|overdose|suicide attempt|"
    r"hosh nahi|behosh ho gaya|behosh ho gayi|"
    # Severe allergic
    r"anaphylaxis|throat swelling|gala band"
    r")\b",
    re.IGNORECASE,
)

# Urgent cardiac patterns (chest pain alone → URGENT, not EMERGENCY)
_URGENT_CARDIAC = re.compile(
    r"\b(chest pain|chest tightness|chest pressure|seene mein dard|seene mein jalan)\b",
    re.IGNORECASE,
)

# ── Urgent patterns (score 65–89) ─────────────────────────────────────────────
_URGENT_PATTERNS = re.compile(
    r"\b("
    r"high fever|fever.*104|fever.*105|fever.*106|"
    r"tez bukhar|bahut bukhar|"
    r"blood in stool|blood in urine|vomiting blood|coughing blood|"
    r"khoon aa raha|khoon tha|"
    r"severe pain|unbearable pain|"
    r"bahut dard|behad dard|"
    r"severe headache|worst headache|"
    r"suddenly blurred vision|sudden vision loss|"
    r"contractions|labour pain|water broke|leaking fluid|"
    r"prasav peeda|paani toot gaya|"
    r"child.*breathing|baby.*breathing|infant.*fever"
    r")\b",
    re.IGNORECASE,
)

# ── Moderate patterns (score 35–64) ──────────────────────────────────────────
_MODERATE_PATTERNS = re.compile(
    r"\b("
    r"fever|headache|vomiting|nausea|diarrhea|diarrhoea|"
    r"bukhar|ulti|dast|sar dard|"
    r"back pain|joint pain|knee pain|shoulder pain|"
    r"kamar dard|ghutne dard|"
    r"cough|cold|sore throat|runny nose|"
    r"khansi|zukam|gala kharab|"
    r"rash|itching|skin|"
    r"khujli|chakte|"
    r"stomach pain|abdominal pain|cramps|"
    r"pet dard|maida|aant"
    r")\b",
    re.IGNORECASE,
)

# ── Specialty keyword map (weighted scoring) ──────────────────────────────────
_SPECIALTY_MAP: list[tuple[str, int, str]] = [
    # (pattern, weight, specialty)
    (r"\b(chest pain|heart|cardiac|palpitation|BP|blood pressure|hypertension)\b", 3, "cardiology"),
    (r"\b(pregnant|pregnancy|period|menstrual|ovary|PCOD|PCOS|cervix|uterus|breast)\b", 3, "womens_health"),
    (r"\b(child|baby|infant|toddler|pediatric|kid|newborn)\b", 3, "pediatrics"),
    (r"\b(skin|rash|acne|eczema|psoriasis|itching|dermatitis|hair loss)\b", 3, "dermatology"),
    (r"\b(bone|joint|fracture|sprain|back pain|knee|shoulder|arthritis|ortho)\b", 3, "orthopedics"),
    (r"\b(stomach|gastro|liver|acidity|acid reflux|IBS|diarrhea|vomiting|nausea)\b", 3, "gastroenterology"),
    (r"\b(cough|asthma|breathing|lung|TB|tuberculosis|COPD|pulmonary|chest)\b", 3, "pulmonology"),
    (r"\b(headache|migraine|seizure|epilepsy|memory|nerves|neuro|vertigo)\b", 3, "neurology"),
    (r"\b(ear|nose|throat|ENT|tonsil|sinusitis|hearing|voice)\b", 3, "ent"),
    (r"\b(eye|vision|glasses|cataract|glaucoma|retina)\b", 3, "ophthalmology"),
    (r"\b(mental|anxiety|depression|stress|sleep|insomnia|psychiatric)\b", 3, "psychiatry"),
    (r"\b(diabetes|thyroid|hormone|endocrine|sugar|insulin)\b", 3, "endocrinology"),
    (r"\b(kidney|urinary|UTI|prostate|bladder|renal)\b", 3, "urology"),
]


def _score_specialty(text: str) -> str:
    scores: dict[str, int] = {}
    for pattern, weight, specialty in _SPECIALTY_MAP:
        matches = len(re.findall(pattern, text, re.IGNORECASE))
        if matches:
            scores[specialty] = scores.get(specialty, 0) + matches * weight
    if not scores:
        return "general_medicine"
    return max(scores, key=lambda k: scores[k])


class TriageResult:
    __slots__ = (
        "risk_score", "risk_level", "routing_decision",
        "specialty", "is_hard_emergency", "emergency_response",
    )

    def __init__(
        self,
        risk_score: int,
        risk_level: str,
        routing_decision: str,
        specialty: str,
        is_hard_emergency: bool = False,
        emergency_response: str = "",
    ):
        self.risk_score         = risk_score
        self.risk_level         = risk_level
        self.routing_decision   = routing_decision
        self.specialty          = specialty
        self.is_hard_emergency  = is_hard_emergency
        self.emergency_response = emergency_response

    def to_dict(self) -> dict:
        return {
            "risk_score":        self.risk_score,
            "risk_level":        self.risk_level,
            "routing_decision":  self.routing_decision,
            "specialty":         self.specialty,
            "is_hard_emergency": self.is_hard_emergency,
        }

    def as_prompt_note(self) -> str:
        """Format as a system note injected before Gemini runs."""
        return (
            f"[TRIAGE PRE-ASSESSMENT]\n"
            f"Risk Level: {self.risk_level} (score {self.risk_score}/100)\n"
            f"Recommended Routing: {self.routing_decision}\n"
            f"Suggested Specialty: {self.specialty}\n"
            f"[Use this as your starting point — do not contradict without clinical reason]"
        )


class TriageEngine:
    """
    Deterministic triage scoring.
    Call score() before running Gemini.
    """

    def score(self, message: str, patient_context: Optional[dict] = None) -> TriageResult:
        """
        Score the risk level of a patient message.

        Args:
            message:         Current patient message
            patient_context: From MemoryService.get_patient_context() — used to
                             escalate risk if allergies / conditions are relevant

        Returns:
            TriageResult
        """
        text = message.strip()

        # ── Hard emergency check (instant response, skip Gemini) ──────────
        if _EMERGENCY_HARD.search(text):
            specialty = _score_specialty(text)
            return TriageResult(
                risk_score=95,
                risk_level="EMERGENCY",
                routing_decision="call_108",
                specialty=specialty,
                is_hard_emergency=True,
                emergency_response=self._emergency_response(text),
            )

        # ── Urgent cardiac (chest pain alone → URGENT, not EMERGENCY) ───
        if _URGENT_CARDIAC.search(text):
            return TriageResult(
                risk_score=75,
                risk_level="URGENT",
                routing_decision="book_now",
                specialty="cardiology",
            )

        # ── Urgent ────────────────────────────────────────────────────────
        if _URGENT_PATTERNS.search(text):
            specialty = _score_specialty(text)
            score = self._calc_score(text, base=70)
            return TriageResult(
                risk_score=score,
                risk_level="URGENT",
                routing_decision="book_now",
                specialty=specialty,
            )

        # ── Moderate ──────────────────────────────────────────────────────
        if _MODERATE_PATTERNS.search(text):
            specialty = _score_specialty(text)
            score = self._calc_score(text, base=45)

            # Escalate if patient has relevant conditions
            if patient_context:
                facts = patient_context.get("facts", {})
                if facts.get("condition"):
                    score = min(score + 10, 64)

            routing = "async_prescription" if score >= 50 else "home_care"
            return TriageResult(
                risk_score=score,
                risk_level="MODERATE",
                routing_decision=routing,
                specialty=specialty,
            )

        # ── Low / greeting / unclear ──────────────────────────────────────
        specialty = _score_specialty(text) if len(text) > 20 else "general_medicine"
        return TriageResult(
            risk_score=15,
            risk_level="LOW",
            routing_decision="home_care",
            specialty=specialty,
        )

    def _calc_score(self, text: str, base: int) -> int:
        """Refine score within a tier based on symptom density."""
        word_count = len(text.split())
        # More symptoms described = slightly higher score
        bonus = min(word_count // 10, 10)
        return min(base + bonus, base + 10)

    def _emergency_response(self, text: str) -> str:
        """Pre-built emergency response — no Gemini needed."""
        # Obstetric emergency
        if re.search(r"\b(pregnant|bleed|placenta|eclampsia|convulsion)\b", text, re.I):
            return (
                "🚨 **यह एक medical emergency है।**\n\n"
                "**अभी 108 पर call करें** — Free ambulance, 24/7 available है।\n\n"
                "किसी को अपने पास रहने दें। खुद drive न करें।\n"
                "हम एक emergency doctor को alert कर रहे हैं।\n\n"
                "📞 **108** | Nirbhaya Helpline: **1800-111-565**"
            )

        # Cardiac
        if re.search(r"\b(chest pain|heart|cardiac|seene mein dard)\b", text, re.I):
            return (
                "🚨 **यह Cardiac Emergency हो सकती है।**\n\n"
                "**अभी 108 पर call करें।**\n\n"
                "बैठ जाएं, ज़्यादा हिलें नहीं। कोई आपके पास हो।\n"
                "अगर Aspirin available है और allergy नहीं — 325mg लें।\n\n"
                "📞 **108** | Cardiac Helpline: **1800-120-6364**"
            )

        # General emergency
        return (
            "🚨 **Emergency — तुरंत मदद चाहिए।**\n\n"
            "**अभी 108 पर call करें** — Free, 24/7।\n\n"
            "किसी को अपने साथ रहने दें।\n"
            "हम आपके लिए एक doctor को alert कर रहे हैं।\n\n"
            "📞 **108**"
        )
