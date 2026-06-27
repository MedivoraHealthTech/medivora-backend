"""
Medivora Emotional Context Service — Phase 2

Responsibilities:
  - Detect emotional state from current message
  - Read persisted emotional state from last session (via memory)
  - Build an emotional briefing injected into the AI prompt
  - Persist detected state back to patient_memory after each turn

Emotional states:
  neutral     — standard interaction
  anxious     — worried, scared, stressed
  grieving    — loss, death, bereavement
  in_crisis   — suicidal ideation, self-harm, severe distress
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Keyword sets ──────────────────────────────────────────────────────────────

_CRISIS_PATTERNS = re.compile(
    r"\b("
    r"suicid|kill myself|end my life|want to die|don't want to live|"
    r"self.harm|cut myself|hurt myself|no reason to live|"
    r"jeena nahi|marna chahta|marna chahti|zindagi khatam"
    r")\b",
    re.IGNORECASE,
)

_GRIEF_PATTERNS = re.compile(
    r"\b("
    r"died|passed away|lost (my|a)|death of|funeral|bereav|mourning|"
    r"mar gaye|chale gaye|kho diya|guzar gaye"
    r")\b",
    re.IGNORECASE,
)

_ANXIOUS_PATTERNS = re.compile(
    r"\b("
    r"scared|worried|anxious|nervous|panic|afraid|fear|stressed|overwhelm|"
    r"darr|pareshan|tension|ghabra|chinta|fikar|dara hua|dara hui"
    r")\b",
    re.IGNORECASE,
)

_SENSITIVE_PATTERNS = re.compile(
    r"\b("
    r"period|menstrual|pregnancy|pregnant|miscarriage|abortion|"
    r"sexual|STI|STD|HIV|infertility|mental health|depression|"
    r"garbhavati|mahavari|mansik"
    r")\b",
    re.IGNORECASE,
)


def detect_current_state(message: str) -> str:
    """
    Detect emotional state from the current message text.
    Returns one of: 'in_crisis', 'grieving', 'anxious', 'sensitive', 'neutral'
    """
    if _CRISIS_PATTERNS.search(message):
        return "in_crisis"
    if _GRIEF_PATTERNS.search(message):
        return "grieving"
    if _ANXIOUS_PATTERNS.search(message):
        return "anxious"
    if _SENSITIVE_PATTERNS.search(message):
        return "sensitive"
    return "neutral"


class EmotionalContextBuilder:
    """
    Builds the emotional briefing injected into the AI system prompt.

    Combines:
      - Current message emotional state (from keyword detection)
      - Persisted state from last session (from patient_memory)
    """

    def build(
        self,
        current_message: str,
        memory_facts: dict,
        is_first_message: bool = False,
    ) -> dict:
        """
        Returns:
          {
            current_state:  str,    # detected from this message
            past_state:     str,    # from last session memory
            system_note:    str,    # inject into adapted_message
            is_high_risk:   bool,   # True for in_crisis
          }
        """
        current_state = detect_current_state(current_message)

        # Read persisted state from last session
        emotional_facts = memory_facts.get("emotional_state", [])
        past_state = emotional_facts[0]["value"] if emotional_facts else "neutral"

        # Determine effective state (current takes priority)
        effective = current_state if current_state != "neutral" else past_state

        system_note = self._build_note(
            current_state, past_state, is_first_message
        )

        return {
            "current_state": current_state,
            "past_state":    past_state,
            "effective":     effective,
            "system_note":   system_note,
            "is_high_risk":  current_state == "in_crisis",
        }

    def _build_note(
        self,
        current: str,
        past: str,
        is_first: bool,
    ) -> str:
        notes = []

        if current == "in_crisis":
            notes.append(
                "[SYSTEM NOTE: CRISIS DETECTED. Lead IMMEDIATELY with: "
                "'Main yahan hoon. Aap akele nahi hain.' / "
                "'I'm here with you. You are not alone.' "
                "Provide iCall (9152987821) and Vandrevala (1860-2662-345) helplines. "
                "Do NOT jump to medical info. Stay emotionally present.]"
            )
        elif current == "grieving":
            notes.append(
                "[SYSTEM NOTE: Patient is grieving. Open with 2 sentences of pure empathy. "
                "Do not rush to medical advice. Acknowledge their loss first.]"
            )
        elif current == "anxious":
            notes.append(
                "[SYSTEM NOTE: Patient seems anxious/worried. "
                "Lead with reassurance before any clinical information. "
                "Keep tone extra warm and calm.]"
            )
        elif current == "sensitive":
            notes.append(
                "[SYSTEM NOTE: Sensitive health topic. "
                "Apply double softness, use non-judgmental language, "
                "reduce information density.]"
            )

        # Returning patient with past distress
        if is_first and past in ("anxious", "grieving", "in_crisis") and current == "neutral":
            notes.append(
                f"[SYSTEM NOTE: Patient was emotionally {past} in their last visit. "
                "Open warmly — check in on how they are feeling before anything else.]"
            )

        return "\n".join(notes)
