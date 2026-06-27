"""
Medivora Context Builder — Phase 6

Assembles the enriched prompt context from all layers before sending to Gemini.
This is the single place where all context is combined:
  - Patient memory (facts + past summaries)
  - Emotional context note
  - Triage pre-assessment
  - Original user message

Output is the `adapted_message` string that replaces the raw user message
as input to the ADK agent.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ContextBuilder:
    """
    Builds the enriched message sent to Gemini.

    Usage:
        builder = ContextBuilder()
        adapted = builder.build(
            message=raw_message,
            memory_note=memory_svc.format_context_for_prompt(ctx),
            emotional_note=emotional_ctx["system_note"],
            triage_note=triage_result.as_prompt_note(),
        )
    """

    def build(
        self,
        message: str,
        memory_note: str = "",
        emotional_note: str = "",
        triage_note: str = "",
    ) -> str:
        """
        Combine all context layers into a single enriched message.

        Order (most important last, so Gemini reads the actual message fresh):
          1. Patient memory (background context)
          2. Triage pre-assessment (structured medical context)
          3. Emotional note (tone/style instruction)
          4. Original user message
        """
        parts = []

        if memory_note:
            parts.append(memory_note)

        if triage_note:
            parts.append(triage_note)

        if emotional_note:
            parts.append(emotional_note)

        # Always end with the actual user message
        parts.append(message)

        return "\n\n".join(p for p in parts if p)

    def build_first_message(
        self,
        message: str,
        memory_note: str = "",
        emotional_note: str = "",
        triage_note: str = "",
        patient_name: Optional[str] = None,
    ) -> str:
        """
        Enriched build for the first message of a session.
        Adds a returning-patient greeting hint if patient_name is known.
        """
        parts = []

        if memory_note:
            parts.append(memory_note)

        if patient_name and memory_note:
            # Returning patient — add greeting hint
            parts.append(
                f"[SYSTEM NOTE: This is a returning patient named {patient_name}. "
                f"Greet them warmly by name and acknowledge that you remember them "
                f"before asking for their current concern.]"
            )

        if triage_note:
            parts.append(triage_note)

        if emotional_note:
            parts.append(emotional_note)

        parts.append(message)
        return "\n\n".join(p for p in parts if p)
