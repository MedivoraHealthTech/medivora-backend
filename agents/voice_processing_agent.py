"""
VoiceProcessingAgent — Stub for voice/audio input processing.
Will be implemented when voice consultations are enabled.
"""

import logging

logger = logging.getLogger("medivora.voice")


class VoiceProcessingAgent:
    """Processes audio input for voice-based medical consultations."""

    def __init__(self):
        logger.info("VoiceProcessingAgent initialized (stub mode)")

    async def process_audio(self, audio_file) -> str:
        """Convert audio to text. Returns transcribed text."""
        logger.warning("VoiceProcessingAgent.process_audio called but not yet implemented")
        return ""

    async def is_available(self) -> bool:
        return False
