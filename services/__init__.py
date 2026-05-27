"""
Medivora Backend Services — Memory-Driven Healthcare Modules

Phases:
  1. memory.py        — Patient fact storage + session summaries
  2. emotional.py     — Emotional state detection + continuity
  3. triage.py        — Deterministic pre-Gemini risk scoring
  4. (pgvector)       — Semantic embedding search (see migration 025)
  5. safety.py        — Post-Gemini response safety validator
  6. context_builder.py — Assembles enriched prompt from all layers
"""
