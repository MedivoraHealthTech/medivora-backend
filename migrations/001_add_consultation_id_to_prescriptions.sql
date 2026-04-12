-- Migration 001: Add consultation_id to prescriptions
-- Run this against your live Supabase database.

ALTER TABLE prescriptions
    ADD COLUMN IF NOT EXISTS consultation_id UUID REFERENCES consultations(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_prescriptions_consultation ON prescriptions(consultation_id);
