-- Migration 008: Track when a patient first joins a consultation call
ALTER TABLE consultations
  ADD COLUMN IF NOT EXISTS patient_joined_at TIMESTAMPTZ;
