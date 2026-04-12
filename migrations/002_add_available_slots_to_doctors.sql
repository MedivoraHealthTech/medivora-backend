-- Migration: Add available_slots JSONB column to doctors table
ALTER TABLE doctors
    ADD COLUMN IF NOT EXISTS available_slots JSONB DEFAULT '[]'::jsonb;
