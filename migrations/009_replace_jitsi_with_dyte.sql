-- Migration 009: Replace Jitsi columns with Dyte meeting ID
ALTER TABLE consultations ADD COLUMN IF NOT EXISTS dyte_meeting_id VARCHAR(255);
ALTER TABLE consultations DROP COLUMN IF EXISTS room_name;
ALTER TABLE consultations DROP COLUMN IF EXISTS room_url;
ALTER TABLE consultations DROP COLUMN IF EXISTS patient_token;
ALTER TABLE consultations DROP COLUMN IF EXISTS doctor_token;
