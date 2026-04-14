-- Migration 010: Add daily_meeting_id column to consultations
-- This column stores the Daily.co room name for video consultations

ALTER TABLE consultations
  ADD COLUMN IF NOT EXISTS daily_meeting_id VARCHAR(255);
