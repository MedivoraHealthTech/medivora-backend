-- Migration 004: Drop full_name column from profiles, use first_name + last_name
-- Run this ONCE against the live database.

-- Backfill first_name/last_name from full_name for all existing rows
UPDATE profiles
SET
  first_name = TRIM(SPLIT_PART(full_name, ' ', 1)),
  last_name  = TRIM(SUBSTR(full_name, STRPOS(full_name, ' ') + 1))
WHERE full_name IS NOT NULL
  AND full_name NOT IN ('User', 'Deleted Account')
  AND full_name NOT LIKE 'User_%'
  AND (first_name IS NULL OR first_name = '');

-- Drop full_name
ALTER TABLE profiles DROP COLUMN IF EXISTS full_name;
