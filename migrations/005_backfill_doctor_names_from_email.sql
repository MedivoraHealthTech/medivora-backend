-- Migration 005: Backfill first_name/last_name for doctor profiles that have
-- empty names but a medivora.in email in the format firstname.lastname@medivora.in
-- Run this ONCE against the live database.

UPDATE profiles
SET
  first_name = INITCAP(SPLIT_PART(SPLIT_PART(email, '@', 1), '.', 1)),
  last_name  = INITCAP(SPLIT_PART(SPLIT_PART(email, '@', 1), '.', 2))
WHERE (first_name IS NULL OR first_name = '')
  AND email LIKE '%@medivora.in'
  AND email ~ '^[a-z]+\.[a-z]+@medivora\.in$';
