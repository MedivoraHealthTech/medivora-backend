-- Migration 029: Expand doctor_join_requests status values
-- Adds: draft, submitted, under_review, changes_requested
-- Also adds: review_note column for admin feedback on changes_requested

-- Step 1: Drop old check constraint
ALTER TABLE doctor_join_requests DROP CONSTRAINT IF EXISTS doctor_join_requests_status_check;

-- Step 2: Migrate existing 'pending' rows to 'draft' BEFORE adding new constraint
UPDATE doctor_join_requests SET status = 'draft' WHERE status = 'pending';

-- Step 3: Add new, expanded check constraint (all rows are now valid)
ALTER TABLE doctor_join_requests ADD CONSTRAINT doctor_join_requests_status_check
    CHECK (status IN ('draft', 'submitted', 'under_review', 'approved', 'rejected', 'changes_requested'));

-- Step 4: Add review_note column (stores admin feedback for changes_requested)
ALTER TABLE doctor_join_requests ADD COLUMN IF NOT EXISTS review_note TEXT;
