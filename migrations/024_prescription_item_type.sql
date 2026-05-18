-- ═══════════════════════════════════════════════════════════════════════
-- Migration 024: add item_type to prescription_items
-- ═══════════════════════════════════════════════════════════════════════

ALTER TABLE prescription_items
  ADD COLUMN IF NOT EXISTS item_type VARCHAR(20) DEFAULT 'medicine'
    CHECK (item_type IN ('medicine', 'lab_test'));
