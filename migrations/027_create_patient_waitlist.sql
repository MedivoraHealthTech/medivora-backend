-- Migration 027: Create patient_waitlist table
CREATE TABLE IF NOT EXISTS patient_waitlist (
    id         UUID        DEFAULT uuid_generate_v4() PRIMARY KEY,
    name       TEXT        NOT NULL,
    phone      TEXT        NOT NULL,
    email      TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE patient_waitlist ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Service role full access" ON patient_waitlist FOR ALL USING (true) WITH CHECK (true);
