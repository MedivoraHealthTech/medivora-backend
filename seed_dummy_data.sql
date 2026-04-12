-- ═══════════════════════════════════════════════════════════════════════════
-- MEDIVORA — Dummy Data Seed
-- Run this in Supabase Dashboard → SQL Editor
-- Safe to re-run (all inserts use ON CONFLICT DO NOTHING)
-- ═══════════════════════════════════════════════════════════════════════════

-- ─────────────────────────────────────────────────────────────────────────
-- 1. DOCTOR PROFILES  (profiles table)
-- ─────────────────────────────────────────────────────────────────────────
INSERT INTO profiles (id, user_type, phone, email, first_name, last_name, password_hash, role, status, phone_verified, email_verified)
VALUES
  ('d0000001-beef-4000-a000-000000000001', 'doctor', '+910000000001', 'sarah.chen@medivora.in',   'Sarah',   'Chen',   '$2b$12$demo_placeholder_hash_____', 'doctor', 'active', true, true),
  ('d0000002-beef-4000-a000-000000000002', 'doctor', '+910000000002', 'james.wilson@medivora.in', 'James',   'Wilson', '$2b$12$demo_placeholder_hash_____', 'doctor', 'active', true, true),
  ('d0000003-beef-4000-a000-000000000003', 'doctor', '+910000000003', 'priya.menon@medivora.in',  'Priya',   'Menon',  '$2b$12$demo_placeholder_hash_____', 'doctor', 'active', true, true),
  ('d0000004-beef-4000-a000-000000000004', 'doctor', '+910000000004', 'arjun.mehta@medivora.in',  'Arjun',   'Mehta',  '$2b$12$demo_placeholder_hash_____', 'doctor', 'active', true, true),
  ('d0000005-beef-4000-a000-000000000005', 'doctor', '+910000000005', 'kavita.reddy@medivora.in', 'Kavita',  'Reddy',  '$2b$12$demo_placeholder_hash_____', 'doctor', 'active', true, true),
  ('d0000006-beef-4000-a000-000000000006', 'doctor', '+910000000006', 'vikram.singh@medivora.in', 'Vikram',  'Singh',  '$2b$12$demo_placeholder_hash_____', 'doctor', 'active', true, true),
  ('d0000007-beef-4000-a000-000000000007', 'doctor', '+910000000007', 'ananya.bose@medivora.in',  'Ananya',  'Bose',   '$2b$12$demo_placeholder_hash_____', 'doctor', 'active', true, true),
  ('d0000008-beef-4000-a000-000000000008', 'doctor', '+910000000008', 'rohit.khanna@medivora.in', 'Rohit',   'Khanna', '$2b$12$demo_placeholder_hash_____', 'doctor', 'active', true, true),
  ('d0000009-beef-4000-a000-000000000009', 'doctor', '+910000000009', 'meera.iyer@medivora.in',   'Meera',   'Iyer',   '$2b$12$demo_placeholder_hash_____', 'doctor', 'active', true, true),
  ('d0000010-beef-4000-a000-000000000010', 'doctor', '+910000000010', 'aditya.patil@medivora.in', 'Aditya',  'Patil',  '$2b$12$demo_placeholder_hash_____', 'doctor', 'active', true, true)
ON CONFLICT (id) DO NOTHING;

-- ─────────────────────────────────────────────────────────────────────────
-- 2. DOCTOR RECORDS  (doctors table)
-- ─────────────────────────────────────────────────────────────────────────
INSERT INTO doctors (id, profile_id, nmc_number, specialties, experience_years, medical_college, clinic_name, clinic_address, consultation_fee, available_status, rating, rating_count, cases_handled)
VALUES
  ('e0000001-beef-4000-b000-000000000001', 'd0000001-beef-4000-a000-000000000001', 'NMC-MED-001', '["Neurology"]',         12, 'AIIMS Mumbai',        'NeuroHealth Clinic',      'Mumbai',    800.00,  'available', 4.8, 142,  320),
  ('e0000002-beef-4000-b000-000000000002', 'd0000002-beef-4000-a000-000000000002', 'NMC-MED-002', '["General Physician"]', 18, 'AIIMS Delhi',         'Wilson Medical Centre',   'Delhi',     500.00,  'available', 4.6, 230,  890),
  ('e0000003-beef-4000-b000-000000000003', 'd0000003-beef-4000-a000-000000000003', 'NMC-MED-003', '["Cardiology"]',        15, 'NIMHANS Bengaluru',   'Heart Care Clinic',       'Bengaluru', 1200.00, 'available', 4.9, 198,  510),
  ('e0000004-beef-4000-b000-000000000004', 'd0000004-beef-4000-a000-000000000004', 'NMC-MED-004', '["Dermatology"]',       9,  'KEM Hospital Pune',   'DermaCare Skin Clinic',   'Pune',      700.00,  'busy',      4.5, 175,  420),
  ('e0000005-beef-4000-b000-000000000005', 'd0000005-beef-4000-a000-000000000005', 'NMC-MED-005', '["Pediatrics"]',        11, 'Osmania Medical',     'Children''s First Clinic', 'Hyderabad', 600.00,  'available', 4.7, 265,  730),
  ('e0000006-beef-4000-b000-000000000006', 'd0000006-beef-4000-a000-000000000006', 'NMC-MED-006', '["Orthopedics"]',       20, 'AIIMS Delhi',         'Bone & Joint Centre',     'Delhi',     900.00,  'offline',   4.6, 310, 1100),
  ('e0000007-beef-4000-b000-000000000007', 'd0000007-beef-4000-a000-000000000007', 'NMC-MED-007', '["Gynecology"]',        14, 'Medical College Kol', 'WomenCare Centre',        'Kolkata',   750.00,  'available', 4.9, 287,  640),
  ('e0000008-beef-4000-b000-000000000008', 'd0000008-beef-4000-a000-000000000008', 'NMC-MED-008', '["Psychiatry"]',        8,  'NIMHANS Bengaluru',   'MindWell Clinic',         'Mumbai',    950.00,  'available', 4.4, 121,  290),
  ('e0000009-beef-4000-b000-000000000009', 'd0000009-beef-4000-a000-000000000009', 'NMC-MED-009', '["ENT"]',               16, 'Stanley Medical',     'ENT Speciality Centre',   'Chennai',   650.00,  'available', 4.8, 204,  580),
  ('e0000010-beef-4000-b000-000000000010', 'd0000010-beef-4000-a000-000000000010', 'NMC-MED-010', '["Pulmonology"]',       13, 'BJ Medical Pune',     'LungCare Clinic',         'Pune',      800.00,  'busy',      4.5, 169,  450)
ON CONFLICT (id) DO NOTHING;

-- ─────────────────────────────────────────────────────────────────────────
-- 3. SAMPLE PATIENT + CONSULTATIONS + PRESCRIPTIONS
--    NOTE: Run this AFTER a real user has signed up via the app.
--    Replace <<YOUR_SUPABASE_AUTH_UID>> with the UUID from:
--    Supabase Dashboard → Authentication → Users → (your test user) → User UID
-- ─────────────────────────────────────────────────────────────────────────

-- Step A: Create a patient record for the test user (skip if already exists)
-- INSERT INTO patients (id, profile_id, name, age, gender)
-- SELECT
--   uuid_generate_v4(),
--   '<<YOUR_SUPABASE_AUTH_UID>>',
--   'Test Patient',
--   28,
--   'unknown'
-- WHERE NOT EXISTS (
--   SELECT 1 FROM patients WHERE profile_id = '<<YOUR_SUPABASE_AUTH_UID>>'
-- );

-- ─────────────────────────────────────────────────────────────────────────
-- 4. (Optional) Verify inserts
-- ─────────────────────────────────────────────────────────────────────────
-- SELECT d.id, TRIM(CONCAT(p.first_name, ' ', p.last_name)) AS full_name, d.specialties, d.clinic_address, d.available_status, d.rating
-- FROM doctors d JOIN profiles p ON d.profile_id = p.id
-- ORDER BY p.last_name;
