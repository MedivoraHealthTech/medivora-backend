-- Migration 020b — Seed 9 specialist doctors (simplified, no CTE chaining)
-- Run each INSERT pair together. Profiles first, then doctors via subquery.

-- ── 1. Gynecologist — Dr. Priya Sharma (Gynecology) ──────────────────────
INSERT INTO profiles (user_type, phone, email, first_name, last_name, password_hash, role, status, phone_verified)
VALUES ('doctor', '+919900000001', 'dr.priya.sharma@medivora.in', 'Priya', 'Sharma', 'seeded_by_admin', 'doctor', 'active', TRUE)
ON CONFLICT (phone) DO NOTHING;

INSERT INTO doctors (profile_id, nmc_number, license_verified, specialties, experience_years, medical_college, graduation_year, clinic_name, clinic_address, consultation_fee, available_status, rating, rating_count, cases_handled, sort_order)
SELECT id, 'NMC-GYN-001', TRUE, '["gynecology"]'::jsonb, 14, 'AIIMS New Delhi', 2010, 'Priya Women''s Clinic', 'Andheri West, Mumbai', 900.00, 'available', 4.9, 210, 520, 1
FROM profiles WHERE phone = '+919900000001'
ON CONFLICT (profile_id) DO NOTHING;

-- ── 2. Gynecologist — Dr. Ananya Reddy ───────────────────────────────────
INSERT INTO profiles (user_type, phone, email, first_name, last_name, password_hash, role, status, phone_verified)
VALUES ('doctor', '+919900000002', 'dr.ananya.reddy@medivora.in', 'Ananya', 'Reddy', 'seeded_by_admin', 'doctor', 'active', TRUE)
ON CONFLICT (phone) DO NOTHING;

INSERT INTO doctors (profile_id, nmc_number, license_verified, specialties, experience_years, medical_college, graduation_year, clinic_name, clinic_address, consultation_fee, available_status, rating, rating_count, cases_handled, sort_order)
SELECT id, 'NMC-GYN-002', TRUE, '["gynecology"]'::jsonb, 9, 'Osmania Medical College, Hyderabad', 2015, 'Ananya Maternity & Women''s Health', 'Banjara Hills, Hyderabad', 800.00, 'available', 4.8, 175, 410, 2
FROM profiles WHERE phone = '+919900000002'
ON CONFLICT (profile_id) DO NOTHING;

-- ── 3. Psychiatrist — Dr. Rohan Mehta ────────────────────────────────────
INSERT INTO profiles (user_type, phone, email, first_name, last_name, password_hash, role, status, phone_verified)
VALUES ('doctor', '+919900000003', 'dr.rohan.mehta@medivora.in', 'Rohan', 'Mehta', 'seeded_by_admin', 'doctor', 'active', TRUE)
ON CONFLICT (phone) DO NOTHING;

INSERT INTO doctors (profile_id, nmc_number, license_verified, specialties, experience_years, medical_college, graduation_year, clinic_name, clinic_address, consultation_fee, available_status, rating, rating_count, cases_handled, sort_order)
SELECT id, 'NMC-PSY-001', TRUE, '["psychiatry"]'::jsonb, 11, 'NIMHANS, Bangalore', 2013, 'MindCare Psychiatry Clinic', 'Connaught Place, New Delhi', 1000.00, 'available', 4.9, 300, 680, 3
FROM profiles WHERE phone = '+919900000003'
ON CONFLICT (profile_id) DO NOTHING;

-- ── 4. Psychiatrist — Dr. Kavitha Nair ───────────────────────────────────
INSERT INTO profiles (user_type, phone, email, first_name, last_name, password_hash, role, status, phone_verified)
VALUES ('doctor', '+919900000004', 'dr.kavitha.nair@medivora.in', 'Kavitha', 'Nair', 'seeded_by_admin', 'doctor', 'active', TRUE)
ON CONFLICT (phone) DO NOTHING;

INSERT INTO doctors (profile_id, nmc_number, license_verified, specialties, experience_years, medical_college, graduation_year, clinic_name, clinic_address, consultation_fee, available_status, rating, rating_count, cases_handled, sort_order)
SELECT id, 'NMC-PSY-002', TRUE, '["psychiatry"]'::jsonb, 8, 'Kasturba Medical College, Manipal', 2016, 'Serenity Mental Health Centre', 'Indiranagar, Bangalore', 950.00, 'available', 4.7, 190, 390, 4
FROM profiles WHERE phone = '+919900000004'
ON CONFLICT (profile_id) DO NOTHING;

-- ── 5. Sexologist — Dr. Vikram Joshi ─────────────────────────────────────
INSERT INTO profiles (user_type, phone, email, first_name, last_name, password_hash, role, status, phone_verified)
VALUES ('doctor', '+919900000005', 'dr.vikram.joshi@medivora.in', 'Vikram', 'Joshi', 'seeded_by_admin', 'doctor', 'active', TRUE)
ON CONFLICT (phone) DO NOTHING;

INSERT INTO doctors (profile_id, nmc_number, license_verified, specialties, experience_years, medical_college, graduation_year, clinic_name, clinic_address, consultation_fee, available_status, rating, rating_count, cases_handled, sort_order)
SELECT id, 'NMC-SEX-001', TRUE, '["sexology"]'::jsonb, 16, 'Maulana Azad Medical College, Delhi', 2008, 'Joshi Sexual Health Clinic', 'Lajpat Nagar, New Delhi', 1100.00, 'available', 4.8, 240, 560, 5
FROM profiles WHERE phone = '+919900000005'
ON CONFLICT (profile_id) DO NOTHING;

-- ── 6. Sexologist — Dr. Sunita Patel ─────────────────────────────────────
INSERT INTO profiles (user_type, phone, email, first_name, last_name, password_hash, role, status, phone_verified)
VALUES ('doctor', '+919900000006', 'dr.sunita.patel@medivora.in', 'Sunita', 'Patel', 'seeded_by_admin', 'doctor', 'active', TRUE)
ON CONFLICT (phone) DO NOTHING;

INSERT INTO doctors (profile_id, nmc_number, license_verified, specialties, experience_years, medical_college, graduation_year, clinic_name, clinic_address, consultation_fee, available_status, rating, rating_count, cases_handled, sort_order)
SELECT id, 'NMC-SEX-002', TRUE, '["sexology"]'::jsonb, 12, 'Seth GS Medical College, Mumbai', 2012, 'Patel Women''s Sexual Wellness', 'Juhu, Mumbai', 1000.00, 'available', 4.9, 195, 430, 6
FROM profiles WHERE phone = '+919900000006'
ON CONFLICT (profile_id) DO NOTHING;

-- ── 7. Endocrinologist — Dr. Arjun Krishnan ──────────────────────────────
INSERT INTO profiles (user_type, phone, email, first_name, last_name, password_hash, role, status, phone_verified)
VALUES ('doctor', '+919900000007', 'dr.arjun.krishnan@medivora.in', 'Arjun', 'Krishnan', 'seeded_by_admin', 'doctor', 'active', TRUE)
ON CONFLICT (phone) DO NOTHING;

INSERT INTO doctors (profile_id, nmc_number, license_verified, specialties, experience_years, medical_college, graduation_year, clinic_name, clinic_address, consultation_fee, available_status, rating, rating_count, cases_handled, sort_order)
SELECT id, 'NMC-END-001', TRUE, '["endocrinology"]'::jsonb, 15, 'Madras Medical College, Chennai', 2009, 'Krishnan Diabetes & Hormone Clinic', 'T Nagar, Chennai', 1050.00, 'available', 4.8, 280, 620, 7
FROM profiles WHERE phone = '+919900000007'
ON CONFLICT (profile_id) DO NOTHING;

-- ── 8. Endocrinologist — Dr. Deepa Iyer ──────────────────────────────────
INSERT INTO profiles (user_type, phone, email, first_name, last_name, password_hash, role, status, phone_verified)
VALUES ('doctor', '+919900000008', 'dr.deepa.iyer@medivora.in', 'Deepa', 'Iyer', 'seeded_by_admin', 'doctor', 'active', TRUE)
ON CONFLICT (phone) DO NOTHING;

INSERT INTO doctors (profile_id, nmc_number, license_verified, specialties, experience_years, medical_college, graduation_year, clinic_name, clinic_address, consultation_fee, available_status, rating, rating_count, cases_handled, sort_order)
SELECT id, 'NMC-END-002', TRUE, '["endocrinology"]'::jsonb, 10, 'St. John''s Medical College, Bangalore', 2014, 'Iyer Endocrine & Thyroid Centre', 'Koramangala, Bangalore', 950.00, 'available', 4.7, 165, 370, 8
FROM profiles WHERE phone = '+919900000008'
ON CONFLICT (profile_id) DO NOTHING;

-- ── 9. Dermatologist — Dr. Aarav Gupta ───────────────────────────────────
INSERT INTO profiles (user_type, phone, email, first_name, last_name, password_hash, role, status, phone_verified)
VALUES ('doctor', '+919900000009', 'dr.aarav.gupta@medivora.in', 'Aarav', 'Gupta', 'seeded_by_admin', 'doctor', 'active', TRUE)
ON CONFLICT (phone) DO NOTHING;

INSERT INTO doctors (profile_id, nmc_number, license_verified, specialties, experience_years, medical_college, graduation_year, clinic_name, clinic_address, consultation_fee, available_status, rating, rating_count, cases_handled, sort_order)
SELECT id, 'NMC-DRM-001', TRUE, '["dermatology"]'::jsonb, 7, 'B.J. Medical College, Pune', 2017, 'Gupta Skin & Aesthetic Clinic', 'FC Road, Pune', 750.00, 'available', 4.8, 150, 310, 9
FROM profiles WHERE phone = '+919900000009'
ON CONFLICT (profile_id) DO NOTHING;
