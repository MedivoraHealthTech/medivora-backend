-- Migration 022 — Seed 2 Neurology doctors (Gurugram, Haryana)

-- 1. Neurologist — Dr. Siddharth Bose
INSERT INTO profiles (user_type, phone, email, first_name, last_name, password_hash, role, status, phone_verified)
VALUES ('doctor', '+919900000010', 'dr.siddharth.bose@medivora.in', 'Siddharth', 'Bose', 'seeded_by_admin', 'doctor', 'active', TRUE)
ON CONFLICT (phone) DO NOTHING;

INSERT INTO doctors (profile_id, nmc_number, license_verified, specialties, experience_years, medical_college, graduation_year, clinic_name, clinic_address, consultation_fee, available_status, rating, rating_count, cases_handled, sort_order)
SELECT id, 'NMC-NEU-001', TRUE, '["neurology"]'::jsonb, 13, 'AIIMS New Delhi', 2011, 'Bose Neuro & Brain Clinic', 'Sector 38, Gurugram, Haryana', 1100.00, 'available', 4.9, 260, 590, 10
FROM profiles WHERE phone = '+919900000010'
ON CONFLICT (profile_id) DO NOTHING;

-- 2. Neurologist — Dr. Nalini Chatterjee
INSERT INTO profiles (user_type, phone, email, first_name, last_name, password_hash, role, status, phone_verified)
VALUES ('doctor', '+919900000011', 'dr.nalini.chatterjee@medivora.in', 'Nalini', 'Chatterjee', 'seeded_by_admin', 'doctor', 'active', TRUE)
ON CONFLICT (phone) DO NOTHING;

INSERT INTO doctors (profile_id, nmc_number, license_verified, specialties, experience_years, medical_college, graduation_year, clinic_name, clinic_address, consultation_fee, available_status, rating, rating_count, cases_handled, sort_order)
SELECT id, 'NMC-NEU-002', TRUE, '["neurology"]'::jsonb, 10, 'NIMHANS, Bangalore', 2014, 'Chatterjee Neurology Centre', 'DLF Phase 2, Gurugram, Haryana', 1000.00, 'available', 4.8, 195, 430, 11
FROM profiles WHERE phone = '+919900000011'
ON CONFLICT (profile_id) DO NOTHING;
