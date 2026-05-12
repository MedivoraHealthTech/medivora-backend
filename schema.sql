-- ═══════════════════════════════════════════════════════════════════════════
-- MEDIVORA — Supabase Database Schema
-- Run this entire file in Supabase SQL Editor (https://supabase.com/dashboard)
-- ═══════════════════════════════════════════════════════════════════════════

-- Drop ALL existing tables + functions for a clean slate
DROP TABLE IF EXISTS promocodes, payments, consultations, safety_violations, drug_blacklist,
  login_attempts, otp_tokens, notifications, approval_requests, chat_messages,
  chat_sessions, triage_assessments, prescription_items, prescriptions,
  lab_tests, reports, medical_records, doctors, patients, profiles CASCADE;

DROP FUNCTION IF EXISTS update_updated_at() CASCADE;
DROP FUNCTION IF EXISTS increment_message_count() CASCADE;

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ─────────────────────────────────────────────────────────────────────────
-- 1. PROFILES — Unified user table (patients, doctors, admins)
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE profiles (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_type     VARCHAR(20) NOT NULL CHECK (user_type IN ('patient', 'doctor', 'admin')),
    phone         VARCHAR(20) UNIQUE NOT NULL,
    email         VARCHAR(255) UNIQUE,
    first_name    VARCHAR(150),
    last_name     VARCHAR(150),
    password_hash VARCHAR(255) NOT NULL,
    role          VARCHAR(50) NOT NULL DEFAULT 'patient',
    status        VARCHAR(50) NOT NULL DEFAULT 'active'
                    CHECK (status IN ('active', 'inactive', 'suspended', 'pending_verification')),
    phone_verified BOOLEAN DEFAULT FALSE,
    email_verified BOOLEAN DEFAULT FALSE,
    last_login     TIMESTAMPTZ,
    created_at     TIMESTAMPTZ DEFAULT NOW(),
    updated_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_profiles_phone ON profiles(phone);
CREATE INDEX idx_profiles_email ON profiles(email);
CREATE INDEX idx_profiles_user_type ON profiles(user_type);
CREATE INDEX idx_profiles_status ON profiles(status);

-- ─────────────────────────────────────────────────────────────────────────
-- 2. PATIENTS — Extended patient information
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE patients (
    id                      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    profile_id              UUID UNIQUE NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,

    -- Demographics
    age                     INT,
    gender                  VARCHAR(20),
    date_of_birth           DATE,
    address                 TEXT,
    city                    VARCHAR(100),
    state                   VARCHAR(100),
    postal_code             VARCHAR(20),

    -- Vitals
    blood_group             VARCHAR(10),
    height_cm               DECIMAL(5,2),
    weight_kg               DECIMAL(5,2),

    -- Emergency contact
    emergency_contact_name  VARCHAR(255),
    emergency_contact_phone VARCHAR(20),
    emergency_contact_relation VARCHAR(100),

    -- Medical (JSONB arrays)
    medical_history         JSONB DEFAULT '[]'::jsonb,
    allergies               JSONB DEFAULT '[]'::jsonb,
    current_medications     JSONB DEFAULT '[]'::jsonb,
    chronic_conditions      JSONB DEFAULT '[]'::jsonb,

    -- Flags
    is_smoker               BOOLEAN DEFAULT FALSE,
    is_alcohol_user         BOOLEAN DEFAULT FALSE,
    is_pregnant             BOOLEAN DEFAULT FALSE,
    is_nursing              BOOLEAN DEFAULT FALSE,

    -- Insurance
    insurance_provider      VARCHAR(255),
    insurance_policy_number VARCHAR(255),

    created_at              TIMESTAMPTZ DEFAULT NOW(),
    updated_at              TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_patients_profile_id ON patients(profile_id);

-- ─────────────────────────────────────────────────────────────────────────
-- 3. DOCTORS — Credentials and specialties
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE doctors (
    id                    UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    profile_id            UUID UNIQUE NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,

    -- License
    nmc_number            VARCHAR(50) UNIQUE,
    license_verified      BOOLEAN DEFAULT FALSE,

    -- Professional
    specialties           JSONB DEFAULT '[]'::jsonb,
    experience_years      INT,
    medical_college       VARCHAR(255),
    graduation_year       INT,

    -- Practice
    clinic_name           VARCHAR(255),
    clinic_address        TEXT,
    clinic_phone          VARCHAR(20),
    consultation_fee      DECIMAL(10,2),

    -- Availability
    available_status      VARCHAR(50) DEFAULT 'offline'
                            CHECK (available_status IN ('available', 'busy', 'offline', 'on_leave', 'suspended', 'inactive')),
    available_slots       JSONB DEFAULT '[]'::jsonb,

    -- Metrics
    rating                DECIMAL(3,2) DEFAULT 0.0,
    rating_count          INT DEFAULT 0,
    cases_handled         INT DEFAULT 0,

    created_at            TIMESTAMPTZ DEFAULT NOW(),
    updated_at            TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_doctors_profile_id ON doctors(profile_id);
CREATE INDEX idx_doctors_nmc ON doctors(nmc_number);
CREATE INDEX idx_doctors_status ON doctors(available_status);

-- ─────────────────────────────────────────────────────────────────────────
-- 4. MEDICAL_RECORDS — Patient medical history entries
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE medical_records (
    id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id        UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,

    record_type       VARCHAR(100) NOT NULL,  -- 'condition', 'surgery', 'allergy', 'vaccination', etc.
    title             VARCHAR(255),
    description       TEXT,
    diagnosis         TEXT,

    onset_date        DATE,
    resolution_date   DATE,
    status            VARCHAR(50) DEFAULT 'ongoing'
                        CHECK (status IN ('ongoing', 'resolved', 'chronic', 'acute')),

    clinical_notes    TEXT,
    treatment_summary TEXT,
    medications       JSONB DEFAULT '[]'::jsonb,

    created_by_doctor_id UUID REFERENCES doctors(id) ON DELETE SET NULL,

    created_at        TIMESTAMPTZ DEFAULT NOW(),
    updated_at        TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_medical_records_patient ON medical_records(patient_id);
CREATE INDEX idx_medical_records_type ON medical_records(record_type);

-- ─────────────────────────────────────────────────────────────────────────
-- 5. REPORTS — Lab reports, imaging, diagnostics
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE reports (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id          UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,

    report_type         VARCHAR(100) NOT NULL,   -- 'blood_test', 'xray', 'mri', 'ct_scan', etc.
    report_name         VARCHAR(255),
    status              VARCHAR(50) DEFAULT 'pending'
                          CHECK (status IN ('pending', 'processing', 'completed', 'cancelled')),

    ordered_by_doctor_id UUID REFERENCES doctors(id) ON DELETE SET NULL,
    referring_condition  VARCHAR(255),

    result_summary      TEXT,
    result_values       JSONB DEFAULT '{}'::jsonb,
    normal_ranges       JSONB DEFAULT '{}'::jsonb,

    file_url            VARCHAR(500),
    file_name           VARCHAR(255),
    file_type           VARCHAR(50),

    ordered_at          TIMESTAMPTZ,
    sample_collected_at TIMESTAMPTZ,
    completed_at        TIMESTAMPTZ,

    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_reports_patient ON reports(patient_id);
CREATE INDEX idx_reports_type ON reports(report_type);
CREATE INDEX idx_reports_status ON reports(status);

-- ─────────────────────────────────────────────────────────────────────────
-- 6. LAB_TESTS — Individual lab test orders and results
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE lab_tests (
    id                    UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id            UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    report_id             UUID REFERENCES reports(id) ON DELETE SET NULL,

    test_name             VARCHAR(255) NOT NULL,
    test_code             VARCHAR(50),
    test_category         VARCHAR(100),

    requested_by_doctor_id UUID REFERENCES doctors(id) ON DELETE SET NULL,
    clinical_indication   TEXT,
    priority              VARCHAR(50) DEFAULT 'routine'
                            CHECK (priority IN ('routine', 'urgent', 'stat')),

    status                VARCHAR(50) DEFAULT 'ordered'
                            CHECK (status IN ('ordered', 'collected', 'processing', 'completed', 'cancelled')),
    result_value          VARCHAR(255),
    result_unit           VARCHAR(50),
    normal_min            DECIMAL(10,2),
    normal_max            DECIMAL(10,2),
    is_abnormal           BOOLEAN,
    result_comment        TEXT,

    ordered_at            TIMESTAMPTZ DEFAULT NOW(),
    sample_collected_at   TIMESTAMPTZ,
    completed_at          TIMESTAMPTZ,

    created_at            TIMESTAMPTZ DEFAULT NOW(),
    updated_at            TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_lab_tests_patient ON lab_tests(patient_id);
CREATE INDEX idx_lab_tests_report ON lab_tests(report_id);
CREATE INDEX idx_lab_tests_status ON lab_tests(status);

-- ─────────────────────────────────────────────────────────────────────────
-- 7. CHAT_SESSIONS — Triage conversation threads
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE chat_sessions (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id      UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,

    title           VARCHAR(500),
    status          VARCHAR(50) DEFAULT 'active'
                      CHECK (status IN ('active', 'paused', 'completed', 'archived')),
    session_type    VARCHAR(50) DEFAULT 'triage'
                      CHECK (session_type IN ('triage', 'consultation', 'follow_up')),

    message_count   INT DEFAULT 0,

    started_at      TIMESTAMPTZ DEFAULT NOW(),
    last_activity   TIMESTAMPTZ DEFAULT NOW(),
    completed_at    TIMESTAMPTZ,

    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_chat_sessions_patient ON chat_sessions(patient_id);
CREATE INDEX idx_chat_sessions_status ON chat_sessions(status);

-- ─────────────────────────────────────────────────────────────────────────
-- 8. CHAT_MESSAGES — Individual messages in triage chats
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE chat_messages (
    id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id        UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,

    sender_type       VARCHAR(50) NOT NULL CHECK (sender_type IN ('patient', 'ai', 'doctor')),
    sender_id         UUID,
    message_text      TEXT NOT NULL,
    message_type      VARCHAR(50) DEFAULT 'text'
                        CHECK (message_type IN ('text', 'voice', 'file', 'image')),

    emotional_context VARCHAR(100),

    attachment_url    VARCHAR(500),
    attachment_type   VARCHAR(50),

    created_at        TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_chat_messages_session ON chat_messages(session_id);
CREATE INDEX idx_chat_messages_created ON chat_messages(created_at);

-- ─────────────────────────────────────────────────────────────────────────
-- 9. TRIAGE_ASSESSMENTS — AI-generated medical assessments
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE triage_assessments (
    id                      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id              UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    session_id              UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,

    chief_complaint         VARCHAR(500),
    symptoms                JSONB DEFAULT '[]'::jsonb,
    symptom_severity        INT,
    symptom_duration        VARCHAR(100),

    preliminary_diagnosis   JSONB DEFAULT '[]'::jsonb,
    differential_diagnoses  JSONB DEFAULT '[]'::jsonb,
    risk_level              VARCHAR(50)
                              CHECK (risk_level IN ('low', 'medium', 'high', 'critical', 'emergency')),
    confidence_score        DECIMAL(3,2),

    recommendations         JSONB DEFAULT '[]'::jsonb,
    suggested_specialty     VARCHAR(100),
    suggested_tests         JSONB DEFAULT '[]'::jsonb,

    clinical_reasoning      TEXT,
    warning_signs           JSONB DEFAULT '[]'::jsonb,
    red_flags               JSONB DEFAULT '[]'::jsonb,

    follow_up_required      BOOLEAN DEFAULT FALSE,
    follow_up_days          INT,
    follow_up_instructions  TEXT,

    assigned_doctor_id      UUID REFERENCES doctors(id) ON DELETE SET NULL,
    requires_doctor_approval BOOLEAN DEFAULT TRUE,
    approval_status         VARCHAR(50) DEFAULT 'pending'
                              CHECK (approval_status IN ('pending', 'approved', 'rejected', 'modified')),

    is_pregnant             BOOLEAN DEFAULT FALSE,

    created_at              TIMESTAMPTZ DEFAULT NOW(),
    updated_at              TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_triage_patient ON triage_assessments(patient_id);
CREATE INDEX idx_triage_session ON triage_assessments(session_id);
CREATE INDEX idx_triage_risk ON triage_assessments(risk_level);
CREATE INDEX idx_triage_approval ON triage_assessments(approval_status);

-- ─────────────────────────────────────────────────────────────────────────
-- 10. PRESCRIPTIONS — Prescription documents
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE prescriptions (
    id                      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id              UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    assessment_id           UUID REFERENCES triage_assessments(id) ON DELETE SET NULL,

    prescription_number     VARCHAR(50) UNIQUE,
    status                  VARCHAR(50) DEFAULT 'draft'
                              CHECK (status IN ('draft', 'pending_approval', 'approved', 'modified', 'rejected', 'dispensed', 'expired', 'cancelled')),

    prescribed_by_doctor_id UUID REFERENCES doctors(id) ON DELETE SET NULL,

    general_instructions    JSONB DEFAULT '[]'::jsonb,
    dietary_advice          JSONB DEFAULT '[]'::jsonb,
    warning_signs           JSONB DEFAULT '[]'::jsonb,
    follow_up_instructions  TEXT,

    validity_days           INT DEFAULT 30,
    prescribed_at           TIMESTAMPTZ DEFAULT NOW(),
    approved_at             TIMESTAMPTZ,
    expires_at              TIMESTAMPTZ,

    pdf_url                 VARCHAR(500),
    digital_signature       VARCHAR(500),
    pdf_content_hash        VARCHAR(128),

    created_at              TIMESTAMPTZ DEFAULT NOW(),
    updated_at              TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_prescriptions_patient ON prescriptions(patient_id);
CREATE INDEX idx_prescriptions_doctor ON prescriptions(prescribed_by_doctor_id);
CREATE INDEX idx_prescriptions_status ON prescriptions(status);

-- ─────────────────────────────────────────────────────────────────────────
-- 11. PRESCRIPTION_ITEMS — Individual medicines
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE prescription_items (
    id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    prescription_id   UUID NOT NULL REFERENCES prescriptions(id) ON DELETE CASCADE,

    medicine_name     VARCHAR(255) NOT NULL,
    generic_name      VARCHAR(255),
    strength          VARCHAR(100),
    form              VARCHAR(50),           -- tablet, capsule, syrup, etc.

    dosage            VARCHAR(100),
    frequency         VARCHAR(100),
    duration          VARCHAR(100),
    instructions      TEXT,

    before_food       BOOLEAN,

    contraindications JSONB DEFAULT '[]'::jsonb,
    side_effects      JSONB DEFAULT '[]'::jsonb,
    is_blacklisted    BOOLEAN DEFAULT FALSE,

    cost_estimate     DECIMAL(10,2),

    created_at        TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_rx_items_prescription ON prescription_items(prescription_id);

-- ─────────────────────────────────────────────────────────────────────────
-- 12. APPROVAL_REQUESTS — Doctor approval workflow
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE approval_requests (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id          UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    assessment_id       UUID NOT NULL REFERENCES triage_assessments(id) ON DELETE CASCADE,
    prescription_id     UUID REFERENCES prescriptions(id) ON DELETE SET NULL,

    status              VARCHAR(50) DEFAULT 'pending'
                          CHECK (status IN ('pending', 'approved', 'rejected', 'modified', 'cancelled')),
    priority            INT DEFAULT 3 CHECK (priority BETWEEN 1 AND 5),

    assigned_doctor_id  UUID REFERENCES doctors(id) ON DELETE SET NULL,

    ai_assessment       JSONB NOT NULL,
    proposed_prescription JSONB NOT NULL,
    doctor_notes        TEXT,
    rejection_reason    TEXT,
    original_prescription JSONB,
    modified_prescription JSONB,
    doctor_feedback     TEXT,

    nmc_number          VARCHAR(50),
    signature_hash      VARCHAR(255),

    created_at          TIMESTAMPTZ DEFAULT NOW(),
    responded_at        TIMESTAMPTZ,
    expires_at          TIMESTAMPTZ
);

CREATE INDEX idx_approvals_patient ON approval_requests(patient_id);
CREATE INDEX idx_approvals_doctor ON approval_requests(assigned_doctor_id);
CREATE INDEX idx_approvals_status ON approval_requests(status);
CREATE INDEX idx_approvals_priority ON approval_requests(priority);

-- ─────────────────────────────────────────────────────────────────────────
-- 13. NOTIFICATIONS — In-app notifications
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE notifications (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id             UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,

    notification_type   VARCHAR(100),
    title               VARCHAR(255),
    message             TEXT,

    related_entity_type VARCHAR(100),
    related_entity_id   UUID,

    is_read             BOOLEAN DEFAULT FALSE,
    read_at             TIMESTAMPTZ,

    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_notifications_user ON notifications(user_id);
CREATE INDEX idx_notifications_read ON notifications(is_read);

-- ─────────────────────────────────────────────────────────────────────────
-- 14. DRUG_BLACKLIST — Restricted drugs
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE drug_blacklist (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    drug_name           VARCHAR(255) UNIQUE NOT NULL,
    generic_name        VARCHAR(255),
    category            VARCHAR(100),         -- schedule_x, pregnancy_contraindicated, etc.
    reason              TEXT,
    alternative_drugs   JSONB DEFAULT '[]'::jsonb,
    is_active           BOOLEAN DEFAULT TRUE,
    added_by            UUID REFERENCES profiles(id) ON DELETE SET NULL,

    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_blacklist_name ON drug_blacklist(drug_name);
CREATE INDEX idx_blacklist_active ON drug_blacklist(is_active);

-- ─────────────────────────────────────────────────────────────────────────
-- 15. SAFETY_VIOLATIONS — Audit log for drug safety
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE safety_violations (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    violation_type      VARCHAR(100) NOT NULL,   -- blacklist_filter, hard_nsaid_rejection, etc.
    severity            VARCHAR(50) CHECK (severity IN ('low', 'medium', 'high', 'critical')),

    patient_id          UUID REFERENCES patients(id) ON DELETE SET NULL,
    prescription_id     UUID REFERENCES prescriptions(id) ON DELETE SET NULL,
    approval_id         UUID REFERENCES approval_requests(id) ON DELETE SET NULL,

    drug_name           VARCHAR(255),
    description         TEXT,
    patient_context     JSONB,
    action_taken        VARCHAR(500),

    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_violations_type ON safety_violations(violation_type);
CREATE INDEX idx_violations_severity ON safety_violations(severity);
CREATE INDEX idx_violations_patient ON safety_violations(patient_id);

-- ─────────────────────────────────────────────────────────────────────────
-- 16. OTP_TOKENS — OTP management
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE otp_tokens (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    phone           VARCHAR(20) NOT NULL,
    otp_code        VARCHAR(10) NOT NULL,
    otp_type        VARCHAR(50) DEFAULT 'login'
                      CHECK (otp_type IN ('login', 'verification', 'password_reset')),
    is_used         BOOLEAN DEFAULT FALSE,
    expires_at      TIMESTAMPTZ NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_otp_phone ON otp_tokens(phone);
CREATE INDEX idx_otp_expires ON otp_tokens(expires_at);

-- ─────────────────────────────────────────────────────────────────────────
-- 17. LOGIN_ATTEMPTS — Security tracking
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE login_attempts (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    phone           VARCHAR(20),
    user_type       VARCHAR(50),
    success         BOOLEAN NOT NULL,
    ip_address      VARCHAR(50),
    user_agent      TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_login_phone ON login_attempts(phone);
CREATE INDEX idx_login_created ON login_attempts(created_at);

-- ─────────────────────────────────────────────────────────────────────────
-- 18. CONSULTATIONS — Video/phone consultation sessions
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE consultations (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id          UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    doctor_id           UUID REFERENCES doctors(id) ON DELETE SET NULL,

    consultation_type   VARCHAR(50) CHECK (consultation_type IN ('video', 'phone', 'in_person')),
    status              VARCHAR(50) DEFAULT 'requested'
                          CHECK (status IN ('requested', 'scheduled', 'ongoing', 'completed', 'cancelled', 'no_show')),
    specialty           VARCHAR(100),
    patient_note        TEXT,

    scheduled_at        TIMESTAMPTZ,
    started_at          TIMESTAMPTZ,
    patient_joined_at   TIMESTAMPTZ,
    completed_at        TIMESTAMPTZ,
    duration_minutes    INT,

    daily_meeting_id    VARCHAR(255),

    summary             TEXT,
    follow_up_plan      TEXT,

    payment_id          TEXT,
    payment_order_id    TEXT,

    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_consultations_patient ON consultations(patient_id);
CREATE INDEX idx_consultations_doctor ON consultations(doctor_id);
CREATE INDEX idx_consultations_status ON consultations(status);

-- ─────────────────────────────────────────────────────────────────────────
-- 19. PAYMENTS — Payment records
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE payments (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id          UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    doctor_id           UUID REFERENCES doctors(id) ON DELETE SET NULL,
    consultation_id     UUID REFERENCES consultations(id) ON DELETE SET NULL,

    amount              DECIMAL(10,2) NOT NULL,
    currency            VARCHAR(10) DEFAULT 'INR',
    status              VARCHAR(50) DEFAULT 'pending'
                          CHECK (status IN ('pending', 'completed', 'failed', 'refunded')),

    payment_method      VARCHAR(100),
    transaction_id      VARCHAR(255) UNIQUE,
    gateway_order_id    VARCHAR(255),
    gateway_response    JSONB,

    initiated_at        TIMESTAMPTZ DEFAULT NOW(),
    completed_at        TIMESTAMPTZ,

    refund_amount       DECIMAL(10,2),
    refund_reason       TEXT,
    refunded_at         TIMESTAMPTZ,

    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_payments_patient ON payments(patient_id);
CREATE INDEX idx_payments_status ON payments(status);
CREATE INDEX idx_payments_txn ON payments(transaction_id);

-- ─────────────────────────────────────────────────────────────────────────
-- AUTO-UPDATE TRIGGERS for updated_at columns
-- ─────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_profiles_updated BEFORE UPDATE ON profiles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER trg_patients_updated BEFORE UPDATE ON patients
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER trg_doctors_updated BEFORE UPDATE ON doctors
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER trg_medical_records_updated BEFORE UPDATE ON medical_records
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER trg_reports_updated BEFORE UPDATE ON reports
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER trg_lab_tests_updated BEFORE UPDATE ON lab_tests
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER trg_triage_updated BEFORE UPDATE ON triage_assessments
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER trg_prescriptions_updated BEFORE UPDATE ON prescriptions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER trg_consultations_updated BEFORE UPDATE ON consultations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ─────────────────────────────────────────────────────────────────────────
-- AUTO-CREATE profiles + patients ON SUPABASE AUTH SIGNUP
-- Fires once when auth.users gets a new row (phone OTP first verification
-- or email signup). Eliminates all lazy-create fallback code paths.
-- ─────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER SET search_path = public
AS $$
DECLARE
  v_first_name TEXT;
  v_last_name  TEXT;
  v_phone      TEXT;
BEGIN
  v_first_name := COALESCE(
    NULLIF(NEW.raw_user_meta_data->>'first_name', ''),
    NULLIF(split_part(
      COALESCE(NEW.raw_user_meta_data->>'full_name', NEW.raw_user_meta_data->>'name', ''),
      ' ', 1
    ), ''),
    ''
  );
  v_last_name := COALESCE(
    NULLIF(NEW.raw_user_meta_data->>'last_name', ''),
    NULLIF(regexp_replace(
      COALESCE(NEW.raw_user_meta_data->>'full_name', NEW.raw_user_meta_data->>'name', ''),
      '^[^ ]+ ?', ''
    ), ''),
    ''
  );
  -- Phone: use auth phone; synthetic placeholder for email-only users
  v_phone := COALESCE(
    NULLIF(NEW.phone, ''),
    '+00' || substring(replace(NEW.id::text, '-', ''), 1, 10)
  );

  INSERT INTO public.profiles (
    id, user_type, phone, email, first_name, last_name,
    password_hash, role, status,
    phone_verified, email_verified, created_at, updated_at
  ) VALUES (
    NEW.id, 'patient', v_phone, NULLIF(NEW.email, ''), v_first_name, v_last_name,
    'supabase_managed', 'patient', 'active',
    (NEW.phone IS NOT NULL AND NEW.phone <> ''),
    (NEW.email_confirmed_at IS NOT NULL),
    NOW(), NOW()
  )
  ON CONFLICT DO NOTHING;

  INSERT INTO public.patients (
    profile_id, medical_history, allergies,
    current_medications, chronic_conditions, created_at, updated_at
  ) VALUES (
    NEW.id, '[]'::jsonb, '[]'::jsonb, '[]'::jsonb, '[]'::jsonb, NOW(), NOW()
  )
  ON CONFLICT DO NOTHING;

  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- ─────────────────────────────────────────────────────────────────────────
-- AUTO-INCREMENT message_count ON chat_sessions
-- ─────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION increment_message_count()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE chat_sessions
    SET message_count = message_count + 1,
        last_activity = NOW()
    WHERE id = NEW.session_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_message_count
    AFTER INSERT ON chat_messages
    FOR EACH ROW EXECUTE FUNCTION increment_message_count();

-- ─────────────────────────────────────────────────────────────────────────
-- ENABLE RLS (permissive — backend uses service role key)
-- ─────────────────────────────────────────────────────────────────────────
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE patients ENABLE ROW LEVEL SECURITY;
ALTER TABLE doctors ENABLE ROW LEVEL SECURITY;
ALTER TABLE medical_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE reports ENABLE ROW LEVEL SECURITY;
ALTER TABLE lab_tests ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE triage_assessments ENABLE ROW LEVEL SECURITY;
ALTER TABLE prescriptions ENABLE ROW LEVEL SECURITY;
ALTER TABLE prescription_items ENABLE ROW LEVEL SECURITY;
ALTER TABLE approval_requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE notifications ENABLE ROW LEVEL SECURITY;

-- Permissive policies for service role (backend handles auth)
-- These allow the service_role key full access
CREATE POLICY "Service role full access" ON profiles FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access" ON patients FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access" ON doctors FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access" ON medical_records FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access" ON reports FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access" ON lab_tests FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access" ON chat_sessions FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access" ON chat_messages FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access" ON triage_assessments FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access" ON prescriptions FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access" ON prescription_items FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access" ON approval_requests FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access" ON notifications FOR ALL USING (true) WITH CHECK (true);

-- ─────────────────────────────────────────────────────────────────────────
-- 20. PROMOCODES — Discount codes managed by admins
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE promocodes (
    id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    code             VARCHAR(50) UNIQUE NOT NULL,
    description      TEXT,
    discount_percent INTEGER NOT NULL CHECK (discount_percent > 0 AND discount_percent <= 100),
    is_active        BOOLEAN NOT NULL DEFAULT TRUE,
    max_uses         INTEGER,          -- NULL = unlimited
    uses_count       INTEGER NOT NULL DEFAULT 0,
    expires_at       TIMESTAMPTZ,      -- NULL = never expires
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_promocodes_code ON promocodes(code);

ALTER TABLE promocodes ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Service role full access" ON promocodes FOR ALL USING (true) WITH CHECK (true);

-- Dummy promo codes
INSERT INTO promocodes (code, description, discount_percent, is_active, max_uses, expires_at) VALUES
  ('MEDIVORA10', 'Welcome offer — 10% off your first consultation', 10, TRUE, NULL, NULL),
  ('HEALTH20',   '20% off for health awareness month',             20, TRUE, 100,  NOW() + INTERVAL '90 days'),
  ('SAVE15',     'Flat 15% off on all consultations',              15, TRUE, NULL, NULL),
  ('FIRST50',    '50% off for first-time users',                   50, TRUE, 200,  NOW() + INTERVAL '30 days'),
  ('CARE25',     'Special care discount — 25% off',                25, TRUE, 50,   NOW() + INTERVAL '60 days');

-- ─────────────────────────────────────────────────────────────────────────
-- POST-CREATION: Add consultation_id FK to prescriptions
-- (prescriptions is defined before consultations, so FK added here)
-- ─────────────────────────────────────────────────────────────────────────
ALTER TABLE prescriptions
    ADD COLUMN IF NOT EXISTS consultation_id UUID REFERENCES consultations(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_prescriptions_consultation ON prescriptions(consultation_id);

-- ─────────────────────────────────────────────────────────────────────────
-- POST-CREATION: Add available_slots JSONB to doctors
-- ─────────────────────────────────────────────────────────────────────────
ALTER TABLE doctors
    ADD COLUMN IF NOT EXISTS available_slots JSONB DEFAULT '[]'::jsonb;

-- ─────────────────────────────────────────────────────────────────────────
-- TABLE: faqs
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS faqs (
    id             UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    question       TEXT        NOT NULL,
    points         JSONB       NOT NULL DEFAULT '[]'::jsonb,
    display_order  INT         NOT NULL DEFAULT 0,
    is_active      BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_faqs_active_order ON faqs(is_active, display_order);

ALTER TABLE faqs ENABLE ROW LEVEL SECURITY;
CREATE POLICY "faqs_public_read" ON faqs FOR SELECT USING (TRUE);
CREATE POLICY "faqs_service_all"  ON faqs USING (TRUE) WITH CHECK (TRUE);

-- ─────────────────────────────────────────────────────────────────────────
-- TABLE: prescription_documents — stores generated/signed PDF metadata
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS prescription_documents (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    approval_id         UUID NOT NULL REFERENCES approval_requests(id) ON DELETE CASCADE,
    verification_token  VARCHAR(255) UNIQUE,
    document_url        TEXT,
    signed_by           UUID REFERENCES doctors(id) ON DELETE SET NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_prescription_docs_approval ON prescription_documents(approval_id);
CREATE INDEX IF NOT EXISTS idx_prescription_docs_token ON prescription_documents(verification_token);

-- ─────────────────────────────────────────────────────────────────────────
-- TABLE: prescription_edit_log — audit trail for prescription changes
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS prescription_edit_log (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    approval_id UUID NOT NULL REFERENCES approval_requests(id) ON DELETE CASCADE,
    doctor_id   UUID REFERENCES doctors(id) ON DELETE SET NULL,
    field_name  TEXT,
    old_value   TEXT,
    new_value   TEXT,
    change_type VARCHAR(50),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_edit_log_approval ON prescription_edit_log(approval_id);

-- ═══════════════════════════════════════════════════════════════════════════
ALTER TABLE consultations          ENABLE ROW LEVEL SECURITY;
ALTER TABLE payments               ENABLE ROW LEVEL SECURITY;
ALTER TABLE drug_blacklist         ENABLE ROW LEVEL SECURITY;
ALTER TABLE safety_violations      ENABLE ROW LEVEL SECURITY;
ALTER TABLE otp_tokens             ENABLE ROW LEVEL SECURITY;
ALTER TABLE login_attempts         ENABLE ROW LEVEL SECURITY;
ALTER TABLE prescription_documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE prescription_edit_log  ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Service role full access" ON consultations          FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access" ON payments               FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access" ON drug_blacklist         FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access" ON safety_violations      FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access" ON otp_tokens             FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access" ON login_attempts         FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access" ON prescription_documents FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access" ON prescription_edit_log  FOR ALL USING (true) WITH CHECK (true);

-- ─── Doctor Join Requests ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS doctor_join_requests (
    id                UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    first_name        TEXT        NOT NULL,
    last_name         TEXT        NOT NULL DEFAULT '',
    phone             TEXT        NOT NULL,
    email             TEXT        NOT NULL DEFAULT '',
    specialties       TEXT        NOT NULL DEFAULT 'general_medicine',
    experience_years  INT         NOT NULL DEFAULT 0,
    medical_college   TEXT        NOT NULL DEFAULT '',
    nmc_number        TEXT        NOT NULL DEFAULT '',
    clinic_name       TEXT        NOT NULL DEFAULT '',
    clinic_address    TEXT        NOT NULL DEFAULT '',
    consultation_fee  DECIMAL(10,2),
    notes             TEXT        NOT NULL DEFAULT '',
    status            TEXT        NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending', 'approved', 'rejected')),
    reviewed_at       TIMESTAMPTZ,
    reviewed_by       UUID,
    doctor_id         UUID        REFERENCES doctors(id),  -- set for self-registered doctors
    created_at        TIMESTAMPTZ DEFAULT NOW()
);
ALTER TABLE doctor_join_requests ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Service role full access" ON doctor_join_requests FOR ALL USING (true) WITH CHECK (true);

-- ─── Doctor Waitlist ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS doctor_waitlist (
    id         UUID        DEFAULT uuid_generate_v4() PRIMARY KEY,
    name       TEXT        NOT NULL,
    phone      TEXT        NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE doctor_waitlist ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Service role full access" ON doctor_waitlist FOR ALL USING (true) WITH CHECK (true);

-- ─────────────────────────────────────────────────────────────────────────
-- TABLE: family_members — patient's family / dependents
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS family_members (
    id                   UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id           UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,

    name                 VARCHAR(255) NOT NULL,
    age                  INT,
    gender               VARCHAR(20),
    relationship         VARCHAR(50),   -- spouse, child, parent, sibling, other

    blood_group          VARCHAR(10),
    medical_history      TEXT,
    allergies            TEXT,
    current_medications  TEXT,

    created_at           TIMESTAMPTZ DEFAULT NOW(),
    updated_at           TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_family_members_patient ON family_members(patient_id);

ALTER TABLE family_members ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Service role full access" ON family_members FOR ALL USING (true) WITH CHECK (true);

CREATE TRIGGER trg_family_members_updated BEFORE UPDATE ON family_members
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- DONE! All tables created with indexes, triggers, and RLS policies.
-- ═══════════════════════════════════════════════════════════════════════════
