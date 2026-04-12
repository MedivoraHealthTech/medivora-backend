-- Migration 007: Create faqs table and seed initial FAQ content
-- Run this against the live Supabase database.

-- ── Create table ────────────────────────────────────────────────────────────

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

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies WHERE tablename = 'faqs' AND policyname = 'faqs_public_read'
  ) THEN
    CREATE POLICY "faqs_public_read" ON faqs FOR SELECT USING (TRUE);
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies WHERE tablename = 'faqs' AND policyname = 'faqs_service_all'
  ) THEN
    CREATE POLICY "faqs_service_all" ON faqs USING (TRUE) WITH CHECK (TRUE);
  END IF;
END $$;

-- ── Seed FAQ data ────────────────────────────────────────────────────────────

INSERT INTO faqs (question, points, display_order) VALUES
(
  'Is Medivora a replacement for visiting a doctor in person?',
  '["No — Medivora connects you with real, verified doctors via video consultation.", "It is not a substitute for emergency care. If you have a life-threatening emergency, call 102/108 immediately.", "Our AI triage helps assess severity and routes you to the right specialist, but all final medical decisions are made by licensed doctors."]',
  1
),
(
  'How does the AI triage work? Is it safe?',
  '["You describe your symptoms in Hindi or English and the AI analyses them to estimate severity (mild, moderate, or urgent).", "The triage result is a recommendation — it does not diagnose or prescribe anything on its own.", "A verified doctor reviews your case and provides the actual consultation, diagnosis, and prescription.", "All conversations are encrypted and stored securely."]',
  2
),
(
  'Are the doctors on Medivora verified and licensed?',
  '["Yes. Every doctor undergoes NMC (National Medical Commission) license verification before being listed.", "Doctor profiles display their specialty, experience, and ratings from past consultations.", "You can view a doctor''s credentials before booking an appointment."]',
  3
),
(
  'How do I get a prescription? Is it valid?',
  '["After your consultation, the doctor reviews the AI-suggested prescription, modifies it if needed, and signs it digitally.", "Digital prescriptions issued by Medivora doctors are legally valid in India.", "You can download the signed prescription directly from your account."]',
  4
),
(
  'What happens to my health data and chat history?',
  '["Your symptom data and chat history are private and only visible to you and the doctor you consult.", "Medivora does not sell personal health data to third parties.", "Data is stored with industry-standard encryption and complies with applicable Indian data protection regulations."]',
  5
),
(
  'Can I use Medivora if I don''t have an account yet?',
  '["Yes — you can start a symptom chat without logging in. Your conversation is saved temporarily.", "To book a consultation, receive a prescription, or view your history, you will need to create a free account.", "Sign-up takes under a minute and only requires basic details."]',
  6
),
(
  'What languages does Medivora support?',
  '["The AI agent understands both Hindi and English, and can switch between them mid-conversation.", "Doctor consultations are conducted in the language both you and the doctor are comfortable with.", "More regional language support is planned in future updates."]',
  7
)
ON CONFLICT DO NOTHING;
