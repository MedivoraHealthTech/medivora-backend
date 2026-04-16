-- Migration 012: Enable RLS on tables that were missing it
-- These tables were publicly accessible without Row-Level Security.
-- Backend uses the service role key, so a permissive service-role policy
-- keeps all existing functionality intact while blocking anon/public access.

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
