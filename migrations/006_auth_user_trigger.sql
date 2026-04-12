-- ─────────────────────────────────────────────────────────────────────────
-- Migration 006: Auto-create profiles + patients on Supabase auth signup
--
-- Creates a trigger on auth.users that fires once when a new auth user is
-- created (phone OTP first verification or email signup). Inserts a minimal
-- profiles row and a patients row immediately, so the app never has to
-- lazy-create them later through scattered fallback code paths.
--
-- Run this in the Supabase SQL Editor (requires postgres/service-role access).
-- ─────────────────────────────────────────────────────────────────────────

-- ── 1. Trigger function ───────────────────────────────────────────────────
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
  -- ── Extract name from metadata (set during supabase.auth.signUp) ──
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
    -- Everything after the first space in full_name
    NULLIF(regexp_replace(
      COALESCE(NEW.raw_user_meta_data->>'full_name', NEW.raw_user_meta_data->>'name', ''),
      '^[^ ]+ ?', ''
    ), ''),
    ''
  );

  -- ── Phone: use auth phone if available; synthetic placeholder for email users ──
  -- The placeholder is unique per user (derived from UUID) so it won't collide.
  v_phone := COALESCE(
    NULLIF(NEW.phone, ''),
    '+00' || substring(replace(NEW.id::text, '-', ''), 1, 10)
  );

  -- ── Insert into profiles ──────────────────────────────────────────────
  -- id = auth.users.id so all FKs resolve naturally.
  -- ON CONFLICT DO NOTHING: safe to run multiple times / for existing users.
  INSERT INTO public.profiles (
    id,
    user_type,
    phone,
    email,
    first_name,
    last_name,
    full_name,
    password_hash,
    role,
    status,
    phone_verified,
    email_verified,
    created_at,
    updated_at
  ) VALUES (
    NEW.id,
    'patient',
    v_phone,
    NULLIF(NEW.email, ''),
    v_first_name,
    v_last_name,
    TRIM(v_first_name || ' ' || v_last_name),
    'supabase_managed',
    'patient',
    'active',
    (NEW.phone IS NOT NULL AND NEW.phone <> ''),
    (NEW.email_confirmed_at IS NOT NULL),
    NOW(),
    NOW()
  )
  ON CONFLICT DO NOTHING;

  -- ── Insert into patients ──────────────────────────────────────────────
  -- Minimal row — all clinical fields default to NULL / empty arrays.
  INSERT INTO public.patients (
    profile_id,
    medical_history,
    allergies,
    current_medications,
    chronic_conditions,
    created_at,
    updated_at
  ) VALUES (
    NEW.id,
    '[]'::jsonb,
    '[]'::jsonb,
    '[]'::jsonb,
    '[]'::jsonb,
    NOW(),
    NOW()
  )
  ON CONFLICT DO NOTHING;

  RETURN NEW;
END;
$$;

-- ── 2. Attach trigger to auth.users ──────────────────────────────────────
-- Drop first so re-running this migration is idempotent.
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;

CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW
  EXECUTE FUNCTION public.handle_new_user();
