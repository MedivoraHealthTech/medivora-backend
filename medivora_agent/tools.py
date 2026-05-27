"""
ADK Tool functions for Medivora Medical AI Assistant.
These wrap existing logic from the agents/ directory and database.py.
"""

import re
import uuid
import asyncio
import logging
from datetime import datetime
from typing import Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import DatabaseManager, _name
from diskcache import Cache
from drug_blacklist import load_blacklist_from_db, filter_prescription, seed_default_blacklist

logger = logging.getLogger(__name__)

# Shared DB instance
_db = DatabaseManager()

# Seed drug blacklist on first run
seed_default_blacklist(_db)

# Current user context (set by API before agent runs, read by create_approval_and_notify)
_current_user_id            = ""
_current_user_name          = ""
_current_user_email         = ""
_current_user_authenticated = False   # True only when a valid JWT was present

def set_current_user_id(user_id: str):
    """Set the current user ID for the agent tools to reference."""
    global _current_user_id
    _current_user_id = user_id or ""

def set_current_user(user_id: str, name: str = "", email: str = "", is_authenticated: bool = False):
    """Set full user context (id, name, email, auth status) for the agent tools to reference."""
    global _current_user_id, _current_user_name, _current_user_email, _current_user_authenticated
    _current_user_id            = user_id or ""
    _current_user_name          = name    or ""
    _current_user_email         = email   or ""
    _current_user_authenticated = is_authenticated

# Disk cache for expensive/repeated lookups (survives restarts)
_cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache")
_cache = Cache(_cache_dir)


def _run_async(coro):
    """Helper to run async DB calls from sync tool functions."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


# ──────────────────────────────────────────────────────────────────
# Triage / Registration tools
# ──────────────────────────────────────────────────────────────────

def check_if_symptoms(user_input: str) -> dict:
    """Check whether the user input contains medical symptoms or is registration/greeting info.
    Returns {is_symptoms: true/false, is_urgent: true/false, detected_keywords: [...]}"""

    symptom_keywords = [
        'fever', 'pain', 'headache', 'cough', 'vomit', 'hurt', 'sick',
        'stomach', 'chest', 'breathing', 'dizziness', 'weakness',
        'blood', 'bleeding', 'wound', 'injury', 'fracture', 'swelling',
        'rash', 'itching', 'burning', 'nausea', 'faint', 'unconscious',
        'seizure', 'accident', 'diarrhea', 'cold', 'flu', 'sore throat',
        'back pain', 'joint', 'allergy', 'asthma', 'diabetes', 'bp',
        'dard', 'bukhar', 'khansi', 'ulti', 'khoon', 'chot', 'sujan',
        'jalan', 'behosh', 'dast', 'kamzori', 'chakkar', 'saans',
        'dard', 'bukhar', 'khansi', 'ulti', 'khoon', 'chot', 'sujan',
        'kamzori', 'chakkar', 'saans', 'pet', 'seena', 'kamar',
        'aaraha', 'aa raha', 'ho raha', 'taklif', 'pareshani', 'bimari',
    ]

    urgent_patterns = [
        r'blood', r'bleed', r'khoon', r'khoon', r'khun',
        r'chest\s*pain', r'[roman].*dard', r'seene.*dard',
        r'breath', r'saans', r'saans', r'unconscious', r'behosh', r'behosh',
        r'seizure', r'fits', r'accident', r'poison', r'stroke', r'paralysis',
        r'suicide', r'dying', r'critical', r'severe',
    ]

    text_lower = user_input.lower()
    detected = [kw for kw in symptom_keywords if kw in text_lower]
    is_symptoms = len(detected) > 0
    is_urgent = any(re.search(p, text_lower) for p in urgent_patterns)

    return {
        "status": "success",
        "is_symptoms": is_symptoms,
        "is_urgent": is_urgent,
        "detected_keywords": detected[:10],
    }


def extract_registration(user_input: str) -> dict:
    """Extract patient registration info (name, age, phone) from user input.
    Returns extracted fields or empty if nothing found."""

    data = {}
    text = user_input.strip()

    # Phone
    phone_match = re.search(r'(?:\+91[\s-]?)?(\d{10,12})', text)
    if phone_match:
        data['phone'] = phone_match.group(0).replace(' ', '').replace('-', '')

    # Age
    age_match = re.search(r'(?:age|umar|[roman]|sal|[roman]|years?|yr)[\s:]*(\d{1,3})', text, re.IGNORECASE)
    if age_match:
        age = int(age_match.group(1))
        if 1 <= age <= 120:
            data['age'] = age
    else:
        age_match = re.search(r'\b(\d{1,2})\b', text)
        if age_match:
            age = int(age_match.group(1))
            if 1 <= age <= 100:
                data['age'] = age

    # Name
    skip_words = {
        'skip', 'hi', 'hello', 'name', 'age', 'phone', 'male', 'female',
        'sal', 'year', 'years', 'umar', 'naam', 'mera', 'meri', 'hai',
    }
    parts = [p.strip() for p in text.split(',')]
    if len(parts) >= 2:
        for part in parts:
            if part and not part.isdigit() and not re.match(r'^\+?\d[\d\s-]+$', part):
                candidate = part.strip()
                if candidate.lower() not in skip_words and len(candidate) >= 2:
                    data['name'] = candidate
                    break

    if 'name' not in data:
        name_match = re.search(
            r'(?:naam|name|[roman])[\s:]+([a-zA-Z\u0900-\u097F]+(?:\s+[a-zA-Z\u0900-\u097F]+)*)',
            text, re.IGNORECASE
        )
        if name_match:
            data['name'] = name_match.group(1).strip()

    # Gender
    if re.search(r'\b(male|[roman]|ladka|mard)\b', text, re.IGNORECASE):
        data['gender'] = 'male'
    elif re.search(r'\b(female|[roman]|ladki|aurat)\b', text, re.IGNORECASE):
        data['gender'] = 'female'

    return {"status": "success", "registration_data": data}


def extract_symptoms(user_input: str) -> dict:
    """Extract structured symptom list from user text.
    Returns list of {name, duration, severity} dicts."""

    symptom_patterns = {
        r'(fever|bukhar|bukhar|temperature)': 'fever',
        r'(headache|sir.*dard|[roman].*dard|migraine)': 'headache',
        r'(cough|khansi|khansi)': 'cough',
        r'(stomach.*pain|pet.*dard|pet.*dard)': 'stomach pain',
        r'(vomiting|ulti|ulti|nausea)': 'vomiting',
        r'(diarrhea|dast|dast|loose.*motion)': 'diarrhea',
        r'(chest.*pain|sine.*dard|[roman].*dard)': 'chest pain',
        r'(shortness.*breath|saans.*problem|saans)': 'breathing difficulty',
        r'(weakness|kamzori|kamzori|fatigue|thakan)': 'weakness',
        r'(dizziness|chakkar|chakkar)': 'dizziness',
        r'(blood|bleed|khoon|khoon)': 'bleeding',
        r'(back.*pain|kamar.*dard|kamar.*dard)': 'back pain',
        r'(joint.*pain|jod.*dard|jod.*dard)': 'joint pain',
        r'(sore.*throat|gala.*dard|gala.*dard)': 'sore throat',
        r'(rash|daane|daane|khujli|khujli)': 'skin rash/itching',
        r'(allergy|[roman])': 'allergy',
        r'(swelling|sujan|sujan)': 'swelling',
    }

    # Duration extraction
    duration = "not specified"
    for pat in [r'(\d+)\s*(day|days|din|[roman])', r'(\d+)\s*(week|hafta)', r'(\d+)\s*(month|mahina)']:
        m = re.search(pat, user_input.lower())
        if m:
            duration = m.group()
            break

    # Severity
    text_lower = user_input.lower()
    if any(w in text_lower for w in ['severe', 'bahut', 'bahut', 'zyada', 'unbearable']):
        severity = 8
    elif any(w in text_lower for w in ['mild', 'thoda', 'thoda', 'kam']):
        severity = 3
    else:
        severity = 5

    symptoms = []
    for pattern, name in symptom_patterns.items():
        if re.search(pattern, text_lower):
            symptoms.append({"name": name, "duration": duration, "severity": severity})

    return {"status": "success", "symptoms": symptoms, "count": len(symptoms)}


# ──────────────────────────────────────────────────────────────────
# Assessment tools
# ──────────────────────────────────────────────────────────────────

def assess_risk(symptoms_text: str) -> dict:
    """Assess risk level based on symptoms. Enhanced with obstetric triage.
    Returns risk_level, action, and flags."""

    text_lower = symptoms_text.lower()

    # ── Detect pregnancy context ──
    pregnancy_keywords = ['pregnan', 'garbh', 'garbh', 'weeks pregnant', 'month pregnant',
                          'trimester', 'conceiv', 'expecting', 'gestational']
    is_pregnant = any(kw in text_lower for kw in pregnancy_keywords)

    # ── Obstetric EMERGENCY flags ──
    if is_pregnant:
        obstetric_emergency = [
            'bleeding', 'bleed', 'blood', 'khoon', 'khoon',
            'seizure', 'fits', 'convulsion',
            'unconscious', 'behosh', 'behosh',
            'severe headache', 'vision change', 'blurred vision',
            'placenta', 'abruption',
        ]
        obstetric_urgent = [
            'contraction', 'tightening', 'cramping', 'labour', 'labor', 'dard aa raha',
            'leaking fluid', 'pani aa raha', 'water break', 'pani toot',
            'reduced movement', 'baby not moving', 'bachcha hil nahi',
            'swelling', 'sujan', 'sujan',
            'high bp', 'blood pressure',
            'pain', 'dard', 'dard',
            'vomiting', 'ulti', 'ulti',
        ]

        if any(kw in text_lower for kw in obstetric_emergency):
            _write_triage_from_risk("EMERGENCY", specialty="womens_health", category="Obstetric Emergency")
            return {
                "status": "success",
                "risk_level": "EMERGENCY",
                "action": "Call 108 immediately. Go to nearest hospital with OBG department.",
                "is_pregnant": True,
                "obstetric_flag": "OBSTETRIC EMERGENCY — immediate hospital evaluation required",
            }
        if any(kw in text_lower for kw in obstetric_urgent):
            _write_triage_from_risk("URGENT", specialty="womens_health", category="Obstetric Concern")
            return {
                "status": "success",
                "risk_level": "URGENT",
                "action": "See OBG doctor within 2-4 hours. Do NOT ignore these symptoms during pregnancy.",
                "is_pregnant": True,
                "obstetric_flag": "OBSTETRIC URGENT — requires prompt OBG evaluation",
            }

    # ── General EMERGENCY flags ──
    # NOTE: 'chest pain' alone is NOT an emergency — it's URGENT.
    # Only clear-cut life-threatening presentations trigger EMERGENCY.
    emergency_keywords = [
        'heart attack', 'cardiac arrest', 'crushing chest',
        'difficulty breathing', 'cannot breathe', 'stopped breathing',
        'unconscious', 'not responding',
        'severe bleeding', 'stroke',
        'seizure', 'convulsion', 'poisoning', 'overdose',
        'anaphylaxis', 'throat swelling',
    ]
    if any(kw in text_lower for kw in emergency_keywords):
        _write_triage_from_risk("EMERGENCY")
        return {"status": "success", "risk_level": "EMERGENCY", "action": "Call 108 immediately"}

    # ── Chest pain alone → URGENT (needs assessment, not instant 108) ──
    if 'chest pain' in text_lower or 'chest tightness' in text_lower:
        _write_triage_from_risk("URGENT")
        return {"status": "success", "risk_level": "URGENT", "action": "See cardiologist urgently — assess severity first"}

    # ── General URGENT flags ──
    urgent_keywords = ['high fever', 'blood', 'severe', 'breathing']
    if any(kw in text_lower for kw in urgent_keywords):
        _write_triage_from_risk("URGENT")
        return {"status": "success", "risk_level": "URGENT", "action": "See doctor within 24 hours"}

    # ── Pregnant but no red flags → still at least ROUTINE with OBG note ──
    if is_pregnant:
        _write_triage_from_risk("ROUTINE", specialty="womens_health", category="Pregnancy Query")
        return {
            "status": "success",
            "risk_level": "ROUTINE",
            "action": "Schedule OBG consultation. Monitor symptoms closely.",
            "is_pregnant": True,
            "obstetric_flag": "Pregnancy detected — OBG specialist recommended",
        }

    result = {"status": "success", "risk_level": "ROUTINE", "action": "Schedule appointment"}
    _write_triage_from_risk("ROUTINE")
    return result


def determine_specialty(symptoms_text: str, diagnosis: str = "") -> dict:
    """Determine the appropriate medical specialty based on symptoms and diagnosis.
    Returns the recommended specialty and confidence."""

    SPECIALTY_MAPPING = {
        "cardiology": {
            "keywords": [
                "chest pain", "heart", "palpitation", "blood pressure", "bp",
                "hypertension", "seene mein dard", "dil", "heartbeat", "cardiac",
                "cholesterol", "[roman] [roman] dard", "dil", "dhadkan", "blood pressure",
            ],
            "diagnoses": ["hypertension", "cardiac", "heart", "angina", "arrhythmia"],
        },
        "womens_health": {
            "keywords": [
                "pregnancy", "period", "menstrual", "menstruation", "pregnant",
                "garbh", "mahwari", "pcod", "pcos", "breast", "ovary", "uterus",
                "garbh", "masik", "period", "stan", "gynec", "gynaec",
            ],
            "diagnoses": ["pregnancy", "menstrual", "gynec", "pcod", "pcos", "obstetric"],
        },
        "pediatrics": {
            "keywords": [
                "child", "baby", "infant", "newborn", "bachcha", "toddler",
                "bachcha", "shishu", "navjat",
            ],
            "diagnoses": ["pediatric", "childhood", "infant", "neonatal"],
        },
        "dermatology": {
            "keywords": [
                "skin", "rash", "itching", "acne", "eczema", "psoriasis", "fungal",
                "daane", "khujli", "tvacha", "daad",
                "tvacha", "daane", "khujli", "daad", "fungal",
            ],
            "diagnoses": ["dermatitis", "eczema", "psoriasis", "acne", "fungal", "skin"],
        },
        "orthopedics": {
            "keywords": [
                "bone", "joint", "fracture", "back pain", "knee", "spine", "arthritis",
                "haddi", "jod", "kamar dard", "ghutna",
                "haddi", "jod", "kamar", "ghutna", "fracture",
            ],
            "diagnoses": ["fracture", "arthritis", "spondylitis", "bone", "joint"],
        },
        "gastroenterology": {
            "keywords": [
                "stomach", "abdomen", "digestion", "liver", "acidity", "ulcer",
                "pet dard", "digestive", "gastric", "constipation", "diarrhea",
                "pet", "liver", "acidity", "kabj", "dast",
            ],
            "diagnoses": ["gastritis", "ulcer", "hepatitis", "liver", "ibs", "gastro"],
        },
        "pulmonology": {
            "keywords": [
                "breathing", "lungs", "asthma", "chronic cough", "tb", "tuberculosis",
                "saans", "phefda", "dama",
                "saans", "phephdaa", "dama", "TB",
            ],
            "diagnoses": ["asthma", "pneumonia", "bronchitis", "tuberculosis", "copd"],
        },
        "neurology": {
            "keywords": [
                "migraine", "seizure", "fits", "numbness", "paralysis",
                "stroke", "nerve", "brain",
                "migraine", "mirgi", "lakwa", "sunn",
            ],
            "diagnoses": ["migraine", "epilepsy", "stroke", "neuropathy", "neurological"],
        },
        "ent": {
            "keywords": [
                "ear", "nose", "throat", "sinus", "tonsil", "hearing",
                "kaan", "naak", "gala",
                "kaan", "naak", "gala", "sinus",
            ],
            "diagnoses": ["sinusitis", "tonsillitis", "otitis", "ent"],
        },
        "ophthalmology": {
            "keywords": [
                "eye", "vision", "blindness", "cataract", "glaucoma",
                "aankh", "nazar",
                "aankh", "nazar",
            ],
            "diagnoses": ["cataract", "glaucoma", "conjunctivitis", "eye"],
        },
    }

    text_lower = (symptoms_text + " " + diagnosis).lower()
    scores = {}

    for specialty, mapping in SPECIALTY_MAPPING.items():
        score = 0
        for keyword in mapping["keywords"]:
            if keyword in text_lower:
                score += 2
        for diag_keyword in mapping["diagnoses"]:
            if diag_keyword in text_lower:
                score += 3
        if score > 0:
            scores[specialty] = score

    if not scores:
        return {
            "status": "success",
            "specialty": "general_medicine",
            "confidence": 0.5,
            "reason": "No specific specialty keywords detected, defaulting to general medicine",
        }

    best_specialty = max(scores, key=scores.get)
    max_score = scores[best_specialty]
    confidence = min(0.95, 0.5 + (max_score * 0.05))

    # Track determined specialty so api.py can include it in additional_data
    if _current_user_id:
        _latest_approval_specialty[_current_user_id] = best_specialty

    return {
        "status": "success",
        "specialty": best_specialty,
        "confidence": round(confidence, 2),
        "reason": f"Matched {best_specialty} based on symptom/diagnosis keywords",
        "all_matches": {k: v for k, v in sorted(scores.items(), key=lambda x: -x[1])},
    }


def assess_and_route(symptoms_text: str, diagnosis: str = "") -> dict:
    """Combined risk assessment + specialty determination in a single call.
    Replaces calling assess_risk and determine_specialty separately — saves one API round-trip.
    Returns risk_level, action, specialty, and confidence in one response."""
    risk = assess_risk(symptoms_text)
    specialty = determine_specialty(symptoms_text, diagnosis)
    return {**risk, **specialty}


def get_nearby_facilities(location: str, risk_level: str) -> dict:
    """Get nearby healthcare facilities based on location and risk level."""
    cache_key = f"facilities:{location}:{risk_level}"
    cached = _cache.get(cache_key)
    if cached:
        return cached

    facilities = {
        "phc": [{"name": "Delhi PHC", "distance": "2km", "cost": "Free"}],
        "chc": [{"name": "Delhi CHC", "distance": "5km", "cost": "Free"}],
        "district": [{"name": "Delhi District Hospital", "distance": "10km", "cost": "Free"}],
        "private": [{"name": "Local Clinic", "distance": "3km", "cost": "₹300-500"}],
    }

    if risk_level == "EMERGENCY":
        result = {"status": "success", "facilities": facilities["district"]}
    elif risk_level == "URGENT":
        result = {"status": "success", "facilities": facilities["chc"] + facilities["private"]}
    else:
        result = {"status": "success", "facilities": facilities["phc"]}

    _cache.set(cache_key, result, expire=3600)  # Cache for 1 hour
    return result


# ──────────────────────────────────────────────────────────────────
# Consultation booking tool
# ──────────────────────────────────────────────────────────────────

# Per-user tracking of the latest consultation booked in this process
# (api.py reads this after the agent run to include in additional_data)
_latest_consultation: dict = {}

# Per-user tracking of the specialty determined during the assessment pipeline
# (api.py reads this to include recommended_specialty in additional_data)
_latest_approval_specialty: dict = {}

# Per-user triage result — level, risk_score, specialty, category
# Populated by assess_risk (early signal) and overwritten by create_approval_and_notify (final)
# api.py reads this to include the triage object in the chat response
_latest_triage: dict = {}

# Maps backend risk levels → frontend triage levels and risk scores
_TRIAGE_LEVEL_MAP = {
    "EMERGENCY": ("emergency", 95),
    "URGENT":    ("high",      70),
    "ROUTINE":   ("medium",    40),
    "HOME_CARE": ("low",       15),
}


def _write_triage_from_risk(risk_level: str, specialty: str = "", category: str = ""):
    """Update _latest_triage for the current user from a risk level string."""
    if not _current_user_id:
        return
    level, score = _TRIAGE_LEVEL_MAP.get(risk_level, ("medium", 40))
    entry = _latest_triage.get(_current_user_id, {})
    entry.update({"level": level, "risk_score": score})
    if specialty:
        entry["recommended_speciality"] = specialty
    if category:
        entry["category"] = category
    _latest_triage[_current_user_id] = entry


def book_consultation(specialty: str, patient_note: str, doctor_id: str = "") -> dict:
    """Book a video consultation with a doctor for the current patient.

    Call this when:
    - Risk level is URGENT or EMERGENCY
    - Patient explicitly requests an appointment / wants to see a doctor
    - Symptoms need specialist review beyond AI guidance

    Args:
        specialty: The medical specialty required (e.g. "cardiology", "general_medicine")
        patient_note: Brief summary of chief complaint and symptoms (max 400 chars)
        doctor_id: Optional. The doctor_id from get_available_doctors. If not provided,
                   the first available doctor for the specialty is auto-assigned.

    Returns a dict with status and consultation_id.
    """
    try:
        user_id = _current_user_id
        if not user_id or not _current_user_authenticated:
            logger.info("book_consultation skipped — user is not authenticated")
            return {
                "status": "login_required",
                "message": "Consultation booking requires a Medivora account. Let the patient know their assessment is ready and they should sign up or log in to book an appointment.",
            }

        consultation_id = str(uuid.uuid4())
        resolved_doctor_id = doctor_id.strip() if doctor_id else ""
        doctor_name = ""

        # Auto-assign an available doctor for the specialty if none specified
        if not resolved_doctor_id:
            try:
                all_doctors = _run_async(_db.get_all_doctors())
                spec_lower = specialty.strip().lower().replace(" ", "_")
                candidates = [
                    d for d in all_doctors
                    if d.get("available_status") == "available"
                    and spec_lower in (d.get("specialization") or "").lower().replace(" ", "_")
                ]
                if not candidates:
                    # Fallback: any available doctor
                    candidates = [d for d in all_doctors if d.get("available_status") == "available"]
                if candidates:
                    resolved_doctor_id = candidates[0].get("id", "")
                    doctor_name = _name(candidates[0].get("first_name"), candidates[0].get("last_name")) or candidates[0].get("full_name", "")
            except Exception as e:
                logger.warning(f"book_consultation: auto-assign doctor failed: {e}")

        data = {
            "id":                consultation_id,
            "user_id":           user_id,
            "patient_name":      _current_user_name or "Patient",
            "patient_email":     _current_user_email or "",
            "specialty":         specialty.strip().lower().replace(" ", "_"),
            "patient_note":      (patient_note or "")[:400],
            "consultation_type": "in_person",
            "status":            "requested",
            "doctor_id":         resolved_doctor_id or None,
        }

        # Use sync method directly — Supabase client is synchronous;
        # _run_async wraps in a new thread/event-loop and swallows errors silently.
        _db.create_consultation_from_chat_sync(data)

        # Store so api.py can include it in the response's additional_data
        _latest_consultation[user_id] = {
            "consultation_id": consultation_id,
            "specialty":       specialty,
            "status":          "requested",
        }

        logger.info(f"Consultation booked: {consultation_id}, user={user_id}, specialty={specialty}, doctor_id={resolved_doctor_id}")

        name_part = f"Dr. {doctor_name}" if doctor_name else f"a {specialty} specialist"
        return {
            "status":          "booked",
            "consultation_id": consultation_id,
            "specialty":       specialty,
            "doctor_id":       resolved_doctor_id,
            "message": (
                f"Your consultation with {name_part} has been booked! "
                "Please pay the consultation fee in the Consultations tab to confirm. "
                "You can track it there."
            ),
        }

    except Exception as e:
        logger.error(f"book_consultation error: {e}", exc_info=True)
        return {
            "status":  "error",
            "message": f"Could not book the consultation: {e}. Please try again.",
        }


def get_available_doctors(specialty: str = "") -> dict:
    """Fetch available doctors from the database, optionally filtered by specialty.

    Call this when:
    - Patient asks to see a doctor or book an appointment
    - You want to show who is currently available before booking

    Args:
        specialty: Optional specialty filter (e.g. "Cardiology", "General Physician").
                   Leave empty to get all available doctors.

    Returns a dict with a list of available doctors (name, specialty, city, fee, rating).
    """
    try:
        doctors = _run_async(_db.get_all_doctors())
        available = [d for d in doctors if d.get("available_status") == "available"]
        if specialty:
            spec_lower = specialty.lower()
            filtered = [d for d in available if spec_lower in (d.get("specialization") or "").lower()]
            if filtered:
                available = filtered
        # Return trimmed list (max 5 for readability)
        result = [
            {
                "doctor_id":      d.get("id", ""),
                "name":           _name(d.get("first_name"), d.get("last_name")) or d.get("full_name", "") or "Doctor",
                "specialization": d.get("specialization", "General Physician"),
                "city":           d.get("city", ""),
                "fee":            d.get("consultation_fee", 0),
                "rating":         d.get("rating", 0),
                "available":      True,
            }
            for d in available[:5]
        ]
        return {"status": "success", "doctors": result, "count": len(result)}
    except Exception as e:
        logger.error(f"get_available_doctors error: {e}")
        return {"status": "error", "doctors": [], "count": 0}


# ──────────────────────────────────────────────────────────────────
# Database tools
# ──────────────────────────────────────────────────────────────────

def save_patient_to_db(name: str, age: int, phone: str = "", gender: str = "unknown") -> dict:
    """Save a patient to the database and return patient_id."""
    try:
        from models import PatientProfile
        import threading as _threading
        patient_id = f"patient_{uuid.uuid4().hex[:10]}"
        now = datetime.now()
        patient = PatientProfile(
            id=patient_id, name=name, age=age, gender=gender, phone=phone,
            address="", medical_history=[], allergies=[], current_medications=[],
            emergency_contact="", created_at=now, updated_at=now,
        )
        # Fire-and-forget — patient_id is already generated; no need to block on the write
        _threading.Thread(target=lambda: _run_async(_db.save_patient(patient)), daemon=True).start()
        return {"status": "success", "patient_id": patient_id, "name": name}
    except Exception as e:
        logger.error(f"save_patient_to_db failed: {e}")
        return {"status": "error", "error": str(e)}


def _send_external_notification(doctors: list, approval_id: str, patient_name: str, symptoms: str, risk_level: str, specialty: str):
    """Send SMS alerts to doctors for URGENT/EMERGENCY cases.
    Integrates with MSG91 when configured.
    Falls back to logging when credentials are not set."""
    import os
    import httpx
    msg91_auth_key = os.getenv("MSG91_AUTH_KEY", "")
    msg91_sender_id = os.getenv("MSG91_SENDER_ID", "MEDVRA")
    msg91_alert_template_id = os.getenv("MSG91_ALERT_TEMPLATE_ID", "")

    emoji = "🚨" if risk_level == "EMERGENCY" else "⚠️"
    msg_body = (
        f"{emoji} Medivora {risk_level} Alert\n"
        f"Patient: {patient_name or 'Anonymous'}\n"
        f"Symptoms: {symptoms[:100]}\n"
        f"Specialty: {specialty}\n"
        f"Approval ID: {approval_id}\n"
        f"Please review on your Medivora Doctor Dashboard immediately."
    )

    if msg91_auth_key and msg91_alert_template_id:
        for doc in doctors[:3]:  # Notify up to 3 doctors
            phone = doc.get("phone", "")
            if not phone:
                continue
            try:
                payload = {
                    "flow_id": msg91_alert_template_id,
                    "sender": msg91_sender_id,
                    "mobiles": phone,
                    "RISK": risk_level,
                    "PATIENT": patient_name or "Anonymous",
                    "SYMPTOMS": symptoms[:80],
                    "SPECIALTY": specialty,
                    "APPROVAL": approval_id,
                }
                resp = httpx.post(
                    "https://control.msg91.com/api/v5/flow/",
                    json=payload,
                    headers={"authkey": msg91_auth_key, "Content-Type": "application/json"},
                    timeout=10,
                )
                if resp.status_code == 200:
                    logger.info(f"SMS alert sent to doctor {doc.get('id')} for {approval_id}")
                else:
                    logger.warning(f"MSG91 SMS failed for doctor {doc.get('id')}: {resp.status_code} {resp.text}")
            except Exception as e:
                logger.warning(f"SMS notification failed for {doc.get('id')}: {e}")
    else:
        logger.info(f"External notification skipped (MSG91 not configured). {risk_level} alert for {approval_id}: {msg_body[:80]}")


def create_approval_and_notify(
    patient_name: str,
    symptoms: str,
    diagnosis: str,
    risk_level: str,
    prescription_text: str,
    specialty: str = "",
    user_id: str = "",
) -> dict:
    """Create a doctor approval request and notify all available doctors.
    Returns the approval_id and determined specialty."""
    try:
        # Use module-level user context if not explicitly provided
        if not user_id:
            user_id = _current_user_id
        # Use real user name from context if agent passed "Anonymous"
        if not patient_name or patient_name.lower() in ("anonymous", "anonymous user", ""):
            patient_name = _current_user_name or patient_name or "Patient"

        approval_id = f"apr_{uuid.uuid4().hex[:10]}"
        priority_map = {"EMERGENCY": 1, "URGENT": 2, "ROUTINE": 3, "HOME_CARE": 4}
        priority = priority_map.get(risk_level, 3)

        # Determine specialty from symptoms/diagnosis if not provided
        if not specialty or specialty == "general_medicine":
            spec_result = determine_specialty(symptoms, diagnosis)
            determined_specialty = spec_result["specialty"]
        else:
            determined_specialty = specialty

        # Track determined specialty so api.py can include it in additional_data
        if user_id:
            _latest_approval_specialty[user_id] = determined_specialty
            # Write final triage object (overwrites early estimate from assess_risk)
            category = (diagnosis or symptoms or "")[:60].strip()
            _write_triage_from_risk(risk_level, specialty=determined_specialty, category=category)

        # ── Pregnancy detection ──
        pregnancy_keywords = ['pregnan', 'garbh', 'garbh', 'conceiv', 'expecting', 'trimester', 'weeks pregnant', 'month pregnant']
        combined_text = (symptoms + " " + diagnosis).lower()
        is_pregnant = any(kw in combined_text for kw in pregnancy_keywords)

        safety_warnings = []

        # ── Pregnancy: Force risk escalation for dangerous symptoms ──
        if is_pregnant:
            logger.info("Pregnancy detected — applying pregnancy safety protocols")

            # Force URGENT/EMERGENCY for dangerous pregnancy symptoms
            danger_keywords_emergency = ['bleeding', 'bleed', 'blood', 'khoon', 'seizure', 'fits', 'unconscious']
            danger_keywords_urgent = ['contraction', 'labour', 'labor', 'leaking fluid', 'water break',
                                       'pani toot', 'reduced movement', 'baby not moving', 'swelling',
                                       'high bp', 'headache']

            if any(kw in combined_text for kw in danger_keywords_emergency):
                if risk_level not in ("EMERGENCY",):
                    logger.warning(f"SAFETY OVERRIDE: Escalating risk from {risk_level} to EMERGENCY (pregnancy + danger symptoms)")
                    risk_level = "EMERGENCY"
                    priority = 1
                    safety_warnings.append("Risk escalated to EMERGENCY: dangerous symptoms detected during pregnancy")
            elif any(kw in combined_text for kw in danger_keywords_urgent):
                if risk_level in ("ROUTINE", "HOME_CARE"):
                    logger.warning(f"SAFETY OVERRIDE: Escalating risk from {risk_level} to URGENT (pregnancy + concerning symptoms)")
                    risk_level = "URGENT"
                    priority = 2
                    safety_warnings.append("Risk escalated to URGENT: concerning symptoms detected during pregnancy")

        # ── Phase B: Drug blacklist filter ──
        blacklist = load_blacklist_from_db(_db)
        filtered_prescription, removed_drugs = filter_prescription(prescription_text, blacklist, is_pregnant=is_pregnant)

        # ── Log safety violations for each removed drug ──
        if removed_drugs:
            for drug in removed_drugs:
                try:
                    _run_async(_db.log_safety_violation(
                        approval_id=approval_id,
                        violation_type="blacklist_filter",
                        drug_name=drug,
                        category="pregnancy_contraindicated" if is_pregnant else "schedule_x_or_narcotic",
                        patient_context=f"symptoms: {symptoms[:100]}",
                        is_pregnant=is_pregnant,
                        action_taken="removed",
                        details=f"Drug '{drug}' removed by Medivora Safety Filter from AI-generated prescription",
                    ))
                except Exception:
                    pass
            logger.warning(f"SAFETY VIOLATIONS LOGGED: {len(removed_drugs)} drugs removed — {removed_drugs}")

        # ── Pregnancy: Hard NSAID rejection check (defense in depth) ──
        if is_pregnant:
            nsaid_names = ['diclofenac', 'ibuprofen', 'aspirin', 'naproxen', 'mefenamic acid',
                           'piroxicam', 'indomethacin', 'celecoxib', 'etoricoxib', 'mefenamic']
            prescription_lower = filtered_prescription.lower() if filtered_prescription else ""
            found_nsaids = [n for n in nsaid_names if n in prescription_lower]
            if found_nsaids:
                # Strip the NSAID lines that somehow survived the blacklist filter
                for nsaid in found_nsaids:
                    lines = filtered_prescription.split("\n")
                    filtered_prescription = "\n".join(
                        line for line in lines if nsaid not in line.lower()
                    )
                    if nsaid not in removed_drugs:
                        removed_drugs.append(nsaid)
                    # Log hard NSAID rejection as a critical safety violation
                    try:
                        _run_async(_db.log_safety_violation(
                            approval_id=approval_id,
                            violation_type="hard_nsaid_rejection",
                            drug_name=nsaid,
                            category="pregnancy_contraindicated",
                            patient_context=f"symptoms: {symptoms[:100]}",
                            is_pregnant=True,
                            action_taken="force_removed",
                            details=f"CRITICAL: NSAID '{nsaid}' survived blacklist filter and was force-removed (defense-in-depth)",
                        ))
                    except Exception:
                        pass
                safety_warnings.append(
                    f"Safety Violation Caught: NSAIDs ({', '.join(found_nsaids)}) are contraindicated in pregnancy. "
                    f"Removed from prescription. Use Paracetamol or non-pharmacological relief instead."
                )
                logger.warning(f"HARD NSAID REJECTION: {found_nsaids} stripped from pregnant patient prescription")

        # ── Phase B: Structured clinical note ──
        clinical_note = {
            "hpi": symptoms,
            "vitals": "Placeholder — to be filled by examining doctor",
            "differential_diagnosis": [diagnosis] if diagnosis else [],
            "ai_confidence": 0.8,
            "generated_by": "Sensyva AI",
            "generated_at": datetime.now().isoformat(),
            "record_type": "PROVISIONAL",
        }

        patient_id = f"anonymous_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        import json as _json
        approval_request = {
            "approval_id": approval_id,
            "patient_info": {"id": patient_id},
            "user_id": user_id,
            "ai_assessment": {
                "symptoms": symptoms,
                "specialty": determined_specialty,
                "risk_level": risk_level,
                "diagnosis": diagnosis,
                "recommendations": [],
                "prescription": filtered_prescription,
                "confidence_score": 0.8,
                "blacklisted_removed": removed_drugs,
                "safety_warnings": safety_warnings,
                "is_pregnant": is_pregnant,
            },
            "proposed_prescription": {
                "prescription_text": filtered_prescription,
                "follow_up": "Doctor consultation recommended",
            },
            "clinical_note": _json.dumps(clinical_note),
            "status": "pending_approval",
            "priority": priority,
            "created_at": datetime.now().isoformat(),
        }
        # Fetch doctors synchronously — we need the count for the return value
        # and must fire notifications before this function returns.
        doctors = _run_async(_db.get_available_doctors(determined_specialty))
        notif_msg = f"Naya prescription approval ({determined_specialty}): {patient_name or 'Anonymous'}, Symptoms: {symptoms[:80]}, Risk: {risk_level}"

        import threading as _threading
        from models import PatientProfile as _PP

        def _bg_writes():
            """Fire-and-forget: save approval + patient + notifications in background."""
            try:
                _run_async(_db.save_approval_request(approval_request))
            except Exception:
                pass
            try:
                now2 = datetime.now()
                anon = _PP(
                    id=patient_id, name=patient_name or "Anonymous User", age=25,
                    gender="unknown", phone="", address="", medical_history=[],
                    allergies=[], current_medications=[], emergency_contact="",
                    created_at=now2, updated_at=now2,
                )
                _run_async(_db.save_patient(anon))
            except Exception:
                pass
            if doctors:
                try:
                    _run_async(_db.assign_doctor_to_approval(approval_id, doctors[0]["id"]))
                except Exception:
                    pass
                for doc in doctors:
                    try:
                        _run_async(_db.save_notification(doc["id"], approval_id, notif_msg, priority))
                    except Exception:
                        pass
                if risk_level in ("EMERGENCY", "URGENT"):
                    _send_external_notification(doctors, approval_id, patient_name, symptoms, risk_level, determined_specialty)

        _threading.Thread(target=_bg_writes, daemon=True).start()

        result = {
            "status": "success",
            "approval_id": approval_id,
            "specialty": determined_specialty,
            "risk_level": risk_level,
            "doctors_notified": len(doctors) if doctors else 0,
        }
        if removed_drugs:
            result["removed_drugs"] = removed_drugs
            result["removed_drugs_notice"] = f"{len(removed_drugs)} restricted drug(s) removed by Medivora Safety Filter"
        if safety_warnings:
            result["safety_warnings"] = safety_warnings
        if is_pregnant:
            result["pregnancy_detected"] = True

        return result
    except Exception as e:
        logger.error(f"create_approval_and_notify failed: {e}")
        return {"status": "error", "error": str(e)}
