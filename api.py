from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, List
import uvicorn
import uuid
import asyncio
import httpx
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
import logging
from logging.handlers import RotatingFileHandler
import os
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from diskcache import Cache
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

load_dotenv()

from config import Config, Settings
from database import DatabaseManager, _name
from agents.voice_processing_agent import VoiceProcessingAgent
from auth import (
    create_token, hash_password, validate_password_strength,
    get_current_user, get_current_user_optional,
    require_doctor, require_admin, require_patient,
)

# ADK imports
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types
from medivora_agent import root_agent, voice_agent
from medivora_agent.tools import set_current_user_id, set_current_user
from pdf_generator import generate_prescription_pdf, compute_signature_hash
from drug_blacklist import seed_default_blacklist

# Memory-driven healthcare modules (Phase 1–5)
from services.memory import MemoryService
from services.emotional import EmotionalContextBuilder
from services.triage import TriageEngine
from services.safety import SafetyValidator, log_safety_events
from services.context_builder import ContextBuilder
from services.vision import get_vision_service

_triage_engine    = TriageEngine()
_emotional_builder = EmotionalContextBuilder()
_safety_validator  = SafetyValidator()
_context_builder   = ContextBuilder()

settings = Settings()

# Disk cache for reducing redundant DB queries
_api_cache = Cache(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "api_cache"))

# Configure logging with Rich for beautiful console output
def setup_logging():
    from rich.logging import RichHandler

    log_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # File handler (plain text for log files)
    log_file = "logs/app.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = RotatingFileHandler(log_file, maxBytes=1024 * 1024 * 5, backupCount=5)
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)

    # Rich console handler (colorized, structured output)
    rich_handler = RichHandler(
        rich_tracebacks=True,
        markup=True,
        show_time=True,
        show_path=False,
    )
    rich_handler.setLevel(logging.INFO)

    # Configure logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(rich_handler)

    return logger

logger = setup_logging()

# ── Logfire: structured observability ────────────────────────────
try:
    import logfire
    _logfire_token = os.getenv("LOGFIRE_TOKEN", "")
    if _logfire_token:
        logfire.configure(token=_logfire_token)
        logger.info("Logfire observability enabled")
    else:
        logfire.configure(send_to_logfire=False)   # local-only mode in dev
        logger.info("Logfire running in local mode (no LOGFIRE_TOKEN set)")
except Exception as _lf_err:
    logger.warning(f"Logfire init skipped: {_lf_err}")

# ── Tiktoken: token counter utility ──────────────────────────────
def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Estimate token count for a string (uses cl100k_base encoding)."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4   # fallback: ~4 chars per token

# ADK Session & Runner (replaces old SessionManager)
APP_NAME = "medivora"
USER_ID = "patient"

adk_session_service = InMemorySessionService()
adk_runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=adk_session_service,
)

# Separate session service + runner for voice — uses a lightweight agent
# with thinking disabled to keep latency low.
_voice_session_service = InMemorySessionService()
_voice_runner = Runner(
    agent=voice_agent,
    app_name=APP_NAME + "_voice",
    session_service=_voice_session_service,
)
_voice_active_sessions: set = set()

# Track active session IDs for the /health and /sessions endpoints
_active_sessions: set = set()

# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Medical AI Assistant API (ADK)")
    yield
    logger.info("Shutting down Medical AI Assistant API")

# FastAPI app
app = FastAPI(
    title="Medical AI Assistant API",
    version="2.0.0",
    description="Advanced Medical AI Assistant with session management",
    lifespan=lifespan
)

# CORS - read allowed origins from env, default to local dev
allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:5174,http://localhost:5175,http://localhost:5176,http://localhost:5177,http://localhost:3000,http://localhost:4173")
allowed_origins = [origin.strip() for origin in allowed_origins_str.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── Daily video ──────────────────────────────────────────────────────────────
DAILY_API_KEY  = os.getenv("DAILY_API_KEY", "")
DAILY_BASE_URL = "https://api.daily.co/v1"

def _daily_headers() -> dict:
    return {"Authorization": f"Bearer {DAILY_API_KEY}", "Content-Type": "application/json"}

async def create_daily_room(name: str) -> tuple[str, str]:
    """Create a Daily room and return (room_name, room_url)."""
    import httpx, time, re
    safe_name = re.sub(r"[^a-z0-9-]", "-", name.lower())[:60].strip("-")
    async with httpx.AsyncClient() as client:
        res = await client.post(
            f"{DAILY_BASE_URL}/rooms",
            headers=_daily_headers(),
            json={"name": safe_name, "privacy": "private", "properties": {}},
            timeout=10,
        )
        res.raise_for_status()
        data = res.json()
        return data["name"], data["url"]

async def create_daily_token(room_name: str, is_owner: bool, user_name: str) -> str:
    """Create a Daily meeting token and return it."""
    import httpx, time
    async with httpx.AsyncClient() as client:
        res = await client.post(
            f"{DAILY_BASE_URL}/meeting-tokens",
            headers=_daily_headers(),
            json={"properties": {
                "room_name": room_name,
                "is_owner": is_owner,
                "user_name": user_name,
                "exp": int(time.time()) + 86400,
            }},
            timeout=10,
        )
        res.raise_for_status()
        return res.json()["token"]
# ─────────────────────────────────────────────────────────────────────────────

# Pydantic models
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=settings.max_message_length)
    session_id: Optional[str] = Field(None, pattern=r'^[a-fA-F0-9-]{36}$')
    
    @field_validator('message')
    @classmethod
    def validate_message(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('Message cannot be empty')
        return v.strip()

class ChatResponse(BaseModel):
    response: str
    status: str
    session_id: str
    session_state: str
    additional_data: Optional[Dict] = None

class PatientResponse(BaseModel):
    patient_id: str
    name: str
    phone: str
    age: int
    gender: str
    address: str
    medical_history: Optional[str] = None
    current_medications: Optional[str] = None
    allergies: Optional[str] = None

class SessionInfo(BaseModel):
    session_id: str
    state: str
    created_at: datetime
    last_activity: datetime

# File upload validation
async def validate_upload_file(file: UploadFile) -> UploadFile:
    """Validate uploaded file"""
    if not file:
        return file
    
    try:
        # Check file size
        if file.size and file.size > settings.max_file_size_mb * 1024 * 1024:
            logger.warning(f"File too large: {file.size} bytes")
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB"
            )
        
        # Check file type - more lenient for audio files
        allowed_types = [
            "image/jpeg", "image/png", "image/gif",
            "audio/mpeg", "audio/wav", "audio/ogg", "audio/webm", "audio/mp4",
            "application/pdf", "text/plain", "application/octet-stream"
        ]
        
        # Allow audio files even if content_type is not set
        if file.content_type and file.content_type not in allowed_types:
            # Special handling for audio files
            if file.filename and any(ext in file.filename.lower() for ext in ['.webm', '.wav', '.mp3', '.ogg']):
                logger.info(f"Allowing audio file despite content_type: {file.content_type}")
            else:
                logger.warning(f"File type not allowed: {file.content_type}")
                raise HTTPException(
                    status_code=400,
                    detail=f"File type not allowed: {file.content_type}"
                )
        
        return file
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error validating file: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"File validation error: {str(e)}"
        )

# Retry-wrapped ADK runner for transient Gemini API failures
_HINGLISH_WORDS = {
    "hai", "hain", "hua", "hue", "hogi", "hoga", "ho", "kar", "karo", "karna",
    "mujhe", "meri", "mera", "mere", "mein", "main", "mai", "aur", "ya", "nahi",
    "nhi", "kya", "kuch", "koi", "kal", "aaj", "abhi", "bahut", "thoda", "zyada",
    "accha", "theek", "theek", "bhi", "toh", "to", "se", "pe", "par", "ko",
    "ka", "ki", "ke", "ek", "do", "din", "raat", "sar", "sir", "dard", "pet",
    "bukhar", "khana", "pani", "dawai", "doctor", "matlab", "samajh", "bata",
    "batao", "lagta", "laga", "tha", "thi", "the", "rahega", "raha", "rahi",
    "suniye", "suno", "please", "ji", "haan", "haa", "na", "woh", "wo", "yeh",
    "ye", "unhe", "inhe", "apna", "apni", "apne", "jab", "tab", "kyun", "kaise",
}

def _detect_language_directive(text: str) -> str:
    """Return an explicit language directive to prepend to the user message."""
    if not text:
        return ""
    # Devanagari Unicode block: U+0900–U+097F
    if any('ऀ' <= ch <= 'ॿ' for ch in text):
        return "[LANGUAGE DIRECTIVE: The patient is writing in Devanagari Hindi. Reply ONLY in Devanagari Hindi. Medical terms may stay in English.]"
    words = set(text.lower().split())
    if words & _HINGLISH_WORDS:
        return "[LANGUAGE DIRECTIVE: The patient is writing in Hinglish. Reply in Hinglish — mix Hindi words (Roman script) with English naturally. Do NOT reply in pure English or pure Devanagari.]"
    return "[LANGUAGE DIRECTIVE: The patient is writing in English. Reply in ENGLISH ONLY. Do not use any Hindi or Hinglish words.]"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, asyncio.TimeoutError)),
    reraise=True,
)
async def _run_adk_agent(session_id: str, user_id: str, content, max_retries: int = 2):
    """Run ADK agent with automatic retry on transient failures."""
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            final_response = ""
            async for event in adk_runner.run_async(
                session_id=session_id,
                user_id=user_id,
                new_message=content,
            ):
                if event.is_final_response() and event.content and event.content.parts:
                    # Safely find the first text part — parts can also be
                    # FunctionCall / FunctionResponse objects (no .text attr)
                    for part in event.content.parts:
                        text = getattr(part, "text", None)
                        if text:
                            final_response = text
                            break
            if final_response:
                return final_response
            logger.warning(f"ADK agent returned empty response (attempt {attempt + 1}/{max_retries + 1})")
        except Exception as e:
            last_error = e
            logger.error(f"ADK agent error (attempt {attempt + 1}/{max_retries + 1}): {e}")
            if attempt < max_retries:
                import asyncio as _asyncio
                await _asyncio.sleep(1)
    if last_error:
        logger.error(f"ADK agent failed after {max_retries + 1} attempts: {last_error}")
    return ""


# Routes
@app.get("/")
async def read_root():
    """Root endpoint"""
    return {
        "message": "Welcome to the Medical AI Assistant API",
        "version": "2.0.0",
        "status": "healthy"
    }


# ── Emotional Intelligence Layer ─────────────────────────────────

def detect_emotional_context(message: str) -> str:
    distress_keywords = [
        "scared", "worried", "embarrassed", "ashamed", "don't know",
        "confused", "alone", "nobody", "hopeless", "anxious", "anxiety",
        "depressed", "depression", "crying", "pain", "afraid", "fear",
        "helpless", "lost", "desperate", "nervous", "stress", "stressed",
        "overwhelmed", "darr", "dara", "pareshan", "akela", "akeli",
        "rone", "rona", "tadap", "mushkil"
    ]
    sensitive_topics = [
        "period", "periods", "menstrual", "pregnancy", "pregnant",
        "miscarriage", "abortion", "fertility", "infertility", "ivf",
        "sexual", "abuse", "rape", "assault", "suicide", "self harm",
        "cutting", "overdose", "menses", "masik", "garbh", "baccha"
    ]
    msg_lower = message.lower()
    if any(word in msg_lower for word in distress_keywords):
        return "high_emotional"
    if any(topic in msg_lower for topic in sensitive_topics):
        return "sensitive_topic"
    return "standard"


_GREETING_WORDS = {"hi", "hey", "hello", "hii", "hiii", "yo", "sup", "hola", "namaste", "ok", "okay", "thanks", "thank you", "good morning", "good evening", "good afternoon", "gm", "morning"}
_PLACEHOLDER_TITLE = "New Consultation"

def _is_greeting(text: str) -> bool:
    """Check if a message is just a greeting with no medical content."""
    cleaned = text.strip().lower().rstrip("!.,?")
    return cleaned in _GREETING_WORDS or len(cleaned) <= 3


async def _bg_persist_voice_turn(
    session_id: str, uid: str, new_session: bool,
    msg: str, resp: str, raw_resp: str,
    safety_res, pid, mem_svc, session_summary: str,
):
    """Background task: all DB/memory work after a voice turn. Non-blocking for the audio response."""
    db = DatabaseManager()
    if new_session:
        title = await _generate_session_title(msg, resp)
        await db.create_chat_session(session_id, uid, title)
    else:
        await _maybe_update_session_title(db, session_id, msg, resp)
    await db.save_chat_message(session_id, uid, "ai", resp)
    if safety_res.events and pid:
        await log_safety_events(db, safety_res, session_id, pid, raw_resp)
    if mem_svc and pid:
        await mem_svc.extract_and_store_facts(pid, session_id, f"Patient: {msg}\n\nAI: {resp}")
        if "📋" in resp and "assessment" in resp.lower():
            outcome = (
                "emergency" if safety_res.has_emergency else
                ("consultation_booked" if "book an appointment" in resp.lower() else "prescription_pending")
            )
            await mem_svc.save_session_summary(pid, session_id, session_summary or resp, outcome)
    await db.touch_chat_session(session_id)


async def _bg_persist_chat_turn(
    session_id: str, uid: str, new_session: bool,
    msg: str, resp: str, raw_resp: str,
    safety_res, pid, mem_svc, session_summary: str,
):
    """Background task: all DB + memory work after a text chat turn. Non-blocking for the response."""
    db = DatabaseManager()
    if new_session:
        title = await _generate_session_title(msg, resp)
        await db.create_chat_session(session_id, uid, title)
    else:
        await _maybe_update_session_title(db, session_id, msg, resp)
    await db.save_chat_message(session_id, uid, "ai", resp)
    if safety_res.events and pid:
        await log_safety_events(db, safety_res, session_id, pid, raw_resp)
    if mem_svc and pid:
        await mem_svc.extract_and_store_facts(pid, session_id, f"Patient: {msg}\n\nAI: {resp}")
        if "📋" in resp and "assessment" in resp.lower():
            outcome = (
                "emergency" if safety_res.has_emergency else
                ("consultation_booked" if "book an appointment" in resp.lower() else "prescription_pending")
            )
            await mem_svc.save_session_summary(pid, session_id, session_summary or resp, outcome)
    await db.touch_chat_session(session_id)


async def _generate_session_title(message: str, ai_response: str = "") -> str:
    """Use Gemini to generate a short medical condition noun/phrase for the chat title.
    Uses the patient message + optionally the AI response for better context."""
    # If the message is just a greeting, we can't extract a condition
    if _is_greeting(message) and not ai_response:
        return _PLACEHOLDER_TITLE

    try:
        from google import genai
        from google.genai import types as _gt
        client = genai.Client()

        context = f'Patient: "{message[:300]}"'
        if ai_response:
            context += f'\nAI response: "{ai_response[:300]}"'

        prompt = (
            f'From this medical chat, extract a short condition/topic noun (2-4 words max). '
            f'Return ONLY the noun phrase, nothing else. '
            f'If the conversation is just greetings with no medical topic yet, return exactly "New Consultation".\n'
            f'Examples of good titles: "Headache", "Lower Back Pain", "Anxiety & Stress", "Skin Rash", "Fever & Cold", "Chest Pain", "Pregnancy Query"\n\n'
            f'{context}'
        )

        def _call():
            return client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=_gt.GenerateContentConfig(
                    thinking_config=_gt.ThinkingConfig(thinking_budget=0),
                    max_output_tokens=20,
                ),
            )

        response = await asyncio.to_thread(_call)
        title = response.text.strip().strip('"').strip("'").strip()
        if title and len(title) <= 60:
            return title
    except Exception as e:
        logger.warning(f"Title generation failed, using fallback: {e}")

    if _is_greeting(message):
        return _PLACEHOLDER_TITLE
    return message[:50] + ("..." if len(message) > 50 else "")


async def _maybe_update_session_title(db, session_id: str, message: str, ai_response: str):
    """If session still has a placeholder title, generate a real one from the latest exchange."""
    try:
        # Skip if message is still just a greeting
        if _is_greeting(message):
            return
        session = await db.get_chat_session(session_id)
        if not session:
            return
        current_title = session.get("title", "")
        # Update if title is a placeholder, a short greeting, or the raw "hi"/"hello" text
        needs_update = (
            not current_title
            or current_title == _PLACEHOLDER_TITLE
            or current_title.lower().strip("!., ") in _GREETING_WORDS
            or len(current_title.strip()) <= 3
        )
        if needs_update:
            new_title = await _generate_session_title(message, ai_response)
            if new_title and new_title != _PLACEHOLDER_TITLE:
                await db.update_chat_session_title(session_id, new_title)
                logger.info(f"Updated session {session_id} title: '{current_title}' -> '{new_title}'")
    except Exception as e:
        logger.warning(f"_maybe_update_session_title failed: {e}")


def _strip_internal_notes(text: str) -> str:
    """Remove internal model meta-blocks that leak into patient-facing responses.

    The language directive and system-note blocks are injected into the user
    message so the model knows which language to use. Occasionally the model
    echoes them back verbatim in its output. They must never appear in the
    response the patient sees.
    """
    import re as _re
    # Remove [SYSTEM NOTE: ...] and [LANGUAGE DIRECTIVE: ...] blocks (single or multi-line)
    text = _re.sub(r'\[SYSTEM NOTE:[^\]]*\]\s*', '', text)
    text = _re.sub(r'\[LANGUAGE DIRECTIVE:[^\]]*\]\s*', '', text)
    text = _re.sub(r'\[PATIENT CONTEXT\].*?\[END PATIENT CONTEXT\]\s*', '', text, flags=_re.DOTALL)
    return text.strip()


def soften_response(text: str) -> str:
    text = _strip_internal_notes(text)
    replacements = {
        "You should": "You might consider",
        "You must": "It could really help to",
        "This is abnormal": "This is something worth paying attention to",
        "See a doctor": "Your care team would be great to loop in here",
        "Go to hospital": "Heading to a clinic or hospital would be a good idea",
        "This is serious": "This deserves proper attention",
        "You need to": "It would really help to",
        "immediately": "as soon as you can",
        "You have to": "It would be worth",
    }
    for hard, soft in replacements.items():
        text = text.replace(hard, soft)
    return text


def get_message_count(session_id: str) -> int:
    if not hasattr(get_message_count, "_counts"):
        get_message_count._counts = {}
    count = get_message_count._counts.get(session_id, 0) + 1
    get_message_count._counts[session_id] = count
    return count


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("30/minute")
async def chat_endpoint(
    request: Request,
    background_tasks: BackgroundTasks,
    message: str = Form(...),
    session_id: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None),
    current_user: Optional[Dict] = Depends(get_current_user_optional),
):
    """Main chat endpoint for medical consultation (ADK-powered)"""
    try:
        # Derive user_id from JWT token (not from client form data)
        user_id = current_user["sub"] if current_user else None

        # Validate request
        chat_request = ChatRequest(message=message, session_id=session_id)
        logger.info(f"Chat request: session_id='{session_id}', authenticated={user_id is not None}, message_length={len(message)}")

        # Validate uploaded files
        if audio:
            try:
                audio = await validate_upload_file(audio)
                logger.info(f"Audio file received: {audio.filename}")
            except HTTPException as e:
                logger.warning(f"Audio file validation failed: {e.detail}")
                audio = None

        if file:
            try:
                file = await validate_upload_file(file)
                logger.info(f"File received: {file.filename}")
            except HTTPException as e:
                logger.warning(f"File validation failed: {e.detail}")
                file = None

        # Get or create ADK session
        current_session_id = session_id or str(uuid.uuid4())
        is_new = current_session_id not in _active_sessions

        # Set user context for agent tools (id + name + email from JWT)
        set_current_user(
            user_id or "",
            name             = (current_user or {}).get("name", ""),
            email            = (current_user or {}).get("email", ""),
            is_authenticated = current_user is not None,
        )

        # ── Phase 3: Triage pre-assessment (deterministic, before Gemini) ──
        db_for_memory = DatabaseManager() if user_id else None
        memory_svc    = MemoryService(db_for_memory) if db_for_memory else None

        # P3: Parallelize ADK session creation + patient memory fetch
        async def _create_adk_session():
            if is_new:
                await adk_session_service.create_session(
                    app_name=APP_NAME, user_id=USER_ID, session_id=current_session_id,
                )
                _active_sessions.add(current_session_id)

        async def _fetch_patient_memory():
            if not (user_id and memory_svc):
                return None, {"facts": {}, "recent_summaries": []}
            # P2: Cache patient_id lookup (immutable — profile → patient mapping never changes)
            pid = _patient_id_cache.get(user_id)
            if not pid:
                pid = await memory_svc.get_patient_id(user_id)
                if pid:
                    _patient_id_cache[user_id] = pid
            if not pid:
                return None, {"facts": {}, "recent_summaries": []}
            # P2: Cache patient context per session — skip embedding on follow-up turns
            ctx_key = f"{pid}:{current_session_id}"
            if not is_new and ctx_key in _patient_context_cache:
                return pid, _patient_context_cache[ctx_key]
            ctx = await memory_svc.get_patient_context(pid, query_text=chat_request.message)
            _patient_context_cache[ctx_key] = ctx
            return pid, ctx

        _, (patient_id, patient_context) = await asyncio.gather(
            _create_adk_session(),
            _fetch_patient_memory(),
        )

        triage_result = _triage_engine.score(chat_request.message, patient_context)

        # Hard emergency: skip Gemini entirely — return instant pre-built response
        if triage_result.is_hard_emergency:
            logger.warning(f"HARD EMERGENCY detected: session={current_session_id}, score={triage_result.risk_score}")
            emergency_resp = triage_result.emergency_response
            if user_id and db_for_memory:
                background_tasks.add_task(
                    db_for_memory.save_chat_message,
                    current_session_id, user_id, "ai", emergency_resp,
                )
            return ChatResponse(
                response=emergency_resp,
                status="emergency",
                session_id=current_session_id,
                session_state="active",
                additional_data={
                    "adk_powered": False,
                    "is_authenticated": current_user is not None,
                    "is_medical_report": False,
                    "is_book_appointment": False,
                    "triage": triage_result.to_dict(),
                },
            )

        # ── Phase 2: Emotional context ────────────────────────────────────
        emotional_ctx = _emotional_builder.build(
            current_message=chat_request.message,
            memory_facts=patient_context.get("facts", {}),
            is_first_message=is_new,
        )

        # ── Phase 1: Memory context injection (first message of session) ──
        memory_note  = ""
        # Prefer name from patient_memory facts over JWT (JWT often has no name)
        _name_facts  = patient_context.get("facts", {}).get("preference", [])
        patient_name = next((f["value"] for f in _name_facts if f["key"] == "name"), "")
        if not patient_name:
            patient_name = (current_user or {}).get("name", "") if user_id else ""
        if is_new and patient_id and memory_svc:
            memory_note = memory_svc.format_context_for_prompt(patient_context)

        # ── Phase 6: Build enriched prompt ────────────────────────────────
        # Triage note is NOT injected for URGENT/MODERATE — Gemini assesses
        # those naturally. Hard emergencies are already bypassed above.
        # Only inject for specialty routing hint (memory context only).
        if is_new and memory_note:
            adapted_message = _context_builder.build_first_message(
                message=chat_request.message,
                memory_note=memory_note,
                emotional_note=emotional_ctx["system_note"],
                triage_note="",
                patient_name=patient_name if memory_note else None,
            )
        else:
            adapted_message = _context_builder.build(
                message=chat_request.message,
                emotional_note=emotional_ctx["system_note"],
                triage_note="",
            )

        msg_count = get_message_count(current_session_id)

        # ── Vision Layer: inject pending image context if present ──────────
        # A previous /chat/upload-image call may have stored a [IMAGE CONTEXT]
        # block for this session.  Consume it exactly once here so the AI
        # sees the visual findings on the patient's very next message.
        _image_ctx = _pending_image_contexts.pop(current_session_id, None)
        if _image_ctx:
            adapted_message = f"{_image_ctx}\n\n{adapted_message}"
            logger.info(f"Image context injected into chat turn: session={current_session_id}")

        # Prepend per-message language directive so Gemini reliably mirrors the patient
        lang_directive = _detect_language_directive(chat_request.message)
        if lang_directive:
            adapted_message = f"{lang_directive}\n\n{adapted_message}"

        # Build ADK user message
        content = genai_types.Content(
            role="user",
            parts=[genai_types.Part.from_text(text=adapted_message)],
        )

        # Run agent (with automatic retry on transient failures)
        final_response = await _run_adk_agent(current_session_id, USER_ID, content)

        if not final_response:
            final_response = "I'm sorry, I couldn't process your message. Please try again."

        # If the assessment pipeline ran this turn, its summary_result is stored in
        # session state but never returned to the patient (the root agent's booking
        # message overwrites it).  Prepend the 📋 report so the patient sees it.
        _session_summary = ""
        try:
            _sess = await adk_session_service.get_session(
                app_name=APP_NAME, user_id=USER_ID, session_id=current_session_id
            )
            _session_summary = (_sess.state or {}).get("summary_result", "")
            if _session_summary and "📋" in _session_summary and _session_summary.strip() not in final_response:
                final_response = _session_summary.strip() + "\n\n---\n\n" + final_response
        except Exception:
            pass

        final_response = soften_response(final_response)

        # ── Phase 5: Safety validation ─────────────────────────────────────
        raw_response   = final_response
        safety_result  = _safety_validator.validate(
            response=final_response,
            patient_context=patient_context,
            session_id=current_session_id,
            patient_id=patient_id,
        )
        final_response = safety_result.response

        if msg_count % 5 == 0:
            final_response += "\n\n_Just a reminder — everything you share here is completely private and stays between us._ 💙"

        # P1: All DB + memory work runs in background — response is returned immediately
        if user_id:
            background_tasks.add_task(
                _bg_persist_chat_turn,
                current_session_id, user_id, is_new,
                message, final_response, raw_response,
                safety_result, patient_id, memory_svc, _session_summary,
            )

        status = "session_started" if is_new else "message_processed"

        # Check if the agent booked a consultation during this turn
        from medivora_agent.tools import _latest_consultation, _latest_approval_specialty, _latest_triage
        is_medical_report = (
            "📋" in final_response and (
                "medical assessment" in final_response.lower() or
                "health assessment" in final_response.lower()
            )
        )
        is_book_appointment = "book an appointment" in final_response.lower()
        additional_data: dict = {
            "adk_powered": True,
            "is_authenticated": current_user is not None,
            "is_medical_report": is_medical_report,
            "is_book_appointment": is_book_appointment,
        }
        if user_id and user_id in _latest_consultation:
            booked = _latest_consultation.pop(user_id)
            additional_data["consultation_booked"] = booked
            logger.info(f"Consultation booked this turn: {booked['consultation_id']}")
        if user_id and user_id in _latest_approval_specialty:
            specialty = _latest_approval_specialty.pop(user_id)
            additional_data["recommended_specialty"] = specialty
            logger.info(f"Recommended specialty this turn: {specialty}")
        if user_id and user_id in _latest_triage:
            triage = _latest_triage.pop(user_id)
            additional_data["triage"] = triage
            logger.info(f"Triage this turn: level={triage.get('level')}, score={triage.get('risk_score')}, specialty={triage.get('recommended_speciality')}")

        # If specialty still not determined, try to extract from session summary state or response text
        if "recommended_specialty" not in additional_data:
            import re as _re
            # Specialty roots → canonical key (for natural language like "see a gastroenterologist")
            _SPEC_ROOTS = [
                ("gastroenterolog", "gastroenterology"),
                ("cardiolog",       "cardiology"),
                ("gynaecolog",      "womens_health"),
                ("gynecolog",       "womens_health"),
                ("obstetric",       "womens_health"),
                ("pediatr",         "pediatrics"),
                ("paediatr",        "pediatrics"),
                ("dermatolog",      "dermatology"),
                ("orthopaed",       "orthopedics"),
                ("orthoped",        "orthopedics"),
                ("pulmonolog",      "pulmonology"),
                ("neurolog",        "neurology"),
                ("ophthalmolog",    "ophthalmology"),
                ("otolaryngolog",   "ent"),
            ]
            def _extract_specialty(text):
                if not text:
                    return None
                # Structured format first: "Specialty Needed: gastroenterology"
                m = _re.search(r'Special(?:ty|ist)\s+Needed\*?\*?[:\s]+([a-zA-Z_]+)', text, _re.I)
                if m:
                    return m.group(1).lower().strip()
                # Natural language: look for specialty name roots
                text_l = text.lower()
                for root, key in _SPEC_ROOTS:
                    if root in text_l:
                        return key
                return None

            # 1. Try the session summary (most reliable — full structured report)
            sp = _extract_specialty(_session_summary)
            # 2. Try the final response text
            if not sp:
                sp = _extract_specialty(final_response)
            if sp:
                additional_data["recommended_specialty"] = sp
                logger.info(f"Specialty extracted from response text: {sp}")

        chat_response = ChatResponse(
            response=final_response,
            status=status,
            session_id=current_session_id,
            session_state="active",
            additional_data=additional_data,
        )

        logger.info(f"Chat response: status='{status}', session_id='{current_session_id}'")
        return chat_response

    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Internal server error in chat_endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error occurred")

# ── Image Vision Layer ──────────────────────────────────────────────────────
#
# In-process dict keyed by session_id storing the [IMAGE CONTEXT] prompt note
# for the *next* chat turn.  Entries are consumed (pop'd) exactly once so the
# context is injected into the very first follow-up message and then discarded.
#
# This avoids any DB round-trip on the hot chat path while guaranteeing the
# image context is always available when the patient sends their next message.
_pending_image_contexts: Dict[str, str] = {}

# Per-session patient context cache — avoids repeating the ~1.5s
# text-embedding-004 API call on every follow-up turn within a session.
# _patient_id_cache:      user_id (profile UUID) → patients.id
# _patient_context_cache: "{patient_id}:{session_id}" → context dict
_patient_id_cache: Dict[str, str] = {}
_patient_context_cache: Dict[str, dict] = {}

_ALLOWED_IMAGE_MIMETYPES = {
    "image/jpeg", "image/jpg", "image/png",
    "image/webp", "image/gif", "image/heic", "image/heif",
}
_MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB


class ImageAnalysisResponse(BaseModel):
    image_type:         str
    description:        str
    medical_context:    str
    suggested_questions: List[str]
    urgency_flag:       str
    session_id:         str


@app.post("/chat/upload-image", response_model=ImageAnalysisResponse)
@limiter.limit("10/minute")
async def upload_image_endpoint(
    request: Request,
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    current_user: Optional[Dict] = Depends(get_current_user_optional),
):
    """
    Analyse a patient-uploaded image with Gemini 2.5-flash vision.

    - Accepts multipart/form-data with an 'image' file field.
    - Returns structured findings (image_type, description, medical_context,
      suggested_questions, urgency_flag).
    - Stores a [IMAGE CONTEXT] note keyed by session_id so the very next
      /chat turn automatically receives it as part of the prompt.
    - For authenticated users, persists a lightweight fact to patient_memory
      and a full record to chat_image_analyses.
    """
    # ── Validate MIME type ──────────────────────────────────────────────────
    content_type = (image.content_type or "").lower()
    if content_type not in _ALLOWED_IMAGE_MIMETYPES:
        # Attempt to guess from filename
        import mimetypes as _mimetypes
        guessed, _ = _mimetypes.guess_type(image.filename or "")
        if guessed and guessed.lower() in _ALLOWED_IMAGE_MIMETYPES:
            content_type = guessed.lower()
        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported file type '{content_type}'. "
                    "Please upload a JPEG, PNG, WEBP, or GIF image."
                ),
            )

    # ── Read & size-check ───────────────────────────────────────────────────
    image_bytes = await image.read()
    if len(image_bytes) > _MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image too large. Maximum allowed size is 10 MB.",
        )
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file.")

    # ── Resolve session ─────────────────────────────────────────────────────
    current_session_id = session_id or str(uuid.uuid4())

    # Ensure the ADK session exists so the next /chat turn can attach to it
    if current_session_id not in _active_sessions:
        try:
            await adk_session_service.create_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=current_session_id,
            )
            _active_sessions.add(current_session_id)
        except Exception as _e:
            logger.warning(f"Could not pre-create ADK session for image upload: {_e}")

    # ── Analyse image ───────────────────────────────────────────────────────
    vision_svc = get_vision_service()
    analysis = await vision_svc.analyse_image(
        image_bytes=image_bytes,
        mime_type=content_type,
        session_id=current_session_id,
        patient_id=None,  # resolved below if authenticated
    )

    # ── Store context note for next chat turn ───────────────────────────────
    _pending_image_contexts[current_session_id] = analysis["prompt_note"]
    logger.info(
        f"Image context stored for session={current_session_id}: "
        f"type={analysis['image_type']}, urgency={analysis['urgency_flag']}"
    )

    # ── Persist to memory (background, authenticated only) ──────────────────
    user_id = current_user["sub"] if current_user else None
    if user_id:
        db_bg = DatabaseManager()
        memory_svc = MemoryService(db_bg)

        async def _persist():
            try:
                patient_id = await memory_svc.get_patient_id(user_id)
                if patient_id:
                    await vision_svc.persist_to_memory(
                        db=db_bg,
                        patient_id=patient_id,
                        session_id=current_session_id,
                        analysis=analysis,
                        filename=image.filename or "upload",
                    )
            except Exception as _exc:
                logger.warning(f"Vision persist background task failed: {_exc}")

        background_tasks.add_task(_persist)

    return ImageAnalysisResponse(
        image_type=analysis["image_type"],
        description=analysis["description"],
        medical_context=analysis.get("medical_context", ""),
        suggested_questions=analysis.get("suggested_questions", []),
        urgency_flag=analysis.get("urgency_flag", "none"),
        session_id=current_session_id,
    )


# ── Voice Pipeline: ElevenLabs STT (Scribe v1) → Gemini ADK → ElevenLabs TTS ──────────────────
#
# POST /chat/voice
# Accepts multipart/form-data:
#   audio    — audio file (webm/mp4/wav/ogg/mp3)
#   session_id — optional existing session ID
#
# Returns:
#   audio/mpeg stream (MP3 from ElevenLabs TTS)
#   Headers:
#     X-Transcript  — user's transcribed text
#     X-AI-Text     — AI text response (stripped of markdown)
#     X-Session-Id  — session ID used
#

_VOICE_AUDIO_MIMETYPES = {
    "audio/webm", "audio/mp4", "audio/wav", "audio/wave",
    "audio/ogg", "audio/mpeg", "audio/mp3",
    "application/octet-stream",  # Safari often sends this
}
_MAX_VOICE_BYTES = 25 * 1024 * 1024  # 25 MB max


async def _tts_to_bytes(text: str) -> bytes:
    """Generate MP3 audio bytes — ElevenLabs primary, edge-tts fallback."""
    import httpx as _httpx
    el_key = os.getenv("ELEVENLABS_API_KEY", "")
    if el_key:
        try:
            voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
            model_id = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")
            async with _httpx.AsyncClient(timeout=20) as _el_client:
                resp = await _el_client.post(
                    f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                    headers={
                        "xi-api-key": el_key,
                        "Content-Type": "application/json",
                        "Accept": "audio/mpeg",
                    },
                    json={
                        "text": text,
                        "model_id": model_id,
                        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
                    },
                )
            if resp.status_code == 200:
                return resp.content
            logger.warning(f"ElevenLabs TTS {resp.status_code} — falling back to edge-tts")
        except Exception as _el_err:
            logger.warning(f"ElevenLabs TTS error: {_el_err} — falling back to edge-tts")

    # Fallback: edge-tts
    import edge_tts, io, re as _re
    _deva = _re.compile(r'[\u0900-\u097F]')
    _voice = "hi-IN-SwaraNeural" if _deva.search(text) else "en-IN-NeerjaNeural"
    buf = io.BytesIO()
    async for chunk in edge_tts.Communicate(text, _voice).stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    return buf.getvalue()


# Pre-generate and cache the waiting message audio so the frontend plays
# the same voice as the rest of the conversation.
_waiting_audio_cache: bytes | None = None
_WAITING_MESSAGE = (
    "I'm preparing your medical assessment right now. "
    "This usually takes around 10 to 15 seconds — thank you so much for your patience."
)

@app.get("/chat/waiting-audio")
async def waiting_audio_endpoint():
    """Return a cached MP3 of the waiting message using the same TTS pipeline as AI responses."""
    global _waiting_audio_cache
    if _waiting_audio_cache is None:
        _waiting_audio_cache = await _tts_to_bytes(_WAITING_MESSAGE)
    from fastapi.responses import Response as _Resp
    return _Resp(content=_waiting_audio_cache, media_type="audio/mpeg")


@app.post("/chat/voice")
@limiter.limit("20/minute")
async def voice_chat_endpoint(
    request: Request,
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    current_user: Optional[Dict] = Depends(get_current_user_optional),
):
    """
    Server-side voice pipeline:
      1. Transcribe audio via Gemini 2.5-flash multimodal STT
      2. Pass transcript through existing Gemini/ADK chat pipeline (identical to /chat)
      3. Convert AI text response to speech via edge-tts (hi-IN-SwaraNeural or en-IN-NeerjaNeural)
      4. Return MP3 audio stream with transcript/AI-text in response headers
    """
    from fastapi.responses import StreamingResponse
    import io

    # ── 1. Read & validate audio file ───────────────────────────────────────
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")
    if len(audio_bytes) > _MAX_VOICE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Audio file too large. Maximum size is 25 MB.",
        )

    # Pre-STT setup — user_id and session are known before transcription completes
    import asyncio as _aio
    user_id = current_user["sub"] if current_user else None
    current_session_id = session_id or str(uuid.uuid4())
    is_new = current_session_id not in _active_sessions
    db_for_memory = DatabaseManager() if user_id else None
    memory_svc = MemoryService(db_for_memory) if db_for_memory else None

    set_current_user(
        user_id or "",
        name             = (current_user or {}).get("name", ""),
        email            = (current_user or {}).get("email", ""),
        is_authenticated = current_user is not None,
    )

    # ── 2. Parallel: STT + session creation + patient ID lookup ─────────────
    # All three are independent — run concurrently to eliminate serial waiting.
    try:
        _ct = (audio.content_type or "").lower()
        if "mp4" in _ct or "m4a" in _ct:
            _audio_filename = "audio.mp4"
            _audio_mime = "audio/mp4"
        elif "wav" in _ct or "wave" in _ct:
            _audio_filename = "audio.wav"
            _audio_mime = "audio/wav"
        elif "ogg" in _ct:
            _audio_filename = "audio.ogg"
            _audio_mime = "audio/ogg"
        elif "mpeg" in _ct or "mp3" in _ct:
            _audio_filename = "audio.mp3"
            _audio_mime = "audio/mpeg"
        else:
            _audio_filename = "audio.webm"
            _audio_mime = "audio/webm"  # default — Chrome MediaRecorder output

        async def _run_stt_async():
            """Transcribe via ElevenLabs Scribe v1 (primary), Gemini 2.5-flash fallback."""
            import httpx as _hx
            _el_key = os.getenv("ELEVENLABS_API_KEY", "")
            if _el_key:
                try:
                    async with _hx.AsyncClient(timeout=30) as _stt_client:
                        resp = await _stt_client.post(
                            "https://api.elevenlabs.io/v1/speech-to-text",
                            headers={"xi-api-key": _el_key},
                            files={"file": (_audio_filename, audio_bytes, _audio_mime)},
                            data={
                                "model_id": "scribe_v1",
                                "tag_audio_events": "false",
                                "diarize": "false",
                            },
                        )
                    if resp.status_code == 200:
                        return resp.json().get("text", "").strip()
                    logger.warning(f"ElevenLabs STT {resp.status_code} — falling back to Gemini STT")
                except Exception as _stt_err:
                    logger.warning(f"ElevenLabs STT error: {_stt_err} — falling back to Gemini STT")

            # Fallback: Gemini 2.5-flash STT
            from google.genai import types as _stt_types
            def _gemini_stt():
                from google import genai as _stt_genai
                _c = _stt_genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
                return _c.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[
                        _stt_types.Part.from_bytes(data=audio_bytes, mime_type=_audio_mime),
                        _stt_types.Part.from_text(text=(
                            "Transcribe this audio exactly as spoken. "
                            "Output ONLY the transcribed text, nothing else. "
                            "Preserve the original language (Hindi or English or mixed Hinglish)."
                        )),
                    ],
                    config=_stt_types.GenerateContentConfig(
                        thinking_config=_stt_types.ThinkingConfig(thinking_budget=0),
                    ),
                )
            stt_resp = await _aio.to_thread(_gemini_stt)
            return (stt_resp.text or "").strip()

        async def _create_session_if_new():
            if is_new:
                await adk_session_service.create_session(
                    app_name=APP_NAME,
                    user_id=USER_ID,
                    session_id=current_session_id,
                )
                _active_sessions.add(current_session_id)

        async def _fetch_patient_context():
            """Fetch patient_id + context, runs parallel with STT. Uses P2 cache."""
            if not (user_id and memory_svc):
                return None, {"facts": {}, "recent_summaries": []}
            # P2: cache patient_id (immutable)
            pid = _patient_id_cache.get(user_id)
            if not pid:
                pid = await memory_svc.get_patient_id(user_id)
                if pid:
                    _patient_id_cache[user_id] = pid
            if not pid:
                return None, {"facts": {}, "recent_summaries": []}
            # P2: skip DB + embedding on follow-up turns within same session
            ctx_key = f"{pid}:{current_session_id}"
            if not is_new and ctx_key in _patient_context_cache:
                return pid, _patient_context_cache[ctx_key]
            # transcript not yet available (parallel with STT) — use recency retrieval
            ctx = await memory_svc.get_patient_context(pid, query_text=None)
            _patient_context_cache[ctx_key] = ctx
            return pid, ctx

        transcript, _, (patient_id, patient_context) = await _aio.gather(
            _run_stt_async(),
            _create_session_if_new(),
            _fetch_patient_context(),
        )

        logger.info(f"STT transcript ({len(transcript)} chars): '{transcript[:120]}'")
    except Exception as e:
        logger.error(f"STT error: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"Speech transcription failed: {str(e)}")

    if not transcript or len(transcript) < 3:
        # Instead of a hard error, return a graceful TTS nudge so the conversation continues
        _noise_text = "Sorry, I couldn't hear that clearly — could you speak again?"
        try:
            _noise_audio = await _tts_to_bytes(_noise_text)
        except Exception:
            _noise_audio = b""

        from fastapi.responses import StreamingResponse as _SR
        import urllib.parse as _ulp_n
        _safe_n = lambda v: _ulp_n.quote(v.replace("\n", " "), safe=" !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")[:500]
        _noise_hdrs = {
            "X-Transcript": "",
            "X-AI-Text": _safe_n(_noise_text),
            "X-Session-Id": current_session_id,
            "X-Is-Medical-Report": "false",
            "X-Is-Book-Appointment": "false",
            "X-Triage": "",
            "X-Specialty": "",
            "Access-Control-Expose-Headers": (
                "X-Transcript, X-AI-Text, X-Session-Id, "
                "X-Is-Medical-Report, X-Is-Book-Appointment, X-Triage, X-Specialty"
            ),
        }
        if _noise_audio:
            return _SR(io.BytesIO(_noise_audio), media_type="audio/mpeg", headers=_noise_hdrs)
        raise HTTPException(status_code=400, detail=_noise_text)

    # ── 3. Run through existing Gemini/ADK chat pipeline ─────────────────────
    # user_id, current_session_id, is_new, memory_svc, patient_id, patient_context
    # are all already set above from the parallel phase.
    chat_request = ChatRequest(message=transcript, session_id=session_id)

    triage_result = _triage_engine.score(transcript, patient_context)

    if triage_result.is_hard_emergency:
        logger.warning(f"HARD EMERGENCY in voice turn: session={current_session_id}")
        emergency_resp = triage_result.emergency_response
        # TTS for emergency response
        try:
            _emrg_text = stripMarkdown_py(emergency_resp)
            audio_data = await _tts_to_bytes(_emrg_text)
        except Exception as tts_err:
            logger.error(f"TTS error for emergency response: {tts_err}")
            audio_data = b""

        def _safe_hdr_emrg(v: str, n: int = 500) -> str:
            import urllib.parse as _ulp_emrg
            s = v.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
            return _ulp_emrg.quote(s, safe=" !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")[:n]

        headers = {
            "X-Transcript":           _safe_hdr_emrg(transcript),
            "X-AI-Text":              _safe_hdr_emrg(emergency_resp),
            "X-Session-Id":           current_session_id,
            "X-Is-Medical-Report":    "false",
            "X-Is-Book-Appointment":  "false",
            "X-Triage":               "",
            "X-Specialty":            "",
            "Access-Control-Expose-Headers": (
                "X-Transcript, X-AI-Text, X-Session-Id, "
                "X-Is-Medical-Report, X-Is-Book-Appointment, X-Triage, X-Specialty"
            ),
        }
        if audio_data:
            return StreamingResponse(
                io.BytesIO(audio_data),
                media_type="audio/mpeg",
                headers=headers,
            )
        raise HTTPException(status_code=200, detail=emergency_resp)

    # Emotional + memory context
    emotional_ctx = _emotional_builder.build(
        current_message=transcript,
        memory_facts=patient_context.get("facts", {}),
        is_first_message=is_new,
    )
    memory_note  = ""
    _name_facts  = patient_context.get("facts", {}).get("preference", [])
    patient_name = next((f["value"] for f in _name_facts if f["key"] == "name"), "")
    if not patient_name:
        patient_name = (current_user or {}).get("name", "") if user_id else ""
    if is_new and patient_id and memory_svc:
        memory_note = memory_svc.format_context_for_prompt(patient_context)

    if is_new and memory_note:
        adapted_message = _context_builder.build_first_message(
            message=transcript,
            memory_note=memory_note,
            emotional_note=emotional_ctx["system_note"],
            triage_note="",
            patient_name=patient_name if memory_note else None,
        )
    else:
        adapted_message = _context_builder.build(
            message=transcript,
            emotional_note=emotional_ctx["system_note"],
            triage_note="",
        )

    # Inject pending image context if present
    _image_ctx = _pending_image_contexts.pop(current_session_id, None)
    if _image_ctx:
        adapted_message = f"{_image_ctx}\n\n{adapted_message}"

    # ── 3b. Run through the same ADK agent pipeline as /chat ────────────────
    msg_count = get_message_count(current_session_id)

    # Prepend per-message language directive (same as text /chat)
    lang_directive = _detect_language_directive(transcript)
    if lang_directive:
        adapted_message = f"{lang_directive}\n\n{adapted_message}"

    content = genai_types.Content(
        role="user",
        parts=[genai_types.Part.from_text(text=adapted_message)],
    )

    final_response = await _run_adk_agent(current_session_id, USER_ID, content)

    if not final_response:
        final_response = "I'm sorry, I couldn't process your message. Please try again."

    # Session summary (same as text chat)
    _session_summary = ""
    try:
        _sess = await adk_session_service.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=current_session_id
        )
        _session_summary = (_sess.state or {}).get("summary_result", "")
        if _session_summary and "📋" in _session_summary and _session_summary.strip() not in final_response:
            final_response = _session_summary.strip() + "\n\n---\n\n" + final_response
    except Exception:
        pass

    final_response = soften_response(final_response)

    # Safety validation
    raw_response  = final_response
    safety_result = _safety_validator.validate(
        response=final_response,
        patient_context=patient_context,
        session_id=current_session_id,
        patient_id=patient_id,
    )
    final_response = safety_result.response

    if msg_count % 5 == 0:
        final_response += "\n\n_Just a reminder — everything you share here is completely private and stays between us._ 💙"

    # All DB/memory work runs in the background — audio response does not depend on it.
    if user_id:
        background_tasks.add_task(
            _bg_persist_voice_turn,
            current_session_id, user_id, is_new,
            transcript, final_response, raw_response,
            safety_result, patient_id, memory_svc, _session_summary,
        )

    # ── 4. edge-tts TTS ──────────────────────────────────────────────────────
    tts_text = stripMarkdown_py(final_response)

    # Voice: when the response is a medical report, always use a short fixed
    # phrase — reading the full triage card aloud is noisy and unhelpful.
    import re as _re_tts
    _card_start = _re_tts.search(r'📋\s*MEDIVORA|MEDIVORA HEALTH ASSESSMENT', tts_text)
    if _card_start:
        # Medical report — speak a short, warm fixed phrase instead of reading the card
        tts_text = (
            "Your medical assessment is ready. "
            "Please check the screen and tap Book an Appointment to connect with a specialist."
        )

    # Truncate to 800 chars for voice — keeps TTS fast (~1s) and audio short
    if len(tts_text) > 800:
        tts_text = tts_text[:800].rsplit(' ', 1)[0] + "…"

    try:
        audio_data = await _tts_to_bytes(tts_text)
        logger.info(
            f"TTS generated {len(audio_data)} bytes for session={current_session_id}"
        )
    except Exception as tts_err:
        logger.error(f"TTS error: {tts_err}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"Text-to-speech failed: {str(tts_err)}")

    def _safe_header(value: str, max_len: int = 500) -> str:
        """Sanitise a string for use as an HTTP header value.

        HTTP/1.1 headers must be Latin-1 safe. Non-ASCII chars are
        percent-encoded (RFC 3986) so the header stays valid.
        The frontend must call decodeURIComponent() to decode them back.

        For large content (max_len > 500, i.e. medical reports), newlines are
        preserved as %0A so the frontend can reconstruct line breaks.
        For short headers (transcript, small text), newlines are collapsed to spaces.
        """
        import urllib.parse as _ulp
        if max_len > 500:
            # Preserve newlines as %0A for structured content (triage cards)
            sanitised = value.replace("\r\n", "\n").replace("\r", "\n")
            encoded = _ulp.quote(sanitised, safe=" !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
        else:
            sanitised = value.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
            encoded = _ulp.quote(sanitised, safe=" !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
        return encoded[:max_len]

    # ── Detect triage/booking/report flags (mirrors /chat logic) ───────────────
    from medivora_agent.tools import _latest_triage, _latest_approval_specialty
    import json as _json_v

    _v_is_medical_report = (
        "📋" in final_response and (
            "medical assessment" in final_response.lower() or
            "health assessment" in final_response.lower()
        )
    )
    _v_is_book_appointment = "book an appointment" in final_response.lower()

    _v_triage: dict | None = None
    _v_specialty: str | None = None
    if user_id and user_id in _latest_triage:
        _v_triage = _latest_triage.pop(user_id)
    if user_id and user_id in _latest_approval_specialty:
        _v_specialty = _latest_approval_specialty.pop(user_id)

    if not _v_specialty:
        import re as _re_vsp
        _V_SPEC_ROOTS = [
            ("gastroenterolog", "gastroenterology"),
            ("cardiolog",       "cardiology"),
            ("gynaecolog",      "womens_health"),
            ("gynecolog",       "womens_health"),
            ("obstetric",       "womens_health"),
            ("pediatr",         "pediatrics"),
            ("paediatr",        "pediatrics"),
            ("dermatolog",      "dermatology"),
            ("orthopaed",       "orthopedics"),
            ("orthoped",        "orthopedics"),
            ("pulmonolog",      "pulmonology"),
            ("neurolog",        "neurology"),
            ("ophthalmolog",    "ophthalmology"),
            ("otolaryngolog",   "ent"),
        ]
        def _v_extract_sp(text):
            if not text:
                return None
            m = _re_vsp.search(r'Special(?:ty|ist)\s+Needed\*?\*?[:\s]+([a-zA-Z_]+)', text, _re_vsp.I)
            if m:
                return m.group(1).lower().strip()
            tl = text.lower()
            for root, key in _V_SPEC_ROOTS:
                if root in tl:
                    return key
            return None
        _v_specialty = _v_extract_sp(_session_summary) or _v_extract_sp(final_response)

    headers = {
        "X-Transcript":           _safe_header(transcript),
        # Medical reports need the full text so the triage card renders completely.
        # Regular turns stay at 500 chars (only audio matters, text isn't shown).
        "X-AI-Text":              _safe_header(final_response, max_len=4000 if _v_is_medical_report else 500),
        "X-Session-Id":           current_session_id,
        "X-Is-Medical-Report":    "true" if _v_is_medical_report else "false",
        "X-Is-Book-Appointment":  "true" if _v_is_book_appointment else "false",
        "X-Triage":               _safe_header(_json_v.dumps(_v_triage), max_len=2000) if _v_triage else "",
        "X-Specialty":            _v_specialty or "",
        "Access-Control-Expose-Headers": (
            "X-Transcript, X-AI-Text, X-Session-Id, "
            "X-Is-Medical-Report, X-Is-Book-Appointment, X-Triage, X-Specialty"
        ),
    }

    return StreamingResponse(
        io.BytesIO(audio_data),
        media_type="audio/mpeg",
        headers=headers,
    )


def stripMarkdown_py(text: str) -> str:
    """Strip markdown formatting for TTS — mirrors frontend stripMarkdown()."""
    import re as _re
    text = _re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = _re.sub(r'\*(.*?)\*', r'\1', text)
    text = _re.sub(r'#{1,6}\s*', '', text)
    text = _re.sub(r'`{1,3}[^`]*`{1,3}', '', text)
    text = _re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = text.replace('&nbsp;', ' ').replace('•', '').replace('📋', '').replace('💙', '')
    text = _re.sub(r'<[^>]+>', '', text)
    text = _re.sub(r'\n{2,}', '. ', text)
    text = text.replace('\n', ' ')
    return text.strip()


# ── Pre-login Chat Restore ───────────────────────────────────────

class RestoreMessage(BaseModel):
    role: str        # "user" | "ai"
    text: str
    timestamp: Optional[str] = None
    isReport: Optional[bool] = False
    isBooking: Optional[bool] = False

class ChatRestoreRequest(BaseModel):
    messages: List[RestoreMessage]

    @field_validator('messages')
    @classmethod
    def validate_messages(cls, v):
        if len(v) > 50:
            raise ValueError('Too many messages (max 50)')
        total_chars = sum(len(m.text) for m in v)
        if total_chars > 100_000:
            raise ValueError('Message content too large')
        return v


@app.post("/chat/restore")
async def restore_prelogin_chat(
    body: ChatRestoreRequest,
    current_user: Dict = Depends(get_current_user),
):
    """
    Restore a pre-login anonymous chat for an authenticated user.
    Creates a new session, saves all messages to DB, and injects
    conversation history into the ADK session so the agent can continue
    with full context.
    """
    user_id = current_user["sub"]
    user_name = current_user.get("name", "") or current_user.get("email", "").split("@")[0]

    if not body.messages:
        raise HTTPException(status_code=400, detail="No messages to restore")

    new_session_id = str(uuid.uuid4())

    try:
        # 1. Create ADK session
        await adk_session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=new_session_id,
        )
        _active_sessions.add(new_session_id)

        # 2. Inject conversation history into ADK session so the agent
        #    has full context for the next message
        adk_session = await adk_session_service.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=new_session_id
        )
        from google.adk.events import Event
        AGENT_NAME = "medivora_medical_assistant"
        for msg in body.messages:
            try:
                role    = "user" if msg.role == "user" else "model"
                author  = "user" if msg.role == "user" else AGENT_NAME
                content = genai_types.Content(
                    role=role,
                    parts=[genai_types.Part.from_text(text=msg.text)],
                )
                event = Event(author=author, content=content)
                await adk_session_service.append_event(adk_session, event)
            except Exception as e:
                logger.warning(f"restore: failed to inject event: {e}")

        # 3. Persist session + messages in DB
        db = DatabaseManager()
        # Use the first user message as the session title
        first_user_text = next((m.text for m in body.messages if m.role == "user"), "Restored session")
        title = first_user_text[:80] + ("…" if len(first_user_text) > 80 else "")
        await db.create_chat_session(new_session_id, user_id, title)

        for msg in body.messages:
            sender = "user" if msg.role == "user" else "ai"
            await db.save_chat_message(new_session_id, user_id, sender, msg.text)

        await db.touch_chat_session(new_session_id)

        logger.info(f"Chat restored: session={new_session_id}, user={user_id}, messages={len(body.messages)}")

        return {
            "session_id":    new_session_id,
            "message_count": len(body.messages),
            "status":        "restored",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"restore_prelogin_chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not restore chat session")


# ── User (Patient) Authentication ────────────────────────────────

# Lockout configuration
MAX_FAILED_ATTEMPTS = int(os.getenv("MAX_FAILED_LOGIN_ATTEMPTS", "5"))
LOCKOUT_WINDOW_MINUTES = int(os.getenv("LOCKOUT_WINDOW_MINUTES", "15"))
ADMIN_SETUP_KEY = os.getenv("ADMIN_SETUP_KEY", "")  # Must be set in .env to enable first-run setup


class RegisterRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    phone: str = Field(..., min_length=3, max_length=100, description="Phone number or email")
    password: str = Field(..., min_length=6, max_length=100)


class LoginRequest(BaseModel):
    phone: str = Field(..., min_length=3, max_length=100, description="Phone number or email")
    password: str = Field(..., min_length=4, max_length=100)


@app.post("/payment/create-order")
async def create_razorpay_order(
    request: Request,
    current_user: Dict = Depends(get_current_user),
):
    """Create a Razorpay order for a consultation booking."""
    import razorpay, time
    body = await request.json()
    amount_inr = int(body.get("amount", 500))   # full amount in rupees
    doctor_id  = body.get("doctor_id", "")
    doctor_name = body.get("doctor_name", "Doctor")

    key_id     = settings.RAZORPAY_KEY_ID
    key_secret = settings.RAZORPAY_KEY_SECRET

    if not key_id or not key_secret or "REPLACE_ME" in key_id:
        raise HTTPException(status_code=503, detail="Payment gateway not configured. Set RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET in backend .env")

    client = razorpay.Client(auth=(key_id, key_secret))
    order_data = {
        "amount":   amount_inr * 100,   # paise
        "currency": "INR",
        "receipt":  f"medi_{doctor_id[-8:]}_{int(time.time())}",
        "notes": {
            "doctor_id":   doctor_id,
            "doctor_name": doctor_name,
            "patient_id":  current_user.get("id", ""),
        },
    }
    try:
        order = client.order.create(data=order_data)
    except Exception as e:
        logger.error(f"Razorpay order creation failed: {e}")
        raise HTTPException(status_code=502, detail=f"Payment gateway error: {str(e)}")

    return {
        "order_id":  order["id"],
        "amount":    order["amount"],
        "currency":  order["currency"],
        "key_id":    key_id,
    }


@app.post("/auth/register")
@limiter.limit("5/minute")
async def register_user(request: Request, body: RegisterRequest):
    """Register a new patient account."""
    valid, err = validate_password_strength(body.password, role="patient")
    if not valid:
        raise HTTPException(status_code=422, detail=err)
    try:
        db = DatabaseManager()
        user = await db.register_user(body.name, body.phone, body.password)
        if user:
            token = create_token(user["id"], role="patient")
            logger.info(f"Patient registered: {body.phone}")
            return {"message": "Registration successful", "user": user, "token": token, "role": "patient"}
        else:
            raise HTTPException(status_code=409, detail="Phone number already registered. Please login.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering user: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")


@app.post("/auth/login")
@limiter.limit("10/minute")
async def login_user(request: Request, body: LoginRequest):
    """Patient-only login endpoint. Returns patient JWT on success."""
    ip = request.client.host if request.client else "unknown"
    try:
        db = DatabaseManager()

        # Brute-force lockout check
        failed = await db.count_failed_attempts(body.phone, "patient", LOCKOUT_WINDOW_MINUTES)
        if failed >= MAX_FAILED_ATTEMPTS:
            logger.warning(f"Patient account locked: {body.phone}")
            raise HTTPException(
                status_code=429,
                detail=f"Too many failed attempts. Account locked for {LOCKOUT_WINDOW_MINUTES} minutes."
            )

        user = await db.login_user(body.phone, body.password)
        if user:
            await db.reset_failed_attempts(body.phone, "patient")
            await db.record_login_attempt(body.phone, "patient", True, ip)
            token = create_token(user["id"], role="patient")
            logger.info(f"Patient login successful: {body.phone}")
            return {"message": "Login successful", "user": user, "token": token, "role": "patient"}

        await db.record_login_attempt(body.phone, "patient", False, ip)
        raise HTTPException(status_code=401, detail="Invalid phone number or password")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error logging in: {e}")
        raise HTTPException(status_code=500, detail="Login failed")


@app.post("/auth/send-otp")
@limiter.limit("5/minute")
async def send_otp(request: Request, phone: str = Form(...)):
    """Legacy endpoint — kept for backwards compatibility. New code uses routers/auth.py."""
    import random
    phone = phone.strip()
    if not phone or len(phone) < 6:
        raise HTTPException(status_code=422, detail="Invalid phone number")
    otp = str(random.randint(100000, 999999))
    db = DatabaseManager()
    await db.create_otp(phone, otp, ttl_minutes=1)
    logger.info(f"OTP generated for {phone}: {otp}")
    return {"message": "OTP ready (demo mode)", "demo": True, "otp": otp}


@app.post("/auth/verify-otp")
@limiter.limit("10/minute")
async def verify_otp(
    request: Request,
    phone: str = Form(...),
    otp: str = Form(...),
    name: str = Form(default=""),
):
    """
    Verify the OTP. If the user is new and name is provided, create the account.
    Returns JWT + user on success.
    """
    phone = phone.strip()
    otp = otp.strip()
    db = DatabaseManager()

    ok = await db.verify_otp(phone, otp)
    if not ok:
        raise HTTPException(status_code=401, detail="Invalid or expired OTP. Please request a new one.")

    # Check if user already exists
    user = await db.get_user_by_phone(phone)
    if user:
        token = create_token(user["id"], role="patient")
        logger.info(f"OTP login successful: {phone}")
        return {"message": "Login successful", "user": user, "token": token, "role": "patient", "new_user": False}

    # New user — need a name to register
    if not name.strip():
        return {"message": "New user — name required", "new_user": True, "phone": phone}

    # Create account with a random placeholder password (OTP-only users don't use passwords)
    import secrets
    placeholder_pw = secrets.token_hex(16)
    user = await db.register_user(name.strip(), phone, placeholder_pw)
    if not user:
        raise HTTPException(status_code=500, detail="Could not create account. Please try again.")
    token = create_token(user["id"], role="patient")
    logger.info(f"New patient registered via OTP: {phone}")
    return {"message": "Account created", "user": user, "token": token, "role": "patient", "new_user": True}


@app.delete("/account")
async def delete_account(current_user: Dict = Depends(get_current_user)):
    """Permanently delete the authenticated user's account and all associated data."""
    user_id = current_user["sub"]
    role    = current_user.get("role", "patient")
    try:
        db = DatabaseManager()

        if role == "doctor":
            # Anonymise and deactivate the doctor record
            await db.update_doctor_profile(user_id,
                available_status="inactive",
                nmc_number="DELETED",
                clinic_name="",
                clinic_address="",
                clinic_phone="",
            )
            # Mark profile as deleted
            db.client.table("doctors").update({
                "name": "Deleted Account",
                "phone": f"deleted_{user_id[:8]}",
                "email": "",
                "status": "deleted",
            }).eq("id", user_id).execute()
        else:
            # Patients: anonymise profiles row
            try:
                db.client.table("profiles").update({
                    "first_name":    "Deleted",
                    "last_name":     "Account",
                    "phone":         f"deleted_{user_id[:8]}",
                    "email":         "",
                    "status":        "deleted",
                    "password_hash": "",
                }).eq("id", user_id).execute()
            except Exception:
                pass

            # Anonymise patients row
            try:
                db.client.table("patients").update({
                    "medical_history":     None,
                    "allergies":           None,
                    "current_medications": None,
                    "address":             None,
                    "emergency_contact_name":  None,
                    "emergency_contact_phone": None,
                }).eq("profile_id", user_id).execute()
            except Exception:
                pass

            # Try to delete the Supabase Auth user (requires service role key)
            try:
                from supabase import create_client
                from config import settings
                admin_client = create_client(settings.SUPABASE_URL, getattr(settings, "SUPABASE_SERVICE_ROLE_KEY", settings.SUPABASE_KEY))
                admin_client.auth.admin.delete_user(user_id)
                logger.info(f"Supabase auth user {user_id} deleted")
            except Exception as e:
                logger.warning(f"Could not delete Supabase auth user {user_id}: {e}")

        logger.info(f"Account deleted: {user_id} (role={role})")
        return {"message": "Account deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting account {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete account. Please try again.")


@app.get("/auth/user/{user_id}")
async def get_user(user_id: str, current_user: Dict = Depends(get_current_user)):
    """Get patient info by ID (authenticated, own record only)."""
    if current_user["sub"] != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    try:
        db = DatabaseManager()
        user = await db.get_user_by_id(user_id)
        if user:
            # If profile has no first/last name, backfill from JWT auth metadata and persist
            auth_name = (current_user.get("name") or "").strip()
            has_name = bool(user.get("first_name") or user.get("last_name"))
            if auth_name and not has_name:
                name_parts = auth_name.split(" ", 1)
                first_nm = name_parts[0]
                last_nm  = name_parts[1] if len(name_parts) > 1 else ""
                try:
                    db.client.table("profiles").update({
                        "first_name": first_nm,
                        "last_name":  last_nm,
                    }).eq("id", user_id).execute()
                    user["first_name"] = first_nm
                    user["last_name"]  = last_nm
                except Exception:
                    pass
            return {"user": user}

        # No profiles row — user authenticated via Supabase Auth directly.
        # Auto-create a minimal profile keyed on the Supabase UID.
        phone = current_user.get("phone", "")
        email = current_user.get("email", "") or ""
        if phone and not phone.startswith("+"):
            phone = "+" + phone
        import secrets, bcrypt
        pw_hash = bcrypt.hashpw(secrets.token_hex(16).encode(), bcrypt.gensalt()).decode()
        # Names live inside user_metadata in the Supabase JWT
        user_meta = current_user.get("user_metadata", {}) or {}
        first_nm = (user_meta.get("first_name") or "").strip()
        last_nm  = (user_meta.get("last_name") or "").strip()
        if not first_nm and not last_nm:
            full = (user_meta.get("full_name") or current_user.get("name") or "").strip()
            parts = full.split(" ", 1) if full else []
            first_nm = parts[0] if parts else ""
            last_nm  = parts[1] if len(parts) > 1 else ""
        try:
            result = db.client.table("profiles").insert({
                "id":            user_id,
                "first_name":    first_nm,
                "last_name":     last_nm,
                "phone":         phone or f"+00{user_id.replace('-','')[:10]}",
                "email":         email if email else None,
                "password_hash": pw_hash,
                "user_type":     "patient",
                "role":          "patient",
            }).execute()
            new_user = result.data[0] if result.data else None
        except Exception as insert_err:
            logger.warning(f"Auto-create profile failed for {user_id}: {insert_err}")
            new_user = None

        if new_user:
            # Also ensure a patients row exists
            try:
                db.client.table("patients").insert({
                    "profile_id":          user_id,
                    "medical_history":     [],
                    "allergies":           [],
                    "current_medications": [],
                    "chronic_conditions":  [],
                }).execute()
            except Exception:
                pass  # Already exists or non-critical
            return {"user": new_user}
        raise HTTPException(status_code=404, detail="User not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving user")


@app.put("/auth/user/{user_id}")
async def update_user_profile(
    user_id: str,
    current_user: Dict = Depends(get_current_user),
    first_name: str = Form(default=None),
    last_name: str = Form(default=None),
    email: str = Form(default=None),
    phone: str = Form(default=None),
    age: str = Form(default=None),
    gender: str = Form(default=None),
    blood_group: str = Form(default=None),
    height_cm: str = Form(default=None),
    weight_kg: str = Form(default=None),
    address: str = Form(default=None),
    emergency_contact: str = Form(default=None),
    emergency_contact_name: str = Form(default=None),
    emergency_contact_phone: str = Form(default=None),
    emergency_contact_relation: str = Form(default=None),
    medical_history: str = Form(default=None),
    current_medications: str = Form(default=None),
    allergies: str = Form(default=None),
    chronic_conditions: str = Form(default=None),
    is_smoker: str = Form(default=None),
    is_alcohol_user: str = Form(default=None),
    is_pregnant: str = Form(default=None),
    is_nursing: str = Form(default=None),
):
    """Update patient profile (authenticated, own record only).
    Phone can be added/updated for email-signup users (whose stored phone is a placeholder)."""
    if current_user["sub"] != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    try:
        db = DatabaseManager()
        now_iso = datetime.now(timezone.utc).isoformat()

        # Split: profiles table gets name/email/phone, patients table gets medical/demographic
        profile_data = {}
        patient_data = {}

        if first_name is not None and first_name.strip():
            profile_data["first_name"] = first_name.strip()
        if last_name is not None and last_name.strip():
            profile_data["last_name"] = last_name.strip()
        if email is not None and email.strip():
            profile_data["email"] = email.strip()

        # Allow phone update for email-signup users (JWT has no phone claim)
        # For OTP/phone-based users the JWT "phone" claim is set — disallow changing it
        if phone is not None and phone.strip():
            cleaned_phone = phone.strip()
            if not cleaned_phone.startswith("+"):
                cleaned_phone = "+" + cleaned_phone
            jwt_phone = (current_user.get("phone") or "").strip()
            if not jwt_phone:
                # Email-signup user — allow setting phone in profiles table
                profile_data["phone"] = cleaned_phone

        if age is not None and age.strip():
            try: patient_data["age"] = int(age)
            except ValueError: pass
        if gender is not None and gender.strip():
            patient_data["gender"] = gender.strip()
        if blood_group is not None and blood_group.strip():
            patient_data["blood_group"] = blood_group.strip()
        if height_cm is not None and height_cm.strip():
            try: patient_data["height_cm"] = float(height_cm)
            except ValueError: pass
        if weight_kg is not None and weight_kg.strip():
            try: patient_data["weight_kg"] = float(weight_kg)
            except ValueError: pass
        if address is not None and address.strip():
            patient_data["address"] = address.strip()
        # Emergency contact — legacy single-field and new split fields
        if emergency_contact is not None and emergency_contact.strip():
            patient_data["emergency_contact_phone"] = emergency_contact.strip()
        if emergency_contact_name is not None and emergency_contact_name.strip():
            patient_data["emergency_contact_name"] = emergency_contact_name.strip()
        if emergency_contact_phone is not None and emergency_contact_phone.strip():
            patient_data["emergency_contact_phone"] = emergency_contact_phone.strip()
        if emergency_contact_relation is not None and emergency_contact_relation.strip():
            patient_data["emergency_contact_relation"] = emergency_contact_relation.strip()
        # JSONB array fields — stored as comma-separated text converted to JSON arrays
        def _to_json_array(val: str):
            import json
            items = [v.strip() for v in val.split(',') if v.strip()]
            return json.dumps(items)
        if medical_history is not None and medical_history.strip():
            patient_data["medical_history"] = _to_json_array(medical_history)
        if current_medications is not None and current_medications.strip():
            patient_data["current_medications"] = _to_json_array(current_medications)
        if allergies is not None and allergies.strip():
            patient_data["allergies"] = _to_json_array(allergies)
        if chronic_conditions is not None and chronic_conditions.strip():
            patient_data["chronic_conditions"] = _to_json_array(chronic_conditions)
        # Boolean lifestyle flags
        for flag_key, flag_val in [
            ("is_smoker", is_smoker), ("is_alcohol_user", is_alcohol_user),
            ("is_pregnant", is_pregnant), ("is_nursing", is_nursing),
        ]:
            if flag_val is not None:
                patient_data[flag_key] = flag_val.lower() == "true"

        if not profile_data and not patient_data:
            raise HTTPException(status_code=400, detail="No fields to update")

        # Update profiles table (first_name, last_name, email)
        if profile_data:
            profile_data["updated_at"] = now_iso
            db.client.table("profiles").update(profile_data).eq("id", user_id).execute()

        # Update patients table (medical/demographic fields)
        if patient_data:
            patient_data["updated_at"] = now_iso
            patient_id = await db._resolve_patient_id(user_id)
            if patient_id:
                db.client.table("patients").update(patient_data).eq("id", patient_id).execute()
            else:
                patient_data["profile_id"] = user_id
                db.client.table("patients").insert(patient_data).execute()

        # Return merged profile + patient data
        updated_user = await db.get_user_by_id(user_id)
        return {"message": "Profile updated", "user": updated_user}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user profile: {e}")
        raise HTTPException(status_code=500, detail="Error updating profile")


# ── Admin Authentication ──────────────────────────────────────────

class AdminLoginRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=100)
    password: str = Field(..., min_length=8, max_length=100)


class AdminSetupRequest(BaseModel):
    setup_key: str
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., min_length=5, max_length=200)
    password: str = Field(..., min_length=10, max_length=100)
    full_name: str = Field(..., min_length=1, max_length=100)


@app.post("/admin/setup")
@limiter.limit("3/hour")
async def admin_setup(request: Request, body: AdminSetupRequest):
    """
    One-time admin creation endpoint. Protected by ADMIN_SETUP_KEY env variable.
    Only works when no admin exists yet (first-run), or when setup key matches.
    """
    if not ADMIN_SETUP_KEY or body.setup_key != ADMIN_SETUP_KEY:
        raise HTTPException(status_code=403, detail="Invalid setup key")

    valid, err = validate_password_strength(body.password, role="admin")
    if not valid:
        raise HTTPException(status_code=422, detail=err)

    try:
        db = DatabaseManager()
        admin = await db.create_admin(body.username, body.email, body.password, body.full_name)
        if admin:
            logger.info(f"Admin account created: {body.username}")
            return {"message": "Admin account created successfully", "admin": admin}
        raise HTTPException(status_code=409, detail="Admin username or email already exists")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating admin: {e}")
        raise HTTPException(status_code=500, detail="Admin setup failed")


@app.post("/admin/login")
@limiter.limit("5/minute")
async def admin_login(request: Request, body: AdminLoginRequest):
    """
    Admin-only login. Returns a short-lived JWT (4h) with role=admin.
    Subject to 5-attempt brute-force lockout per 15 minutes.
    """
    ip = request.client.host if request.client else "unknown"
    try:
        db = DatabaseManager()

        # Brute-force lockout
        failed = await db.count_failed_attempts(body.username, "admin", LOCKOUT_WINDOW_MINUTES)
        if failed >= MAX_FAILED_ATTEMPTS:
            logger.warning(f"Admin account locked after failed attempts: {body.username}")
            raise HTTPException(
                status_code=429,
                detail=f"Too many failed attempts. Account locked for {LOCKOUT_WINDOW_MINUTES} minutes."
            )

        admin = await db.login_admin(body.username, body.password)
        if admin:
            await db.reset_failed_attempts(body.username, "admin")
            await db.record_login_attempt(body.username, "admin", True, ip)
            token = create_token(admin["id"], role="admin")
            logger.info(f"Admin login successful: {body.username}")
            return {"message": "Login successful", "admin": admin, "token": token, "role": "admin"}

        await db.record_login_attempt(body.username, "admin", False, ip)
        raise HTTPException(status_code=401, detail="Invalid admin credentials")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Admin login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")


@app.get("/admin/analytics")
async def get_admin_analytics(current_admin: Dict = Depends(require_admin)):
    """Rich analytics data for the admin dashboard."""
    from collections import Counter, defaultdict
    import calendar

    db = DatabaseManager()

    # ── Patients ─────────────────────────────────────────────────────────
    patients_res = db.client.table("patients").select(
        "age, gender, blood_group, chronic_conditions, medical_history, "
        "allergies, is_smoker, is_alcohol_user, is_pregnant, is_nursing, created_at"
    ).execute()
    patients = patients_res.data or []

    # Gender
    gender_raw = Counter(
        (r.get("gender") or "unknown").lower() for r in patients
    )
    gender_dist = [
        {"label": k.capitalize() if k != "unknown" else "Not specified", "value": v}
        for k, v in gender_raw.items()
    ]

    # Age groups
    age_buckets = {"0-17": 0, "18-30": 0, "31-45": 0, "46-60": 0, "60+": 0, "Unknown": 0}
    for r in patients:
        age = r.get("age")
        if not age:
            age_buckets["Unknown"] += 1
        elif age <= 17:
            age_buckets["0-17"] += 1
        elif age <= 30:
            age_buckets["18-30"] += 1
        elif age <= 45:
            age_buckets["31-45"] += 1
        elif age <= 60:
            age_buckets["46-60"] += 1
        else:
            age_buckets["60+"] += 1
    age_dist = [{"label": k, "value": v} for k, v in age_buckets.items() if v > 0]

    # Blood groups
    bg_raw = Counter(r.get("blood_group") or "Unknown" for r in patients)
    blood_group_dist = [{"label": k, "value": v} for k, v in bg_raw.most_common()]

    # Chronic conditions
    all_conditions = []
    for r in patients:
        conds = r.get("chronic_conditions") or []
        if isinstance(conds, list):
            all_conditions.extend([c for c in conds if c])
        elif isinstance(conds, str) and conds:
            all_conditions.append(conds)
        # Also check medical_history text field
        mh = r.get("medical_history") or []
        if isinstance(mh, list):
            all_conditions.extend([c for c in mh if c])
        elif isinstance(mh, str) and mh:
            all_conditions.append(mh)
    conditions_dist = [{"label": k, "value": v} for k, v in Counter(all_conditions).most_common(8)]

    # Lifestyle
    total_p = len(patients)
    lifestyle = [
        {"label": "Smokers",         "value": sum(1 for r in patients if r.get("is_smoker"))},
        {"label": "Alcohol users",   "value": sum(1 for r in patients if r.get("is_alcohol_user"))},
        {"label": "Pregnant",        "value": sum(1 for r in patients if r.get("is_pregnant"))},
        {"label": "Nursing",         "value": sum(1 for r in patients if r.get("is_nursing"))},
    ]

    # Monthly patient sign-ups (last 6 months)
    monthly_patients: dict = defaultdict(int)
    for r in patients:
        ts = r.get("created_at") or ""
        if ts:
            month_key = ts[:7]  # YYYY-MM
            monthly_patients[month_key] += 1

    # ── Consultations ────────────────────────────────────────────────────
    consults_res = db.client.table("consultations").select(
        "doctor_id, status, consultation_type, created_at"
    ).execute()
    consults = consults_res.data or []

    consult_status_dist = [
        {"label": k.capitalize(), "value": v}
        for k, v in Counter(r.get("status") for r in consults).most_common()
    ]
    consult_type_dist = [
        {"label": ("Video" if k == "video" else "In-Person") if k else "Unknown", "value": v}
        for k, v in Counter(r.get("consultation_type") for r in consults).most_common()
    ]

    # Consultations by doctor — enrich with name
    by_doctor_id = Counter(r.get("doctor_id") for r in consults if r.get("doctor_id"))
    doctor_ids = [did for did, _ in by_doctor_id.most_common(8)]
    doctor_names: dict = {}
    if doctor_ids:
        docs_res = db.client.table("doctors").select("id, profile_id").in_("id", doctor_ids).execute()
        prof_ids = [d["profile_id"] for d in (docs_res.data or []) if d.get("profile_id")]
        profs_res = db.client.table("profiles").select("id, first_name, last_name").in_("id", prof_ids).execute()
        prof_map = {p["id"]: p for p in (profs_res.data or [])}
        for d in (docs_res.data or []):
            prof = prof_map.get(d.get("profile_id"), {})
            name = f"Dr. {(prof.get('first_name') or '')} {(prof.get('last_name') or '')}".strip()
            doctor_names[d["id"]] = name if name != "Dr." else f"Doctor {d['id'][:6]}"
    consults_by_doctor = [
        {"label": doctor_names.get(did, f"Doctor {did[:6]}"), "value": cnt}
        for did, cnt in by_doctor_id.most_common(8)
    ]

    # Monthly consultations (last 6 months)
    monthly_consults: dict = defaultdict(int)
    for r in consults:
        ts = r.get("created_at") or ""
        if ts:
            month_key = ts[:7]
            monthly_consults[month_key] += 1

    # Build unified monthly growth (last 6 months)
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    month_labels = []
    for i in range(5, -1, -1):
        year  = now.year  if now.month - i > 0 else now.year - 1
        month = (now.month - i - 1) % 12 + 1
        key   = f"{year}-{month:02d}"
        month_labels.append((key, calendar.month_abbr[month]))
    monthly_growth = [
        {
            "month":         label,
            "patients":      monthly_patients.get(key, 0),
            "consultations": monthly_consults.get(key, 0),
        }
        for key, label in month_labels
    ]

    # ── Doctors ──────────────────────────────────────────────────────────
    doctors_res = db.client.table("doctors").select("available_status").execute()
    doctors = doctors_res.data or []
    doctor_status_dist = [
        {"label": k.capitalize().replace("_", " "), "value": v}
        for k, v in Counter(r.get("available_status") for r in doctors).most_common()
    ]

    # Doctor join requests pipeline
    djr_res = db.client.table("doctor_join_requests").select("status").execute()
    djr_status_dist = [
        {"label": k.capitalize(), "value": v}
        for k, v in Counter(r.get("status") for r in (djr_res.data or [])).most_common()
    ]

    # ── Chat sessions ────────────────────────────────────────────────────
    sessions_res = db.client.table("chat_sessions").select("status, created_at").execute()
    sessions = sessions_res.data or []
    session_status_dist = [
        {"label": k.capitalize().replace("_", " "), "value": v}
        for k, v in Counter(r.get("status") for r in sessions).most_common()
    ]

    # ── Prescriptions ────────────────────────────────────────────────────
    px_res = db.client.table("prescriptions").select("status").execute()
    px_status_dist = [
        {"label": k.capitalize() if k else "Unknown", "value": v}
        for k, v in Counter(r.get("status") for r in (px_res.data or [])).most_common()
    ]

    # ── Revenue — bookings × consultation_fee ────────────────────────────
    # Fetch all doctors with fee info
    all_docs_res = db.client.table("doctors") \
        .select("id, profile_id, consultation_fee") \
        .execute()
    all_docs = all_docs_res.data or []
    doc_fee_map = {d["id"]: float(d.get("consultation_fee") or 0) for d in all_docs}
    doc_profile_map = {d["id"]: d.get("profile_id") for d in all_docs}

    # Count non-cancelled consultations per doctor (reuse consults fetched above)
    booking_count: dict = defaultdict(int)
    for c in consults:
        if c.get("status") not in ("cancelled", "no_show") and c.get("doctor_id"):
            booking_count[c["doctor_id"]] += 1

    # Resolve doctor names for all doctors with bookings
    doc_names: dict = {}
    booked_doc_ids = list(booking_count.keys())
    if booked_doc_ids:
        prof_ids = [doc_profile_map[did] for did in booked_doc_ids if doc_profile_map.get(did)]
        if prof_ids:
            profs_res = db.client.table("profiles") \
                .select("id, first_name, last_name") \
                .in_("id", prof_ids) \
                .execute()
            prof_name_map = {
                p["id"]: f"Dr. {p.get('first_name','')} {p.get('last_name','')}".strip()
                for p in (profs_res.data or [])
            }
            for did in booked_doc_ids:
                pid = doc_profile_map.get(did)
                doc_names[did] = prof_name_map.get(pid, f"Doctor {did[:6]}")

    # Calculate revenue per doctor
    doctor_rev: dict = {
        did: cnt * doc_fee_map.get(did, 0)
        for did, cnt in booking_count.items()
    }
    total_revenue = sum(doctor_rev.values())

    doctor_rev_list = [
        {
            "label": doc_names.get(did, f"Doctor {did[:6]}"),
            "value": round(rev, 2),
            "bookings": booking_count[did],
            "fee": doc_fee_map.get(did, 0),
        }
        for did, rev in sorted(doctor_rev.items(), key=lambda x: -x[1])
        if rev > 0
    ]

    return {
        "patients": {
            "total":         total_p,
            "gender":        gender_dist,
            "age_groups":    age_dist,
            "blood_groups":  blood_group_dist,
            "conditions":    conditions_dist,
            "lifestyle":     lifestyle,
        },
        "consultations": {
            "total":     len(consults),
            "by_status": consult_status_dist,
            "by_type":   consult_type_dist,
            "by_doctor": consults_by_doctor,
        },
        "doctors": {
            "total":        len(doctors),
            "by_status":    doctor_status_dist,
            "join_requests": djr_status_dist,
        },
        "sessions": {
            "total":     len(sessions),
            "by_status": session_status_dist,
        },
        "prescriptions": {
            "total":     len(px_res.data or []),
            "by_status": px_status_dist,
        },
        "monthly_growth": monthly_growth,
        "revenue": {
            "total":      round(total_revenue, 2),
            "by_doctor":  doctor_rev_list,
        },
    }


@app.get("/admin/stats")
async def get_admin_stats(current_admin: Dict = Depends(require_admin)):
    """System-wide statistics (admin only)."""
    try:
        db = DatabaseManager()
        stats = await db.get_system_stats()
        return {"stats": stats, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Error getting admin stats: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving stats")


@app.get("/admin/users")
async def admin_list_users(
    limit: int = 50,
    offset: int = 0,
    current_admin: Dict = Depends(require_admin),
):
    """List all patient accounts (admin only)."""
    try:
        db = DatabaseManager()
        users = await db.get_all_users(limit=limit, offset=offset)
        return {"users": users, "count": len(users), "limit": limit, "offset": offset}
    except Exception as e:
        logger.error(f"Error listing users: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving users")


@app.get("/admin/doctors")
async def admin_list_doctors(current_admin: Dict = Depends(require_admin)):
    """List all doctor accounts (admin only)."""
    try:
        db = DatabaseManager()
        doctors = await db.get_all_doctors()
        return {"doctors": doctors, "count": len(doctors)}
    except Exception as e:
        logger.error(f"Error listing doctors: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving doctors")


class _DoctorStatusBody(BaseModel):
    status: str

@app.patch("/admin/doctors/{doctor_id}/status")
async def admin_update_doctor_status(
    doctor_id: str,
    body: _DoctorStatusBody,
    current_admin: Dict = Depends(require_admin),
):
    """Update doctor account status: available | suspended | on_leave (admin only)."""
    status = body.status
    allowed = {"available", "suspended", "on_leave", "inactive"}
    if status not in allowed:
        raise HTTPException(status_code=422, detail=f"Status must be one of: {allowed}")
    try:
        db = DatabaseManager()
        ok = await db.update_doctor_status(doctor_id, status)
        if not ok:
            raise HTTPException(status_code=404, detail="Doctor not found")
        logger.info(f"Admin {current_admin['sub']} set doctor {doctor_id} status → {status}")
        return {"message": f"Doctor status updated to '{status}'", "doctor_id": doctor_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating doctor status: {e}")
        raise HTTPException(status_code=500, detail="Error updating doctor status")



# ── Doctor Join Requests ──────────────────────────────────────────────────────

class DoctorJoinRequestBody(BaseModel):
    first_name: str
    last_name: str = ""
    phone: str
    email: str = ""
    specialties: str = "general_medicine"
    experience_years: int = 0
    medical_college: str = ""
    nmc_number: str = ""
    clinic_name: str = ""
    clinic_address: str = ""
    consultation_fee: Optional[float] = None
    notes: str = ""

@app.post("/doctor-requests")
async def submit_doctor_join_request(body: DoctorJoinRequestBody):
    """Public endpoint — doctor submits a joining request for admin review."""
    db = DatabaseManager()
    # Prevent duplicate pending requests for the same phone
    existing = db.client.table("doctor_join_requests").select("id", "status").eq("phone", body.phone).eq("status", "pending").limit(1).execute()
    if existing.data:
        raise HTTPException(status_code=409, detail="A pending request already exists for this phone number.")
    data = body.model_dump()
    req = await db.create_doctor_join_request(data)
    return {"message": "Request submitted successfully", "id": req.get("id")}

@app.post("/doctors/me/submit-approval")
async def doctor_submit_approval(current_doctor: Dict = Depends(require_doctor)):
    """Authenticated doctor submits their profile for admin approval.
    Creates a doctor_join_requests row linked to their existing doctor record.
    Note: JWT sub = profile_id (not doctors.id), so we look up by profile_id."""
    db = DatabaseManager()
    profile_id = current_doctor["sub"]

    # Resolve the doctors table row from profile_id
    doc_row = db.client.table("doctors").select("*").eq("profile_id", profile_id).limit(1).execute()
    if not doc_row.data:
        raise HTTPException(status_code=404, detail="Doctor record not found.")
    doctor_row = doc_row.data[0]
    doctor_id = doctor_row["id"]

    # Also get profile info (name, phone, email)
    prof_row = db.client.table("profiles").select("first_name, last_name, phone, email").eq("id", profile_id).limit(1).execute()
    prof = prof_row.data[0] if prof_row.data else {}

    # Block duplicate pending requests
    existing = db.client.table("doctor_join_requests") \
        .select("id", "status") \
        .eq("doctor_id", doctor_id) \
        .eq("status", "pending") \
        .limit(1).execute()
    if existing.data:
        raise HTTPException(status_code=409, detail="You already have a pending approval request.")

    specs_list = doctor_row.get("specialties") or []
    specs_str  = ", ".join(specs_list) if specs_list else "general_medicine"

    data = {
        "doctor_id":        doctor_id,
        "first_name":       prof.get("first_name") or "",
        "last_name":        prof.get("last_name")  or "",
        "phone":            prof.get("phone")      or "",
        "email":            prof.get("email")      or "",
        "specialties":      specs_str,
        "experience_years": doctor_row.get("experience_years") or 0,
        "medical_college":  doctor_row.get("medical_college")  or "",
        "nmc_number":       doctor_row.get("nmc_number")        or "",
        "clinic_name":      doctor_row.get("clinic_name")       or "",
        "clinic_address":   doctor_row.get("clinic_address")    or "",
        "consultation_fee": doctor_row.get("consultation_fee"),
        "notes":            "",
    }
    req = await db.create_doctor_join_request(data)
    logger.info(f"Doctor {doctor_id} submitted approval request {req.get('id')}")
    return {"message": "Approval request submitted successfully", "id": req.get("id")}

@app.get("/admin/doctor-requests")
async def admin_list_doctor_requests(
    status: Optional[str] = None,
    current_admin: Dict = Depends(require_admin),
):
    """List doctor join requests, optionally filtered by status."""
    db = DatabaseManager()
    requests = await db.get_doctor_join_requests(status=status)
    return {"requests": requests, "count": len(requests)}

@app.post("/admin/doctor-requests/{request_id}/approve")
async def admin_approve_doctor_request(
    request_id: str,
    current_admin: Dict = Depends(require_admin),
):
    """Approve a doctor join request.
    - If request has doctor_id: activate the existing self-registered doctor.
    - Otherwise: create a new profile + doctor account (old admin-invite flow)."""
    import uuid as _uuid, bcrypt as _bcrypt
    db = DatabaseManager()
    req = await db.get_doctor_join_request(request_id)
    if not req:
        raise HTTPException(status_code=404, detail="Request not found")
    if req["status"] != "pending":
        raise HTTPException(status_code=409, detail=f"Request is already {req['status']}")

    # ── Self-registered doctor (has doctor_id) ─────────────────────────────
    if req.get("doctor_id"):
        doctor_id = req["doctor_id"]
        db.client.table("doctors") \
            .update({"available_status": "available"}) \
            .eq("id", doctor_id) \
            .execute()
        await db.update_doctor_join_request_status(request_id, "approved", current_admin["sub"])
        logger.info(f"Admin {current_admin['sub']} approved self-registered doctor {doctor_id} via request {request_id}")
        return {"message": "Doctor approved and activated", "doctor_id": doctor_id}

    # ── Admin-invited doctor (no doctor_id) — create new account ──────────
    # Check phone not already registered
    existing_profile = db.client.table("profiles").select("id").eq("phone", req["phone"]).limit(1).execute()
    if existing_profile.data:
        raise HTTPException(status_code=409, detail="A doctor with this phone number already exists.")

    profile_id = str(_uuid.uuid4())
    doctor_id  = str(_uuid.uuid4())
    temp_pw    = _uuid.uuid4().hex[:12]  # temporary password — doctor must reset
    pw_hash    = _bcrypt.hashpw(temp_pw.encode(), _bcrypt.gensalt()).decode()

    # Create profile row
    db.client.table("profiles").insert({
        "id":            profile_id,
        "user_type":     "doctor",
        "role":          "doctor",
        "first_name":    req["first_name"],
        "last_name":     req["last_name"],
        "phone":         req["phone"],
        "email":         req["email"],
        "password_hash": pw_hash,
        "status":        "active",
    }).execute()

    # Create doctor row
    specs = [s.strip() for s in req["specialties"].split(",") if s.strip()]
    doctor_row = {
        "id":               doctor_id,
        "profile_id":       profile_id,
        "specialties":      specs,
        "experience_years": req["experience_years"],
        "medical_college":  req["medical_college"],
        "nmc_number":       req["nmc_number"] or None,
        "clinic_name":      req["clinic_name"] or None,
        "clinic_address":   req["clinic_address"] or None,
        "available_status": "available",
        "license_verified": False,
    }
    if req.get("consultation_fee") is not None:
        doctor_row["consultation_fee"] = req["consultation_fee"]
    db.client.table("doctors").insert(doctor_row).execute()

    # Mark request as approved
    await db.update_doctor_join_request_status(request_id, "approved", current_admin["sub"])

    logger.info(f"Admin {current_admin['sub']} approved doctor request {request_id} → new doctor {doctor_id}")
    return {"message": "Doctor approved and account created", "doctor_id": doctor_id, "temp_password": temp_pw}

@app.post("/admin/doctor-requests/{request_id}/reject")
async def admin_reject_doctor_request(
    request_id: str,
    current_admin: Dict = Depends(require_admin),
):
    """Reject a doctor join request."""
    db = DatabaseManager()
    req = await db.get_doctor_join_request(request_id)
    if not req:
        raise HTTPException(status_code=404, detail="Request not found")
    if req["status"] != "pending":
        raise HTTPException(status_code=409, detail=f"Request is already {req['status']}")
    await db.update_doctor_join_request_status(request_id, "rejected", current_admin["sub"])
    logger.info(f"Admin {current_admin['sub']} rejected doctor request {request_id}")
    return {"message": "Request rejected"}


@app.get("/admin/patients/{profile_id}/family-members")
async def admin_get_patient_family_members(profile_id: str, current_admin: Dict = Depends(require_admin)):
    """Return family members for a specific patient (admin only)."""
    try:
        db = DatabaseManager()
        patient_id = await db._resolve_patient_id(profile_id)
        if not patient_id:
            return {"members": []}
        members = await db.get_family_members(patient_id)
        return {"members": members}
    except Exception as e:
        logger.error(f"Error fetching family members for patient {profile_id}: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving family members")


@app.get("/admin/prescriptions")
async def admin_list_prescriptions(current_admin: Dict = Depends(require_admin)):
    """List all prescriptions with patient and doctor details (admin only)."""
    try:
        db = DatabaseManager()
        approvals = await db.get_all_approvals()
        return {"prescriptions": approvals, "count": len(approvals)}
    except Exception as e:
        logger.error(f"Error listing prescriptions: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving prescriptions")



# ─────────────────────────────────────────────────────────────────────────────
# FAMILY MEMBERS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/family-members")
async def list_family_members(current_user: Dict = Depends(require_patient)):
    """List all family members for the authenticated patient."""
    try:
        db = DatabaseManager()
        patient_id = await db._resolve_patient_id(current_user["sub"])
        if not patient_id:
            raise HTTPException(status_code=404, detail="Patient record not found")
        members = await db.get_family_members(patient_id)
        return {"members": members}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing family members: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving family members")


@app.post("/family-members")
async def add_family_member(body: Dict = Body(...), current_user: Dict = Depends(require_patient)):
    """Add a new family member."""
    try:
        if not body.get("name", "").strip():
            raise HTTPException(status_code=422, detail="Name is required")
        db = DatabaseManager()
        patient_id = await db._resolve_patient_id(current_user["sub"])
        if not patient_id:
            raise HTTPException(status_code=404, detail="Patient record not found")
        member = await db.create_family_member(patient_id, body)
        return {"member": member}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding family member: {e}")
        raise HTTPException(status_code=500, detail="Error adding family member")


@app.put("/family-members/{member_id}")
async def update_family_member(member_id: str, body: Dict = Body(...), current_user: Dict = Depends(require_patient)):
    """Update an existing family member."""
    try:
        db = DatabaseManager()
        patient_id = await db._resolve_patient_id(current_user["sub"])
        if not patient_id:
            raise HTTPException(status_code=404, detail="Patient record not found")
        member = await db.update_family_member(member_id, patient_id, body)
        if not member:
            raise HTTPException(status_code=404, detail="Family member not found")
        return {"member": member}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating family member: {e}")
        raise HTTPException(status_code=500, detail="Error updating family member")


@app.delete("/family-members/{member_id}")
async def delete_family_member(member_id: str, current_user: Dict = Depends(require_patient)):
    """Delete a family member."""
    try:
        db = DatabaseManager()
        patient_id = await db._resolve_patient_id(current_user["sub"])
        if not patient_id:
            raise HTTPException(status_code=404, detail="Patient record not found")
        await db.delete_family_member(member_id, patient_id)
        return {"message": "Family member deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting family member: {e}")
        raise HTTPException(status_code=500, detail="Error deleting family member")


@app.get("/patient/{phone}")
async def get_patient_info(phone: str):
    """Get patient information by phone number"""
    try:
        if not phone.isdigit() or len(phone) < 10:
            raise HTTPException(status_code=400, detail="Invalid phone number format")

        db = DatabaseManager()
        patient = await db.get_patient_by_phone(phone)

        if patient:
            return patient
        else:
            raise HTTPException(status_code=404, detail="Patient not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving patient info: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving patient information")

@app.delete("/session/{session_id}")
async def end_session(session_id: str):
    """End a chat session"""
    try:
        if session_id in _active_sessions:
            _active_sessions.discard(session_id)
            return {"message": "Session ended successfully", "session_id": session_id}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ending session: {e}")
        raise HTTPException(status_code=500, detail="Error ending session")

@app.get("/sessions/active")
async def get_active_sessions():
    """Get count of active sessions"""
    return {
        "active_sessions": len(_active_sessions),
        "timestamp": datetime.now().isoformat(),
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Medical AI Assistant (ADK)",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(_active_sessions),
    }

@app.get("/doctors/count")
async def get_doctors_count():
    """Get the total number of onboarded doctors (cached for 60s)"""
    try:
        cached = _api_cache.get("doctors_count")
        if cached:
            return cached
        db = DatabaseManager()
        doctors = await db.get_all_doctors()
        result = {"count": len(doctors), "timestamp": datetime.now().isoformat()}
        _api_cache.set("doctors_count", result, expire=60)
        return result
    except Exception as e:
        logger.error(f"Error getting doctor count: {e}")
        return {"count": 0, "error": "Unable to retrieve count"}

@app.get("/prescriptions/my")
async def get_my_prescriptions(current_user: Dict = Depends(get_current_user)):
    """Get all prescriptions for the logged-in user, enriched with doctor details.
    Pending prescriptions have their content masked (Phase C: patient wait state)."""
    try:
        db = DatabaseManager()
        user_id = current_user["sub"]
        approvals = await db.get_approvals_by_user(user_id)
        # Enrich with doctor details + check for generated PDF
        for a in approvals:
            if a.get("approved_by"):
                doctor = await db.get_doctor_by_id(a["approved_by"])
                a["doctor_details"] = doctor
            elif a.get("assigned_doctor"):
                doctor = await db.get_doctor_by_id(a["assigned_doctor"])
                a["doctor_details"] = doctor
            else:
                a["doctor_details"] = None

            # Check if PDF has been generated
            doc = await db.get_prescription_document(a["approval_id"])
            a["has_pdf"] = doc is not None
            a["download_available"] = doc is not None and not doc.get("download_used", False)
            a["download_url"] = (
                f"{settings.backend_base_url}/prescriptions/{a['approval_id']}/download"
                if doc is not None else None
            )

            # Phase C: Mask prescription content for unapproved items
            if a.get("status") in ("pending_approval", "provisional"):
                a["prescription_data"] = {"masked": True, "message": "Prescription awaiting doctor approval"}
                if isinstance(a.get("assessment_data"), dict):
                    a["assessment_data"]["prescription"] = "[Hidden until approved by doctor]"
                a["provisional_notice"] = "Report generated by Sensyva AI. Awaiting Human Doctor Review."

        return {"prescriptions": approvals, "count": len(approvals)}
    except Exception as e:
        logger.error(f"Error getting user prescriptions: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving prescriptions")


@app.get("/prescriptions/{patient_id}")
async def get_patient_prescriptions(patient_id: str, current_user: Dict = Depends(get_current_user)):
    """Get all prescriptions for a patient (authenticated)"""
    try:
        db = DatabaseManager()
        approvals = await db.get_approvals_by_patient(patient_id)
        return {"prescriptions": approvals, "count": len(approvals)}
    except Exception as e:
        logger.error(f"Error getting prescriptions: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving prescriptions")

@app.get("/prescriptions")
async def get_all_prescriptions(current_user: Dict = Depends(require_doctor)):
    """Get all prescriptions (doctor only)"""
    try:
        db = DatabaseManager()
        approvals = await db.get_all_approvals()
        return {"prescriptions": approvals, "count": len(approvals)}
    except Exception as e:
        logger.error(f"Error getting all prescriptions: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving prescriptions")


@app.get("/doctors/patients")
async def get_doctor_patients(current_user: Dict = Depends(require_doctor)):
    """
    Doctor: get all unique patients who have submitted an AI assessment.
    Returns patient list with their latest assessment, risk level, and prescription status.
    """
    try:
        db = DatabaseManager()
        # Get all approval records
        all_approvals = await db.get_all_approvals()

        # Group by patient_id — keep latest per patient
        patients_map = {}
        for approval in (all_approvals or []):
            pid = approval.get("patient_id") or approval.get("user_id")
            if not pid:
                continue
            existing = patients_map.get(pid)
            # Keep the most recent approval per patient
            if not existing or approval.get("created_at", "") > existing.get("latest_created_at", ""):
                assessment = approval.get("assessment_data") or {}
                patients_map[pid] = {
                    "patient_id":    pid,
                    "patient_name":  approval.get("patient_name") or assessment.get("patient_name") or "Patient",
                    "phone":         approval.get("patient_phone") or "",
                    "age":           approval.get("patient_age") or assessment.get("age") or "—",
                    "gender":        approval.get("patient_gender") or assessment.get("gender") or "—",
                    "latest_condition": assessment.get("primary_condition") or assessment.get("diagnosis") or "Assessment",
                    "latest_risk":   assessment.get("risk_level") or "ROUTINE",
                    "latest_status": approval.get("status") or "pending_approval",
                    "latest_created_at": approval.get("created_at") or "",
                    "total_consultations": 0,
                    "approval_id":   approval.get("approval_id") or "",
                }

        # Count total consultations per patient
        for pid, pdata in patients_map.items():
            pdata["total_consultations"] = sum(
                1 for a in (all_approvals or [])
                if (a.get("patient_id") or a.get("user_id")) == pid
            )

        patients = sorted(
            patients_map.values(),
            key=lambda x: x.get("latest_created_at", ""),
            reverse=True,
        )
        return {"patients": patients, "count": len(patients)}
    except Exception as e:
        logger.error(f"Error getting doctor patients: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving patients")


@app.get("/doctors/patients/{patient_id}")
async def get_patient_detail(
    patient_id: str,
    current_user: Dict = Depends(require_doctor),
):
    """
    Doctor: full history for one patient — all assessments, prescriptions, consultations.
    """
    try:
        db = DatabaseManager()
        # Patient profile
        patient = await db.get_user_by_id(patient_id)
        # All their prescriptions/approvals
        approvals = await db.get_approvals_by_patient(patient_id)
        # All their consultations
        consultations = await db.get_patient_consultations(patient_id) if hasattr(db, "get_patient_consultations") else []

        return {
            "patient":       patient or {"id": patient_id, "name": "Patient"},
            "prescriptions": approvals or [],
            "consultations": consultations or [],
        }
    except Exception as e:
        logger.error(f"Error getting patient detail: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving patient details")

# Voice transcription - lazy init so app doesn't crash if API key is missing
voice_agent = None
try:
    voice_agent = VoiceProcessingAgent()
    logger.info("VoiceProcessingAgent initialized successfully")
except Exception as e:
    logger.warning(f"VoiceProcessingAgent not available: {e}")

@app.post("/voice/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    language: str = Form(default="hi-IN")
):
    """Transcribe audio file to text using OpenAI"""
    if not voice_agent:
        raise HTTPException(status_code=503, detail="Voice transcription service not available")
    try:
        audio_data = await audio.read()

        quality_check = await voice_agent.validate_audio_quality(audio_data)
        if not quality_check["valid"]:
            return {
                "success": False,
                "error": ", ".join(quality_check["issues"]),
                "text": "",
                "confidence": 0.0,
                "language": language,
                "provider": "openai"
            }

        result = await voice_agent.process_audio_stream(audio_data, language)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

# ── Doctor Registration & Login ───────────────────────────────────

@app.post("/doctors/register")
@limiter.limit("5/minute")
async def register_doctor(
    request: Request,
    name: str = Form(...),
    phone: str = Form(...),
    password: str = Form(...),
    email: str = Form(default=""),
    license_number: str = Form(default=""),
    specialties: str = Form(default="general_medicine"),
    experience_years: int = Form(default=0),
):
    """Register a new doctor (with password)"""
    try:
        db = DatabaseManager()
        # Check if doctor already exists
        existing = await db.get_doctor_by_phone(phone)
        if existing:
            raise HTTPException(status_code=409, detail="Doctor already registered. Please login.")

        import uuid
        doctor_id = f"doc_{uuid.uuid4().hex[:8]}"
        doctor_data = {
            "id": doctor_id,
            "name": name,
            "phone": phone,
            "email": email,
            "license_number": license_number or f"LIC_{uuid.uuid4().hex[:6].upper()}",
            "specialties": [s.strip() for s in specialties.split(",")],
            "experience_years": experience_years,
            "password_hash": hash_password(password),
        }

        success = await db.save_doctor(doctor_data)
        if success:
            doctor = await db.get_doctor_by_phone(phone)
            token = create_token(doctor_id, role="doctor")
            logger.info(f"Doctor registered: {phone}")
            return {"message": "Doctor registered successfully", "doctor": doctor, "token": token}
        else:
            raise HTTPException(status_code=500, detail="Failed to register doctor")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering doctor: {e}")
        raise HTTPException(status_code=500, detail="Error registering doctor")


@app.post("/doctors/send-otp")
@limiter.limit("5/minute")
async def doctor_send_otp(request: Request, phone: str = Form(...)):
    """Send OTP to doctor's phone number."""
    import random
    phone = phone.strip()
    if not phone or len(phone) < 6:
        raise HTTPException(status_code=422, detail="Invalid phone number")
    db = DatabaseManager()

    # Block if this phone is registered as a patient account
    existing = db.client.table("profiles").select("user_type").eq("phone", phone).limit(1).execute()
    if existing.data and existing.data[0].get("user_type") == "patient":
        raise HTTPException(
            status_code=409,
            detail="This phone number is registered as a patient account. Please use the patient login instead.",
        )

    otp = str(random.randint(100000, 999999))
    await db.create_otp(phone, otp, ttl_minutes=1)
    logger.info(f"Doctor OTP generated for {phone}: {otp}")
    return {"message": "OTP ready (demo mode)", "demo": True, "otp": otp}


@app.post("/doctors/verify-otp")
@limiter.limit("10/minute")
async def doctor_verify_otp(
    request: Request,
    phone:            str = Form(...),
    otp:              str = Form(...),
    name:             str = Form(default=""),
    license_number:   str = Form(default=""),
    specialties:      str = Form(default=""),
    experience_years: str = Form(default="0"),
    email:            str = Form(default=""),
    fee:              str = Form(default="499"),
):
    """
    Verify OTP for doctor.
    - Doctor exists → login, return token.
    - Doctor not found + name provided → register, return token.
    - Doctor not found + no name → return { new_doctor: true }.
    """
    phone = phone.strip()
    otp   = otp.strip()
    db    = DatabaseManager()
    ok = await db.verify_otp(phone, otp)
    if not ok:
        raise HTTPException(status_code=401, detail="Invalid or expired OTP. Please request a new one.")

    # Block if this phone is a patient account
    existing = db.client.table("profiles").select("user_type").eq("phone", phone).limit(1).execute()
    if existing.data and existing.data[0].get("user_type") == "patient":
        raise HTTPException(
            status_code=409,
            detail="This phone number is registered as a patient account. Please use the patient login instead.",
        )

    doctor = await db.get_doctor_by_phone(phone)
    if doctor:
        token = create_token(doctor["id"], role="doctor")
        logger.info(f"Doctor OTP login: {phone}")
        return {"message": "Login successful", "doctor": doctor, "token": token, "role": "doctor", "new_doctor": False}
    if not name.strip():
        return {"message": "New doctor — profile required", "new_doctor": True, "phone": phone}
    import secrets
    doctor_id = f"doc_{uuid.uuid4().hex[:8]}"
    doctor_data = {
        "id":               doctor_id,
        "name":             name.strip(),
        "phone":            phone,
        "email":            email.strip(),
        "license_number":   license_number.strip() or f"NMC_{uuid.uuid4().hex[:6].upper()}",
        "specialties":      [s.strip() for s in specialties.split(",") if s.strip()] or ["general_medicine"],
        "experience_years": int(experience_years) if str(experience_years).isdigit() else 0,
        "fee":              int(fee) if str(fee).isdigit() else 499,
        "password_hash":    hash_password(secrets.token_hex(16)),
        "status":           "pending_approval",
    }
    success = await db.save_doctor(doctor_data)
    if not success:
        raise HTTPException(status_code=500, detail="Could not create doctor account. Please try again.")
    token = create_token(doctor_id, role="doctor")
    doctor = await db.get_doctor_by_phone(phone)
    logger.info(f"New doctor registered via OTP: {phone}")
    return {"message": "Account created", "doctor": doctor, "token": token, "role": "doctor", "new_doctor": True}


@app.post("/doctors/login-via-supabase")
async def doctor_login_via_supabase(current_user: Dict = Depends(get_current_user)):
    """Exchange a verified Supabase JWT for a doctor JWT.

    Called after the frontend completes Supabase Phone OTP verification.
    - Existing doctor → return JWT.
    - New phone (no doctor record yet) → auto-create profile + doctor row, return JWT.
    - Existing PATIENT with consultations → block (409).
    """
    phone = (current_user.get("phone") or "").strip()
    if not phone:
        raise HTTPException(status_code=400, detail="Phone number not found in token.")
    if not phone.startswith("+"):
        phone = "+" + phone

    supabase_uid = current_user["sub"]
    db = DatabaseManager()

    # ── 1. Existing doctor → login ─────────────────────────────────────────
    doctor = await db.get_doctor_by_phone(phone)
    if doctor:
        token = create_token(doctor["id"], role="doctor")
        logger.info(f"Doctor Supabase OTP login: {phone}")
        return {"message": "Login successful", "doctor": doctor, "token": token, "role": "doctor", "new_doctor": False}

    # ── 2. Check for existing patient with real activity → block ───────────
    # The handle_new_user trigger always creates a patient profile on first OTP.
    # Only block if the patient has actual consultations (real patient usage).
    try:
        patient_row = db.client.table("patients").select("id").eq("profile_id", supabase_uid).limit(1).execute()
        if patient_row.data:
            consultations = (
                db.client.table("consultations")
                .select("id")
                .eq("patient_id", patient_row.data[0]["id"])
                .limit(1)
                .execute()
            )
            if consultations.data:
                raise HTTPException(
                    status_code=409,
                    detail="This number is registered as a patient account. Please use the patient login instead.",
                )
    except HTTPException:
        raise
    except Exception:
        pass

    # ── 3. Auto-create doctor account ──────────────────────────────────────
    # Find the real profile for this phone. Two cases:
    # a) Trigger created profiles(id=supabase_uid) for this OTP — use supabase_uid.
    # b) Phone already existed in profiles (different id) — trigger skipped due to
    #    ON CONFLICT; use the existing profile id.

    phone_bare = phone.lstrip("+")
    profile_row = None
    for q_phone in [phone, phone_bare]:
        res = db.client.table("profiles").select("id,user_type").eq("phone", q_phone).limit(1).execute()
        if res.data:
            profile_row = res.data[0]
            break

    if not profile_row:
        # Trigger couldn't create a profile at all — create one now
        try:
            ins = db.client.table("profiles").insert({
                "id":             supabase_uid,
                "phone":          phone,
                "user_type":      "doctor",
                "role":           "doctor",
                "password_hash":  "supabase_managed",
                "status":         "active",
                "phone_verified": True,
            }).execute()
            if ins.data:
                profile_row = {"id": supabase_uid, "user_type": "doctor"}
        except Exception as e:
            logger.error(f"login-via-supabase: profile insert failed: {e}")
            raise HTTPException(status_code=500, detail="Could not create doctor account.")

    profile_id = profile_row["id"]

    # Upgrade the profile to doctor type and normalise phone
    try:
        db.client.table("profiles").update({
            "user_type":      "doctor",
            "role":           "doctor",
            "phone":          phone,
            "phone_verified": True,
        }).eq("id", profile_id).execute()
    except Exception as e:
        logger.error(f"login-via-supabase: profile upgrade failed: {e}")
        raise HTTPException(status_code=500, detail="Could not upgrade profile to doctor.")

    # Remove auto-created patient row (non-fatal if missing)
    try:
        db.client.table("patients").delete().eq("profile_id", profile_id).execute()
    except Exception:
        pass

    # Create doctors row (skip if already exists)
    try:
        db.client.table("doctors").insert({
            "profile_id":       profile_id,
            "available_status": "inactive",
            "specialties":      [],
            "available_slots":  [],
        }).execute()
    except Exception as e:
        # Could be a duplicate (profile already has a doctors row) — check
        existing_doc = db.client.table("doctors").select("id").eq("profile_id", profile_id).limit(1).execute()
        if not existing_doc.data:
            logger.error(f"login-via-supabase: doctors insert failed: {e}")
            raise HTTPException(status_code=500, detail="Could not create doctor record.")

    doctor = await db.get_doctor_by_phone(phone)
    if not doctor:
        raise HTTPException(status_code=500, detail="Doctor account created but could not be retrieved.")

    token = create_token(doctor["id"], role="doctor")
    logger.info(f"Doctor auto-created via Supabase OTP: {phone}")
    return {"message": "Account created", "doctor": doctor, "token": token, "role": "doctor", "new_doctor": True}


@app.post("/doctors/login")
@limiter.limit("10/minute")
async def doctor_login(
    request: Request,
    phone: str = Form(...),
    password: str = Form(...),
):
    """Doctor-only login with phone and password. Returns a 12-hour JWT with role=doctor."""
    ip = request.client.host if request.client else "unknown"
    try:
        db = DatabaseManager()

        # Brute-force lockout
        failed = await db.count_failed_attempts(phone, "doctor", LOCKOUT_WINDOW_MINUTES)
        if failed >= MAX_FAILED_ATTEMPTS:
            logger.warning(f"Doctor account locked: {phone}")
            raise HTTPException(
                status_code=429,
                detail=f"Too many failed attempts. Account locked for {LOCKOUT_WINDOW_MINUTES} minutes."
            )

        doctor = await db.login_doctor(phone, password)
        if doctor:
            await db.reset_failed_attempts(phone, "doctor")
            await db.record_login_attempt(phone, "doctor", True, ip)
            token = create_token(doctor["id"], role="doctor")
            logger.info(f"Doctor login successful: {phone}")
            return {"message": "Login successful", "doctor": doctor, "token": token, "role": "doctor"}

        await db.record_login_attempt(phone, "doctor", False, ip)
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials. If you registered before passwords were required, use /doctors/set-password first."
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in doctor login: {e}")
        raise HTTPException(status_code=500, detail="Login error")


@app.post("/doctors/set-password")
@limiter.limit("5/minute")
async def set_doctor_password(
    request: Request,
    phone: str = Form(...),
    license_number: str = Form(...),
    password: str = Form(..., min_length=6),
):
    """One-time password setup for doctors registered before auth was added"""
    try:
        db = DatabaseManager()
        doctor = await db.get_doctor_by_phone(phone)
        if not doctor:
            raise HTTPException(status_code=404, detail="Doctor not found")
        if doctor.get("license_number") != license_number:
            raise HTTPException(status_code=401, detail="License number mismatch")
        success = await db.set_doctor_password(phone, password)
        if success:
            token = create_token(doctor["id"], role="doctor")
            return {"message": "Password set successfully. You can now login.", "token": token}
        raise HTTPException(status_code=500, detail="Failed to set password")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting doctor password: {e}")
        raise HTTPException(status_code=500, detail="Error setting password")


@app.get("/doctors/{doctor_id}/slots")
async def get_doctor_slots(doctor_id: str):
    """
    Return available time slots for a doctor for the next 7 days.
    Slots are generated from the doctor's available_slots schedule (JSONB array
    of {day, start, end} objects), 30-minute intervals.
    Falls back to Mon–Sat 09:00–17:00 if no slots are configured.
    Already-past slots for today are excluded.
    """
    from datetime import date, datetime, timedelta, time as dtime, timezone

    IST_OFFSET = timezone(timedelta(hours=5, minutes=30))
    now_ist    = datetime.now(IST_OFFSET)
    today      = now_ist.date()

    # Fetch doctor record
    try:
        db     = DatabaseManager()
        doctor = await db.get_doctor_by_id(doctor_id)
    except Exception:
        doctor = None

    # Unavailable doctors have no slots
    if doctor and doctor.get("available_status") in ("offline", "on_leave", "inactive"):
        return {"doctor_id": doctor_id, "slots": {}}

    INTERVAL = timedelta(minutes=30)
    DAY_NAME_TO_WDAY = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6,
    }

    # Build a map of weekday_number -> list of (start_time, end_time, slot_type)
    available_slots = doctor.get("available_slots") if doctor else None
    schedule: dict = {}  # {0: [(dtime(9,0), dtime(17,0), "both")], ...}

    if available_slots and isinstance(available_slots, list) and len(available_slots) > 0:
        for entry in available_slots:
            day_str = str(entry.get("day", "")).strip().lower()
            wday = DAY_NAME_TO_WDAY.get(day_str)
            if wday is None:
                continue
            try:
                sh, sm = map(int, str(entry.get("start", "09:00")).split(":"))
                eh, em = map(int, str(entry.get("end",   "17:00")).split(":"))
                slot_type = str(entry.get("type", "both")).strip().lower()
                if slot_type not in ("online", "offline", "both"):
                    slot_type = "both"
                schedule.setdefault(wday, []).append((dtime(sh, sm), dtime(eh, em), slot_type))
            except Exception:
                continue
    else:
        # Default: Mon–Sat 09:00–23:30, both
        for wday in range(6):
            schedule[wday] = [(dtime(9, 0), dtime(23, 30), "both")]

    slots: dict = {}
    for offset in range(7):
        d = today + timedelta(days=offset)
        ranges = schedule.get(d.weekday(), [])
        if not ranges:
            continue

        day_slots = []
        for (start_t, end_t, slot_type) in ranges:
            cursor = datetime.combine(d, start_t, tzinfo=IST_OFFSET)
            end    = datetime.combine(d, end_t,   tzinfo=IST_OFFSET)
            while cursor <= end:
                if cursor > now_ist + timedelta(minutes=15):
                    day_slots.append({"time": cursor.strftime("%H:%M"), "type": slot_type})
                cursor += INTERVAL

        if day_slots:
            slots[d.isoformat()] = day_slots

    return {"doctor_id": doctor_id, "slots": slots}


@app.get("/doctors")
async def get_all_doctors(specialty: Optional[str] = None):
    """Get all registered doctors visible to patients, optionally filtered by specialty."""
    try:
        db = DatabaseManager()
        doctors = await db.get_all_doctors(patient_facing=True)
        if specialty:
            spec_lower = specialty.lower().strip()
            filtered = [d for d in doctors if spec_lower in (d.get("specialization") or "").lower()]
            # Fall back to full list if no match (prevents empty list)
            doctors = filtered if filtered else doctors
        return {"doctors": doctors, "count": len(doctors), "filtered_by": specialty or None}
    except Exception as e:
        logger.error(f"Error getting doctors: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving doctors")


_PRIVATE_DOCTOR_FIELDS = {"email", "phone", "password_hash", "profile_id"}

@app.get("/doctors/{doctor_id}")
async def get_doctor_public_profile(doctor_id: str):
    """Public profile of a single doctor — personal fields are stripped."""
    try:
        db = DatabaseManager()
        doctor = await db.get_doctor_by_id(doctor_id)
        if not doctor:
            raise HTTPException(status_code=404, detail="Doctor not found")
        # Enrich with full_name from profiles (computed from first_name + last_name)
        profile_id = doctor.get("profile_id")
        if profile_id:
            profile_res = db.client.table("profiles").select("first_name, last_name").eq("id", profile_id).limit(1).execute()
            if profile_res.data:
                doctor["full_name"] = _name(profile_res.data[0].get("first_name"), profile_res.data[0].get("last_name"))
        # Flatten specialties
        specs = doctor.get("specialties") or []
        doctor["specialization"] = specs[0] if specs else "General Physician"
        doctor["city"] = doctor.get("clinic_address") or ""
        # Strip private fields
        for field in _PRIVATE_DOCTOR_FIELDS:
            doctor.pop(field, None)
        return doctor
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting doctor profile {doctor_id}: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving doctor profile")


@app.get("/my/prescriptions")
async def get_my_prescriptions_full(
    current_user: Dict = Depends(get_current_user_optional),
):
    """Patient: list own prescriptions with medicine items and prescribing doctor."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required.")
    try:
        db = DatabaseManager()
        prescriptions = await db.get_patient_prescriptions_full(current_user["sub"])
        return {"prescriptions": prescriptions, "count": len(prescriptions)}
    except Exception as e:
        logger.error(f"Error fetching patient prescriptions: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving prescriptions")



# ── Approval Workflow ────────────────────────────────────────────

@app.get("/approvals/pending")
async def get_pending_approvals(current_user: Dict = Depends(require_doctor)):
    """Get pending approval requests (doctor only)"""
    try:
        db = DatabaseManager()
        approvals = await db.get_pending_approvals(current_user["sub"])
        return {"approvals": approvals, "count": len(approvals)}
    except Exception as e:
        logger.error(f"Error getting pending approvals: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving approvals")


@app.get("/approvals/{approval_id}")
async def get_approval(approval_id: str, current_user: Dict = Depends(get_current_user)):
    """Get a specific approval request (authenticated)"""
    try:
        db = DatabaseManager()
        approval = await db.get_approval_by_id(approval_id)
        if approval:
            return approval
        raise HTTPException(status_code=404, detail="Approval not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting approval: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving approval")


@app.post("/approvals/{approval_id}/approve")
async def approve_prescription(
    approval_id: str,
    notes: str = Form(default=""),
    nmc_number: str = Form(default=""),
    clinic_address: str = Form(default=""),
    current_user: Dict = Depends(require_doctor),
):
    """Approve a prescription with digital signature (doctor only)"""
    try:
        doctor_id = current_user["sub"]
        db = DatabaseManager()

        # Store original prescription snapshot before any changes
        await db.store_original_prescription(approval_id)

        # Compute digital signature hash
        timestamp = datetime.now().isoformat()
        sig_hash = compute_signature_hash(doctor_id, nmc_number, timestamp, approval_id)

        # Update approval status
        success = await db.update_approval_status(
            approval_id=approval_id,
            status="approved",
            doctor_id=doctor_id,
            doctor_notes=notes,
        )

        if not success:
            raise HTTPException(status_code=404, detail="Approval not found")

        # Store signature and NMC details
        await db.store_signature(approval_id, sig_hash, nmc_number, clinic_address)

        # Update doctor profile with NMC if provided
        if nmc_number:
            await db.update_doctor_profile(doctor_id, nmc_number=nmc_number, clinic_address=clinic_address)

        # Log the approval action
        await db.save_edit_log(approval_id, doctor_id, "status", "pending_approval", "approved", "modified")

        # Notify patient
        try:
            approval = await db.get_approval_by_id(approval_id)
            if approval and approval.get("patient_id"):
                doctor_name = current_user.get("name", "Your doctor")
                await db.create_user_notification(
                    user_id=approval["patient_id"],
                    notification_type="prescription",
                    title="Prescription Approved",
                    message=f"Dr. {doctor_name} has approved and signed your prescription. You can now view it in Prescriptions.",
                    priority=2,
                )
        except Exception:
            pass  # Notification failure must not block the approval

        logger.info(f"Prescription approved: {approval_id}, sig: {sig_hash[:16]}")
        return {
            "message": "Prescription approved and signed",
            "approval_id": approval_id,
            "signature_hash": sig_hash,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error approving prescription: {e}")
        raise HTTPException(status_code=500, detail="Error approving prescription")


@app.post("/approvals/{approval_id}/reject")
async def reject_prescription(
    approval_id: str,
    reason: str = Form(...),
    current_user: Dict = Depends(require_doctor),
):
    """Reject a prescription (doctor only)"""
    try:
        doctor_id = current_user["sub"]
        db = DatabaseManager()
        success = await db.update_approval_status(
            approval_id=approval_id,
            status="rejected",
            doctor_id=doctor_id,
            rejection_reason=reason,
        )
        if success:
            logger.info(f"Prescription rejected: {approval_id}")
            # Notify patient
            try:
                approval = await db.get_approval_by_id(approval_id)
                if approval and approval.get("patient_id"):
                    doctor_name = current_user.get("name", "Your doctor")
                    await db.create_user_notification(
                        user_id=approval["patient_id"],
                        notification_type="prescription",
                        title="Prescription Rejected",
                        message=f"Dr. {doctor_name} has reviewed and rejected your prescription request. Reason: {reason or 'Not specified'}.",
                        priority=2,
                    )
            except Exception:
                pass
            return {"message": "Prescription rejected", "approval_id": approval_id}
        raise HTTPException(status_code=404, detail="Approval not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rejecting prescription: {e}")
        raise HTTPException(status_code=500, detail="Error rejecting prescription")


@app.post("/approvals/{approval_id}/modify")
async def modify_prescription(
    approval_id: str,
    modified_prescription: str = Form(...),
    notes: str = Form(default=""),
    nmc_number: str = Form(default=""),
    clinic_address: str = Form(default=""),
    edits_json: str = Form(default="[]"),
    current_user: Dict = Depends(require_doctor),
):
    """Modify a prescription with edit tracking and digital signature (doctor only)"""
    if not notes.strip():
        raise HTTPException(status_code=422, detail="Justification (notes) is required when modifying a prescription")
    try:
        import json as _json
        doctor_id = current_user["sub"]
        db = DatabaseManager()

        # Store original prescription snapshot before edits
        await db.store_original_prescription(approval_id)

        # Get original data for audit trail
        original = await db.get_approval_by_id(approval_id)
        if not original:
            raise HTTPException(status_code=404, detail="Approval not found")

        # Log individual field edits from the frontend diff view
        try:
            edits = _json.loads(edits_json) if edits_json else []
            for edit in edits:
                await db.save_edit_log(
                    approval_id, doctor_id,
                    edit.get("field", "prescription"),
                    edit.get("old_value", ""),
                    edit.get("new_value", ""),
                    edit.get("change_type", "modified"),
                )
        except (_json.JSONDecodeError, TypeError):
            pass

        # Always log the overall prescription change
        old_rx = original.get("prescription_data", {})
        if isinstance(old_rx, dict):
            old_rx = old_rx.get("prescription_text", str(old_rx))
        await db.save_edit_log(approval_id, doctor_id, "prescription_text", str(old_rx), modified_prescription, "modified")

        # Compute digital signature hash
        timestamp = datetime.now().isoformat()
        sig_hash = compute_signature_hash(doctor_id, nmc_number, timestamp, approval_id)

        # Update approval
        success = await db.update_approval_status(
            approval_id=approval_id,
            status="modified",
            doctor_id=doctor_id,
            approved_prescription=modified_prescription,
            doctor_notes=notes,
        )

        if not success:
            raise HTTPException(status_code=404, detail="Approval not found")

        # Store signature
        await db.store_signature(approval_id, sig_hash, nmc_number, clinic_address)
        if nmc_number:
            await db.update_doctor_profile(doctor_id, nmc_number=nmc_number, clinic_address=clinic_address)

        logger.info(f"Prescription modified: {approval_id}, edits logged")
        return {
            "message": "Prescription modified and signed",
            "approval_id": approval_id,
            "signature_hash": sig_hash,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error modifying prescription: {e}")
        raise HTTPException(status_code=500, detail="Error modifying prescription")


# ── Notifications ────────────────────────────────────────────────

@app.get("/notifications/{doctor_id}")
async def get_doctor_notifications(doctor_id: str, unread_only: bool = False, current_user: Dict = Depends(require_doctor)):
    """Get notifications for a doctor (doctor only, own notifications)"""
    if current_user["sub"] != doctor_id:
        raise HTTPException(status_code=403, detail="Access denied")
    try:
        db = DatabaseManager()
        notifications = await db.get_notifications(doctor_id, unread_only)
        return {"notifications": notifications, "count": len(notifications)}
    except Exception as e:
        logger.error(f"Error getting notifications: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving notifications")


@app.post("/notifications/{notification_id}/read")
async def mark_notification_read(notification_id: str, current_user: Dict = Depends(require_doctor)):
    """Mark a notification as read (doctor only)"""
    try:
        db = DatabaseManager()
        success = await db.mark_notification_read(notification_id)
        if success:
            return {"message": "Notification marked as read"}
        raise HTTPException(status_code=404, detail="Notification not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error marking notification read: {e}")
        raise HTTPException(status_code=500, detail="Error updating notification")


# ── Patient Notifications ───────────────────────────────────────

@app.get("/my/notifications")
async def get_my_notifications(unread_only: bool = False, current_user: Dict = Depends(get_current_user)):
    """Get notifications for the currently logged-in patient."""
    try:
        db = DatabaseManager()
        notifications = await db.get_user_notifications(current_user["sub"], unread_only)
        return {"notifications": notifications, "count": len(notifications)}
    except Exception as e:
        logger.error(f"Error getting user notifications: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving notifications")


@app.post("/my/notifications/{notification_id}/read")
async def mark_my_notification_read(notification_id: str, current_user: Dict = Depends(get_current_user)):
    """Mark a notification as read (patient, own notifications only)."""
    try:
        db = DatabaseManager()
        success = await db.mark_notification_read(notification_id, user_id=current_user["sub"])
        if success:
            return {"message": "Notification marked as read"}
        raise HTTPException(status_code=404, detail="Notification not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error marking notification read: {e}")
        raise HTTPException(status_code=500, detail="Error updating notification")


@app.post("/my/notifications/read-all")
async def mark_all_my_notifications_read(current_user: Dict = Depends(get_current_user)):
    """Mark all notifications as read for the current user."""
    try:
        db = DatabaseManager()
        await db.mark_all_notifications_read(current_user["sub"])
        return {"message": "All notifications marked as read"}
    except Exception as e:
        logger.error(f"Error marking all notifications read: {e}")
        raise HTTPException(status_code=500, detail="Error updating notifications")


# ── Prescription Status ─────────────────────────────────────────

@app.get("/prescriptions/status/{approval_id}")
async def get_prescription_status(approval_id: str):
    """Get prescription approval status (for patient polling)"""
    try:
        db = DatabaseManager()
        approval = await db.get_approval_by_id(approval_id)
        if approval:
            return {
                "approval_id": approval_id,
                "status": approval["status"],
                "approved_at": approval.get("approved_at"),
                "doctor_name": approval.get("doctor_name"),
                "doctor_notes": approval.get("doctor_notes"),
                "approved_prescription": approval.get("approved_prescription"),
            }
        raise HTTPException(status_code=404, detail="Approval not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prescription status: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving status")


# ── Chat History ─────────────────────────────────────────────────

@app.get("/chat/sessions/{user_id}")
async def get_user_chat_sessions(user_id: str, current_user: Dict = Depends(get_current_user)):
    """Get all chat sessions for a user (authenticated)"""
    if current_user["sub"] != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    try:
        db = DatabaseManager()
        sessions = await db.get_user_chat_sessions(user_id)
        return {"sessions": sessions, "count": len(sessions)}
    except Exception as e:
        logger.error(f"Error getting chat sessions: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving chat sessions")

@app.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str, current_user: Dict = Depends(get_current_user)):
    """Get all messages for a chat session (authenticated)"""
    try:
        db = DatabaseManager()
        messages = await db.get_chat_conversation(session_id)
        return {"messages": messages, "count": len(messages)}
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving chat history")

@app.delete("/chat/sessions/{session_id}")
async def delete_chat_session(session_id: str, current_user: Dict = Depends(get_current_user)):
    """Delete a chat session and its messages (authenticated)"""
    try:
        db = DatabaseManager()
        success = await db.delete_chat_session(session_id, current_user["sub"])
        if success:
            return {"message": "Chat session deleted"}
        raise HTTPException(status_code=404, detail="Session not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting chat session: {e}")
        raise HTTPException(status_code=500, detail="Error deleting session")


# ── Backfill Chat Session Titles ─────────────────────────────────

@app.post("/chat/sessions/backfill-titles")
async def backfill_session_titles(current_user: Dict = Depends(get_current_user)):
    """Re-generate titles for sessions that have greeting-only titles (e.g. 'Hi')."""
    try:
        db = DatabaseManager()
        user_id = current_user["sub"]
        sessions = await db.get_user_chat_sessions(user_id)
        updated = 0
        for session in sessions:
            title = (session.get("title") or "").strip()
            # Check if title needs updating
            if not title or title.lower().rstrip("!., ") in _GREETING_WORDS or title == _PLACEHOLDER_TITLE or len(title) <= 3:
                sid = session.get("id")
                messages = await db.get_chat_conversation(sid)
                # Find first non-greeting user message
                user_msg = ""
                ai_msg = ""
                for msg in messages:
                    if msg.get("sender") == "user" and not _is_greeting(msg.get("content", "")):
                        user_msg = msg.get("content", "")
                        break
                    if msg.get("sender") == "ai" and msg.get("content", ""):
                        ai_msg = msg.get("content", "")
                # Use whatever context we have
                context_msg = user_msg or ai_msg
                if context_msg:
                    new_title = await _generate_session_title(context_msg, ai_msg if user_msg else "")
                    if new_title and new_title != _PLACEHOLDER_TITLE:
                        await db.update_chat_session_title(sid, new_title)
                        updated += 1
        return {"message": f"Updated {updated} session title(s)", "total_sessions": len(sessions)}
    except Exception as e:
        logger.error(f"Backfill titles error: {e}")
        raise HTTPException(status_code=500, detail="Error backfilling titles")


# ── Drug Blacklist CRUD ──────────────────────────────────────────

@app.get("/drug-blacklist")
async def get_drug_blacklist(current_user: Dict = Depends(require_doctor)):
    """Get the current drug blacklist (doctor only)"""
    try:
        db = DatabaseManager()
        drugs = await db.get_drug_blacklist()
        return {"drugs": drugs, "count": len(drugs)}
    except Exception as e:
        logger.error(f"Error getting drug blacklist: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving blacklist")

@app.post("/drug-blacklist")
async def add_drug_to_blacklist(
    drug_name: str = Form(...),
    drug_category: str = Form(default="schedule_x"),
    reason: str = Form(default=""),
    current_user: Dict = Depends(require_doctor),
):
    """Add a drug to the blacklist (doctor only)"""
    try:
        db = DatabaseManager()
        success = await db.add_drug_to_blacklist(drug_name, drug_category, reason, current_user["sub"])
        if success:
            return {"message": f"Drug '{drug_name}' added to blacklist"}
        raise HTTPException(status_code=500, detail="Failed to add drug")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding drug to blacklist: {e}")
        raise HTTPException(status_code=500, detail="Error adding drug")

@app.delete("/drug-blacklist/{drug_id}")
async def remove_drug_from_blacklist(drug_id: int, current_user: Dict = Depends(require_doctor)):
    """Remove a drug from the blacklist (doctor only)"""
    try:
        db = DatabaseManager()
        success = await db.remove_drug_from_blacklist(drug_id)
        if success:
            return {"message": "Drug removed from blacklist"}
        raise HTTPException(status_code=404, detail="Drug not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing drug from blacklist: {e}")
        raise HTTPException(status_code=500, detail="Error removing drug")


# ── Edit Audit Trail ────────────────────────────────────────────

@app.get("/approvals/{approval_id}/edit-log")
async def get_edit_log(approval_id: str, current_user: Dict = Depends(require_doctor)):
    """Get the edit audit trail for a prescription (doctor only)"""
    try:
        db = DatabaseManager()
        logs = await db.get_edit_log(approval_id)
        return {"edit_log": logs, "count": len(logs)}
    except Exception as e:
        logger.error(f"Error getting edit log: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving edit log")


# ── Doctor Profile Update ───────────────────────────────────────

@app.post("/doctors/{doctor_id}/profile")
async def update_doctor_profile(
    doctor_id: str,
    nmc_number: str = Form(default=""),
    clinic_address: str = Form(default=""),
    current_user: Dict = Depends(require_doctor),
):
    """Update doctor's NMC number and clinic address (doctor only, own profile)"""
    if current_user["sub"] != doctor_id:
        raise HTTPException(status_code=403, detail="Can only update own profile")
    try:
        db = DatabaseManager()
        success = await db.update_doctor_profile(doctor_id, nmc_number=nmc_number, clinic_address=clinic_address)
        if success:
            return {"message": "Profile updated"}
        raise HTTPException(status_code=404, detail="Doctor not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating doctor profile: {e}")
        raise HTTPException(status_code=500, detail="Error updating profile")


# ── PDF Generation & Download ───────────────────────────────────

@app.post("/prescriptions/{approval_id}/generate-pdf")
async def generate_pdf(approval_id: str, current_user: Dict = Depends(require_doctor)):
    """Generate a prescription PDF with QR code (doctor only, after approval)"""
    try:
        db = DatabaseManager()
        approval = await db.get_approval_by_id(approval_id)
        if not approval:
            raise HTTPException(status_code=404, detail="Approval not found")

        if approval["status"] not in ("approved", "modified"):
            raise HTTPException(status_code=400, detail="Can only generate PDF for approved/modified prescriptions")

        doctor = await db.get_doctor_by_id(current_user["sub"]) or {}

        # Generate tokens
        verification_token = uuid.uuid4().hex
        download_token = uuid.uuid4().hex
        verification_url = f"{settings.verification_base_url}/{verification_token}"

        sig_hash = approval.get("signature_hash") or compute_signature_hash(
            current_user["sub"], doctor.get("nmc_number", ""), datetime.now().isoformat(), approval_id
        )

        # Generate PDF
        pdf_path = generate_prescription_pdf(
            approval_data=approval,
            doctor_data=doctor,
            verification_url=verification_url,
            signature_hash=sig_hash,
            is_provisional=False,
        )

        # Compute PDF content hash for tamper detection
        import hashlib as _hashlib
        with open(pdf_path, "rb") as _f:
            pdf_content_hash = _hashlib.sha256(_f.read()).hexdigest()

        # Save document record
        await db.save_prescription_document(
            approval_id=approval_id,
            pdf_path=pdf_path,
            qr_code_data=verification_url,
            verification_token=verification_token,
            download_token=download_token,
            signature_hash=sig_hash,
            pdf_content_hash=pdf_content_hash,
        )

        logger.info(f"PDF generated for {approval_id}, content hash: {pdf_content_hash[:16]}")
        return {
            "message": "PDF generated successfully",
            "approval_id": approval_id,
            "download_url": f"{settings.backend_base_url}/prescriptions/{approval_id}/download",
            "verification_url": verification_url,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating PDF: {e}")
        raise HTTPException(status_code=500, detail="Error generating PDF")


@app.get("/prescriptions/{approval_id}/download")
async def download_prescription_pdf(approval_id: str, current_user: Dict = Depends(get_current_user)):
    """Download the prescription PDF (authenticated user).
    If the file is missing (e.g. Railway ephemeral filesystem), regenerates from DB.
    """
    try:
        db = DatabaseManager()
        doc = await db.get_prescription_document(approval_id)

        pdf_path = doc.get("pdf_path") if doc else None

        # ── Serve cached file if it exists ───────────────────────────
        if pdf_path and os.path.exists(pdf_path):
            if doc.get("download_token"):
                await db.mark_download_used(doc["download_token"])
            return FileResponse(
                path=pdf_path,
                filename=f"prescription_{approval_id}.pdf",
                media_type="application/pdf",
            )

        # ── File missing — regenerate from approval + doctor data ────
        logger.info(f"PDF file missing for approval {approval_id}, regenerating from DB data")
        approval = await db.get_approval_by_id(approval_id)
        if not approval:
            raise HTTPException(status_code=404, detail="PDF not found. Please ask the doctor to generate it.")

        doctor_id = approval.get("approved_by") or approval.get("assigned_doctor")
        doctor_profile = {}
        if doctor_id:
            doctor_profile = await db.get_doctor_full_profile(doctor_id) or {}

        new_pdf_path = generate_prescription_pdf(
            approval_data=approval,
            doctor_data=doctor_profile,
            approval_id=approval_id,
            signature_hash=doc.get("signature_hash", "") if doc else "",
            verification_url=doc.get("verification_url", "") if doc else "",
        )

        # Persist updated path
        try:
            if doc:
                await db.save_prescription_document({**doc, "pdf_path": new_pdf_path})
        except Exception:
            pass

        if doc and doc.get("download_token"):
            await db.mark_download_used(doc["download_token"])

        return FileResponse(
            path=new_pdf_path,
            filename=f"prescription_{approval_id}.pdf",
            media_type="application/pdf",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading PDF: {e}")
        raise HTTPException(status_code=500, detail="Error downloading PDF")


@app.get("/prescriptions/{prescription_id}/download-pdf")
async def download_prescription_pdf_direct(
    prescription_id: str,
    current_user: Dict = Depends(get_current_user),
):
    """Download a prescription PDF by prescription ID (patient or doctor).
    If the file is missing (e.g. on a fresh server / Railway deploy), it is
    regenerated on-the-fly from the data stored in the database.
    """
    import json as _json
    try:
        db = DatabaseManager()

        # ── Fetch full prescription + items ──────────────────────────
        result = db.client.table("prescriptions").select(
            "*, prescription_items(*)"
        ).eq("id", prescription_id).limit(1).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="Prescription not found.")
        rx = result.data[0]

        pdf_path = rx.get("pdf_url")

        # ── Serve the cached file if it exists ───────────────────────
        if pdf_path and os.path.exists(pdf_path):
            return FileResponse(
                path=pdf_path,
                filename=f"prescription_{prescription_id}.pdf",
                media_type="application/pdf",
            )

        # ── File missing — regenerate from DB data ───────────────────
        logger.info(f"PDF file missing for {prescription_id}, regenerating from DB data")

        # Resolve doctor details
        doctor_id = rx.get("prescribed_by_doctor_id")
        doctor_profile = {}
        if doctor_id:
            doctor_profile = await db.get_doctor_full_profile(doctor_id) or {}

        doctor_name    = doctor_profile.get("name", "Doctor")
        doctor_specialty = doctor_profile.get("specialty", "General Medicine")
        nmc_number     = doctor_profile.get("nmc_number", "")

        # Resolve patient details via consultation if available
        patient_name   = "Patient"
        patient_age    = 0
        patient_gender = ""
        consultation_id = rx.get("consultation_id")
        if consultation_id:
            consultation = await db.get_consultation_with_patient(consultation_id)
            if consultation:
                patient_name   = consultation.get("patient_name", "Patient")
                patient_age    = consultation.get("patient_age") or 0
                patient_gender = consultation.get("patient_gender", "")
                doctor_specialty = consultation.get("specialty") or doctor_specialty

        # Parse JSONB fields (may be stored as JSON strings)
        def _parse_list(val):
            if isinstance(val, list):
                return val
            if isinstance(val, str):
                try:
                    return _json.loads(val)
                except Exception:
                    return [val]
            return []

        general_instructions   = _parse_list(rx.get("general_instructions", []))
        dietary_advice         = _parse_list(rx.get("dietary_advice", []))
        warning_signs          = _parse_list(rx.get("warning_signs", []))
        follow_up_instructions = rx.get("follow_up_instructions", "")

        # Build medication list from prescription_items
        raw_items = rx.get("prescription_items") or []
        med_list = [
            {
                "name":         item.get("medicine_name", ""),
                "generic_name": item.get("generic_name", ""),
                "dosage":       item.get("dosage", ""),
                "frequency":    item.get("frequency", ""),
                "duration":     item.get("duration", ""),
                "instructions": item.get("instructions", ""),
                "before_food":  item.get("before_food"),
            }
            for item in raw_items
        ]

        new_pdf_path = generate_prescription_pdf(
            patient_name           = patient_name,
            patient_age            = patient_age,
            patient_gender         = patient_gender,
            doctor_name            = doctor_name,
            doctor_specialty       = doctor_specialty,
            nmc_number             = nmc_number,
            medications            = med_list,
            general_instructions   = general_instructions,
            dietary_advice         = dietary_advice,
            warning_signs          = warning_signs,
            follow_up_instructions = follow_up_instructions,
            approval_id            = prescription_id,
            signature_hash         = rx.get("digital_signature", ""),
        )

        # Persist the newly generated path so future downloads are instant
        try:
            db.client.table("prescriptions").update({"pdf_url": new_pdf_path}).eq("id", prescription_id).execute()
        except Exception:
            pass  # Non-fatal — file is still served this request

        return FileResponse(
            path=new_pdf_path,
            filename=f"prescription_{prescription_id}.pdf",
            media_type="application/pdf",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"download_prescription_pdf_direct error: {e}")
        raise HTTPException(status_code=500, detail="Error downloading PDF")


# ── QR Code Verification (Public) ───────────────────────────────

@app.get("/verify/{verification_token}")
async def verify_prescription(verification_token: str):
    """Public endpoint — verify a prescription via QR code scan"""
    try:
        db = DatabaseManager()
        doc = await db.get_document_by_verification_token(verification_token)
        if not doc:
            raise HTTPException(status_code=404, detail="Prescription not found or invalid verification code")

        # Verify PDF integrity if file exists
        pdf_integrity = "not_checked"
        stored_pdf_hash = doc.get("pdf_content_hash")
        pdf_path = doc.get("pdf_path")
        if stored_pdf_hash and pdf_path and os.path.exists(pdf_path):
            import hashlib as _hashlib
            with open(pdf_path, "rb") as _f:
                current_hash = _hashlib.sha256(_f.read()).hexdigest()
            pdf_integrity = "valid" if current_hash == stored_pdf_hash else "tampered"

        return {
            "verified": True,
            "approval_id": doc.get("approval_id"),
            "status": doc.get("status"),
            "doctor_name": doc.get("doctor_name"),
            "license_number": doc.get("license_number"),
            "nmc_number": doc.get("nmc_number"),
            "patient_name": doc.get("patient_name"),
            "approved_at": doc.get("approved_at"),
            "signature_hash": doc.get("signature_hash"),
            "pdf_integrity": pdf_integrity,
            "generated_at": doc.get("generated_at"),
            "message": "This prescription was digitally verified via Medivora.",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error verifying prescription: {e}")
        raise HTTPException(status_code=500, detail="Verification error")


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again."}
    )

# ── Video Consultation Endpoints ─────────────────────────────────────────────

@app.post("/consultation/request")
async def request_consultation(
    specialty:    str  = Form(default="general_medicine"),
    patient_note: str  = Form(default=""),
    current_user: Dict = Depends(get_current_user),
):
    """Patient requests a video consultation — creates a Daily room."""
    db = DatabaseManager()
    session_id = f"cslt_{uuid.uuid4().hex[:12]}"

    # Create Daily room
    if not DAILY_API_KEY:
        raise HTTPException(status_code=503, detail="Video service not configured.")
    try:
        room_name, room_url = await create_daily_room(f"medivora-{specialty}-{session_id}")
    except Exception as e:
        logger.error(f"Daily room creation failed: {e}")
        raise HTTPException(status_code=502, detail="Could not create video room.")

    session = await db.create_consultation({
        "id":              session_id,
        "patient_id":      current_user["sub"],
        "daily_meeting_id": room_name,
        "specialty":       specialty,
        "patient_note":    patient_note,
    })
    if not session:
        raise HTTPException(status_code=500, detail="Failed to save consultation session.")

    try:
        await db.create_user_notification(
            user_id=current_user["sub"],
            notification_type="consultation",
            title="Consultation Booked",
            message=f"Your consultation room is ready. A doctor will join you shortly. Specialty: {specialty or 'General Medicine'}.",
            priority=2,
        )
    except Exception:
        pass

    return {
        "session_id": session_id,
        "status":     "waiting",
        "message":    "Consultation room created. A doctor will join shortly.",
    }


@app.get("/consultation/my")
async def my_consultations(current_user: Dict = Depends(get_current_user)):
    """Patient: list own consultation history."""
    db = DatabaseManager()
    phone = current_user.get("phone", "")
    sessions = await db.get_patient_consultations(current_user["sub"], phone=phone)
    return {"sessions": sessions}


@app.post("/consultation/{consultation_id}/confirm-payment")
async def confirm_consultation_payment(
    consultation_id: str,
    current_user: Dict = Depends(get_current_user),
):
    """Patient: mark a consultation as payment-confirmed (status → waiting).
    No real payment gateway — used until Razorpay is integrated."""
    db = DatabaseManager()
    session = await db.get_consultation_by_id(consultation_id)
    if not session:
        raise HTTPException(status_code=404, detail="Consultation not found.")
    if session.get("status") not in ("requested",):
        raise HTTPException(status_code=400, detail=f"Consultation is already '{session.get('status')}'.")

    ok = await db.update_consultation(consultation_id, {"status": "scheduled"})
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to update consultation status.")

    # Notify patient
    try:
        await db.create_user_notification(
            user_id           = current_user["sub"],
            notification_type = "consultation",
            title             = "Booking Confirmed",
            message           = "Your payment was received. A doctor will be assigned shortly.",
        )
    except Exception:
        pass

    return {"success": True, "status": "scheduled"}


@app.get("/consultation/pending")
async def pending_consultations(current_user: Dict = Depends(require_doctor)):
    """Doctor: list consultations waiting for a doctor."""
    db = DatabaseManager()
    sessions = await db.get_pending_consultations()
    return {"sessions": sessions}


@app.post("/consultation/{session_id}/join")
async def doctor_join_consultation(
    session_id:   str,
    current_user: Dict = Depends(require_doctor),
):
    """Doctor joins a waiting consultation — returns the doctor room token."""
    db = DatabaseManager()
    session = await db.get_consultation_by_id(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Consultation session not found.")
    if session["status"] not in ("requested", "scheduled", "ongoing", "active"):
        raise HTTPException(status_code=400, detail=f"Session is already {session['status']}.")

    # Resolve profile_id → doctors.id (same as schedule endpoint) so the FK stays correct
    resolved_doctor_id = await db.resolve_doctor_id(current_user["sub"])
    await db.update_consultation(session_id, {
        "doctor_id":  resolved_doctor_id,
        "status":     "ongoing",
        "started_at": datetime.now().isoformat(),
    })

    # Notify patient that doctor has joined
    try:
        patient_id = session.get("patient_id")
        if patient_id:
            doctor_name = current_user.get("name", "A doctor")
            await db.create_user_notification(
                user_id=patient_id,
                notification_type="consultation",
                title="Doctor Has Joined",
                message=f"Dr. {doctor_name} has joined your consultation. Please join the video call now.",
                priority=1,
            )
    except Exception:
        pass

    # Create Daily token for doctor — lazily create room if needed
    daily_room = session.get("daily_meeting_id")
    if not daily_room:
        if not DAILY_API_KEY:
            raise HTTPException(status_code=503, detail="Video service not configured.")
        try:
            daily_room, _ = await create_daily_room(f"medivora-{session.get('specialty','general')}-{session_id}")
            await db.update_consultation(session_id, {"daily_meeting_id": daily_room})
        except Exception as e:
            logger.error(f"Daily room creation failed in doctor join: {e}")
            raise HTTPException(status_code=502, detail="Could not create video room.")

    display_name = current_user.get("name") or "Doctor"
    try:
        auth_token = await create_daily_token(room_name=daily_room, is_owner=False, user_name=display_name)
    except Exception as e:
        logger.error(f"Daily token creation failed in doctor join: {e}")
        raise HTTPException(status_code=502, detail="Could not connect to video room.")

    return {
        "session_id":   session_id,
        "room_url":     f"https://medivora.daily.co/{daily_room}",
        "auth_token":   auth_token,
        "patient_note": session.get("patient_note", ""),
        "specialty":    session.get("specialty", "general_medicine"),
    }


@app.post("/consultation/{session_id}/patient-join")
async def patient_join_consultation(
    session_id:   str,
    current_user: Dict = Depends(get_current_user),
):
    """Record the first time a patient joins a consultation call."""
    db = DatabaseManager()
    session = await db.get_consultation_by_id(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Consultation not found.")

    # Only record once
    if not session.get("patient_joined_at"):
        await db.update_consultation(session_id, {
            "patient_joined_at": datetime.now().isoformat(),
        })

    return {"ok": True}


@app.post("/consultation/{session_id}/end")
async def end_consultation(
    session_id:   str,
    current_user: Dict = Depends(get_current_user),
):
    """Called when a participant leaves the Jitsi room. Does NOT change status —
    the doctor must explicitly call /complete to mark the consultation done."""
    db = DatabaseManager()
    session = await db.get_consultation_by_id(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Consultation session not found.")
    return {"status": session.get("status"), "message": "Call left."}


@app.post("/consultation/{session_id}/complete")
async def complete_consultation(
    session_id:   str,
    current_user: Dict = Depends(require_doctor),
):
    """Doctor explicitly marks the consultation as completed."""
    db = DatabaseManager()
    session = await db.get_consultation_by_id(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Consultation session not found.")

    await db.update_consultation(session_id, {
        "status":       "completed",
        "completed_at": datetime.now().isoformat(),
    })

    return {"status": "completed", "message": "Consultation completed."}


@app.get("/consultation/doctor")
async def doctor_consultations(current_user: Dict = Depends(require_doctor)):
    """Doctor: list all consultations assigned to this doctor."""
    db = DatabaseManager()
    # current_user["sub"] is the Supabase auth UID (profile_id).
    # consultations.doctor_id stores doctors.id, so resolve it first.
    doctor_id = await db.resolve_doctor_id(current_user["sub"])
    sessions = await db.get_doctor_consultations(doctor_id)
    # Also include unassigned pending requests so the doctor can pick them up
    pending = await db.get_pending_consultations()
    # Merge, de-duplicate by id
    seen = {s["id"] for s in sessions}
    for s in pending:
        if s["id"] not in seen:
            sessions.append(s)
            seen.add(s["id"])
    total_income = await db.get_doctor_total_income(doctor_id)
    return {"sessions": sessions, "total_income": total_income}


@app.patch("/consultation/{session_id}/slot")
async def patient_set_slot(
    session_id:        str,
    scheduled_at:      str = Form(...),
    consultation_type: str = Form(default=""),
    current_user: Dict = Depends(get_current_user),
):
    """Patient selects their appointment slot after payment is confirmed."""
    db = DatabaseManager()
    session = await db.get_consultation_by_id(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Consultation not found.")

    # Ensure this consultation belongs to the requesting patient.
    # profiles.id == Supabase auth UID after migration, so direct lookup suffices.
    user_id = current_user["sub"]
    stored_patient_id = session.get("patient_id", "")
    resolved_id = user_id
    try:
        p_row = db.client.table("patients").select("id").eq("profile_id", user_id).limit(1).execute()
        if p_row.data:
            resolved_id = p_row.data[0]["id"]
    except Exception:
        pass
    if stored_patient_id not in (user_id, resolved_id):
        raise HTTPException(status_code=403, detail="Not authorized.")

    update = {"scheduled_at": scheduled_at}
    if consultation_type in ("video", "in_person"):
        update["consultation_type"] = consultation_type

    await db.update_consultation(session_id, update)

    try:
        from datetime import datetime as _dt
        formatted = _dt.fromisoformat(scheduled_at).strftime("%d %b %Y, %I:%M %p")
    except Exception:
        formatted = scheduled_at

    logger.info(f"Patient {user_id} set slot for consultation {session_id}: {formatted} ({consultation_type or 'type unchanged'})")
    return {"message": "Slot saved", "session_id": session_id, "scheduled_at": scheduled_at}


@app.patch("/consultation/{session_id}/schedule")
async def schedule_consultation(
    session_id:   str,
    scheduled_at: str = Form(...),
    note:         str = Form(default=""),
    current_user: Dict = Depends(require_doctor),
):
    """Doctor approves and schedules a consultation with a confirmed time."""
    db = DatabaseManager()
    session = await db.get_consultation_by_id(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Consultation not found.")

    # Resolve profile_id → doctors.id so the correct FK is stored
    doctor_id = await db.resolve_doctor_id(current_user["sub"])

    await db.update_consultation(session_id, {
        "doctor_id":    doctor_id,
        "status":       "scheduled",
        "scheduled_at": scheduled_at,
        "summary":      note or session.get("summary", ""),
    })

    # Notify patient about the (re)schedule
    patient_id = session.get("patient_id")
    if patient_id:
        from datetime import datetime as _dt
        try:
            formatted = _dt.fromisoformat(scheduled_at).strftime("%d %b %Y, %I:%M %p")
        except Exception:
            formatted = scheduled_at

        # Best-effort: resolve doctor display name
        doctor_display = "Your doctor"
        try:
            doctor_profile = await db.get_doctor_full_profile(current_user["sub"]) or {}
            full_name = (doctor_profile.get("full_name") or "").strip()
            if full_name:
                doctor_display = f"Dr. {full_name}"
        except Exception:
            pass

        # Send notification
        try:
            await db.create_user_notification(
                user_id=patient_id,
                notification_type="consultation",
                title="Doctor Assigned",
                message=f"{doctor_display} has been assigned to your consultation, confirmed for {formatted}.",
                priority=2,
            )
        except Exception:
            pass

        # Update stale "A doctor will be assigned shortly" notifications
        try:
            db.client.table("notifications") \
                .update({"message": f"{doctor_display} has been assigned to your consultation."}) \
                .eq("user_id", patient_id) \
                .ilike("message", "%doctor will be assigned shortly%") \
                .execute()
        except Exception:
            pass

    return {"message": "Consultation scheduled", "session_id": session_id, "scheduled_at": scheduled_at}


@app.patch("/consultation/{session_id}/reject")
async def reject_consultation(
    session_id:   str,
    reason:       str = Form(default=""),
    current_user: Dict = Depends(require_doctor),
):
    """Doctor rejects / cancels a consultation request."""
    db = DatabaseManager()
    session = await db.get_consultation_by_id(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Consultation not found.")
    await db.update_consultation(session_id, {
        "doctor_id":   current_user["sub"],
        "status":      "cancelled",
        "summary":     reason or "Rejected by doctor",
    })
    return {"message": "Consultation rejected", "session_id": session_id}


@app.get("/consultation/{session_id}/call-details")
async def get_call_details(
    session_id:   str,
    current_user: Dict = Depends(get_current_user),
):
    """Return a Daily token and room URL for the participant to join the video call."""
    db = DatabaseManager()
    session = await db.get_consultation_by_id(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Consultation not found.")

    role    = current_user.get("role", "patient")
    user_id = current_user.get("sub")

    # Authorisation: patient must own the consultation
    if role != "doctor":
        stored_patient_id = session.get("patient_id", "")
        try:
            p_row = db.client.table("patients").select("id").eq("profile_id", user_id).limit(1).execute()
            resolved_id = p_row.data[0]["id"] if p_row.data else user_id
        except Exception:
            resolved_id = user_id
        if stored_patient_id not in (user_id, resolved_id):
            raise HTTPException(status_code=403, detail="Not authorised.")

    room_name = session.get("daily_meeting_id")  # stored as room_name
    if not room_name:
        # Lazily create a Daily room if one wasn't created at booking time
        if not DAILY_API_KEY:
            raise HTTPException(status_code=503, detail="Video service not configured.")
        try:
            specialty = session.get("specialty", "general")
            room_name, _ = await create_daily_room(f"medivora-{specialty}-{session_id}")
            await db.update_consultation(session_id, {"daily_meeting_id": room_name})
        except Exception as e:
            logger.error(f"Daily lazy room creation failed: {e}")
            raise HTTPException(status_code=502, detail="Could not create video room.")

    display_name = current_user.get("name") or ("Doctor" if role == "doctor" else "Patient")
    is_owner = False  # Daily prebuilt UI works better without owner mode

    try:
        auth_token = await create_daily_token(
            room_name=room_name,
            is_owner=is_owner,
            user_name=display_name,
        )
    except Exception as e:
        logger.error(f"Daily token creation failed: {e}")
        raise HTTPException(status_code=502, detail="Could not connect to video room.")

    return {
        "session_id":   session_id,
        "room_url":     f"https://medivora.daily.co/{room_name}",
        "auth_token":   auth_token,
        "display_name": display_name,
        "role":         role,
        "patient_note": session.get("patient_note", ""),
        "specialty":    session.get("specialty", ""),
        "status":       session.get("status", ""),
    }


# ── Consultation-based Prescription Generation ───────────────────

@app.post("/consultation/{consultation_id}/generate-prescription")
async def generate_consultation_prescription(
    consultation_id: str,
    current_user: Dict = Depends(require_doctor),
):
    """Doctor triggers AI to generate a prescription draft based on the consultation."""
    db = DatabaseManager()
    consultation = await db.get_consultation_with_patient(consultation_id)
    if not consultation:
        raise HTTPException(status_code=404, detail="Consultation not found.")

    if consultation.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Prescription can only be generated for completed consultations.")

    # Build context for AI
    patient_note    = consultation.get("patient_note") or "No note provided."
    specialty       = consultation.get("specialty") or "General Medicine"
    patient_age     = consultation.get("patient_age") or "Unknown"
    patient_gender  = consultation.get("patient_gender") or "Unknown"
    allergies       = consultation.get("patient_allergies") or []
    chronic         = consultation.get("patient_chronic_conditions") or []
    med_history     = consultation.get("patient_medical_history") or []

    prompt = f"""You are a senior doctor generating a prescription for an Indian patient.

Patient Details:
- Age: {patient_age}
- Gender: {patient_gender}
- Allergies: {', '.join(allergies) if allergies else 'None known'}
- Chronic Conditions: {', '.join(str(c) for c in chronic) if chronic else 'None'}
- Medical History: {', '.join(str(h) for h in med_history) if med_history else 'None'}
- Specialty: {specialty}
- Patient's complaint / note: {patient_note}

Generate a complete prescription as a valid JSON object with this exact structure:
{{
  "diagnosis": "<brief clinical impression>",
  "medicines": [
    {{
      "medicine_name": "<brand name>",
      "generic_name": "<generic / INN name>",
      "dosage": "<e.g. 500mg>",
      "frequency": "<e.g. Twice daily>",
      "duration": "<e.g. 5 days>",
      "instructions": "<e.g. After food>",
      "before_food": false
    }}
  ],
  "suggested_tests": [
    {{
      "test_name": "<e.g. Complete Blood Count>",
      "reason": "<brief clinical reason>",
      "priority": "<routine|urgent>"
    }}
  ],
  "general_instructions": ["<instruction 1>", "<instruction 2>"],
  "dietary_advice": ["<advice 1>", "<advice 2>"],
  "warning_signs": ["<warning 1>", "<warning 2>"],
  "follow_up_instructions": "<follow-up advice>"
}}

Rules:
- Only prescribe drugs that are safe for the patient (respect allergies and conditions).
- Prefer generic names; include brand name for clarity.
- No Schedule X / narcotic drugs.
- Include suggested_tests only when clinically indicated; leave as empty array [] if none needed.
- Return ONLY the raw JSON object — no markdown, no extra text.
"""

    try:
        import os as _os
        from google import genai as _genai

        _client = _genai.Client(api_key=_os.getenv("GOOGLE_API_KEY"))
        response = _client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        raw = response.text.strip()
        # Strip any markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw
            raw = raw.rsplit("```", 1)[0].strip()

        import json as _json
        prescription_draft = _json.loads(raw)
    except Exception as e:
        logger.error(f"AI prescription generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"AI generation failed: {str(e)}")

    return {
        "consultation_id": consultation_id,
        "patient_name": consultation.get("patient_name", "Patient"),
        "patient_age": patient_age,
        "patient_gender": patient_gender,
        "specialty": specialty,
        "draft": prescription_draft,
    }


@app.post("/consultation/{consultation_id}/submit-prescription")
async def submit_consultation_prescription(
    consultation_id: str,
    payload: Dict,
    current_user: Dict = Depends(require_doctor),
):
    """Doctor submits (possibly edited) AI prescription — saves, signs, notifies patient."""
    db = DatabaseManager()
    consultation = await db.get_consultation_with_patient(consultation_id)
    if not consultation:
        raise HTTPException(status_code=404, detail="Consultation not found.")

    # Prevent duplicate submission
    existing = await db.get_prescription_by_consultation(consultation_id)
    if existing:
        raise HTTPException(status_code=409, detail="Prescription already submitted for this consultation.")

    doctor_id = await db.resolve_doctor_id(current_user["sub"])
    patient_db_id = consultation.get("patient_db_id")
    if not patient_db_id:
        raise HTTPException(status_code=400, detail="Patient record not found.")

    # Generate prescription number
    import uuid as _uuid
    import random as _random
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td

    rx_number = f"RX-{_dt.now(_tz.utc).strftime('%Y%m%d')}-{_random.randint(100, 999)}"
    now       = _dt.now(_tz.utc)

    # Compute digital signature
    timestamp = now.isoformat()
    sig_hash  = compute_signature_hash(doctor_id, "", timestamp, consultation_id)

    rx_data = {
        "patient_id":             patient_db_id,
        "consultation_id":        consultation_id,
        "prescribed_by_doctor_id": doctor_id,
        "prescription_number":    rx_number,
        "status":                 "approved",
        "general_instructions":   payload.get("general_instructions", []),
        "dietary_advice":         payload.get("dietary_advice", []),
        "warning_signs":          payload.get("warning_signs", []),
        "follow_up_instructions": payload.get("follow_up_instructions", ""),
        "validity_days":          30,
        "prescribed_at":          now.isoformat(),
        "approved_at":            now.isoformat(),
        "expires_at":             (now + _td(days=30)).isoformat(),
        "digital_signature":      sig_hash,
    }

    medicines = payload.get("medicines", [])
    lab_tests = payload.get("lab_tests", [])
    items = []
    for med in medicines:
        items.append({
            "medicine_name":  med.get("medicine_name", ""),
            "generic_name":   med.get("generic_name", ""),
            "dosage":         med.get("dosage", ""),
            "frequency":      med.get("frequency", ""),
            "duration":       med.get("duration", ""),
            "instructions":   med.get("instructions", ""),
            "before_food":    med.get("before_food", False),
            "item_type":      "medicine",
            "contraindications": [],
            "side_effects":   [],
            "is_blacklisted": False,
        })
    for test in lab_tests:
        items.append({
            "medicine_name":  test.get("test_name", ""),
            "generic_name":   test.get("reason", ""),
            "dosage":         test.get("priority", "routine"),
            "frequency":      "",
            "duration":       "",
            "instructions":   test.get("reason", ""),
            "before_food":    False,
            "item_type":      "lab_test",
            "contraindications": [],
            "side_effects":   [],
            "is_blacklisted": False,
        })

    try:
        saved_rx = await db.create_prescription_with_items(rx_data, items)
    except Exception as e:
        logger.error(f"submit_consultation_prescription DB error: {e}")
        raise HTTPException(status_code=500, detail="Failed to save prescription.")

    rx_id = saved_rx.get("id")

    # Auto-generate PDF and store path
    try:
        doctor_profile = await db.get_doctor_full_profile(current_user["sub"]) or {}
        doctor_name  = current_user.get("name", "Doctor")
        nmc_number   = doctor_profile.get("nmc_number", "")
        specialty    = consultation.get("specialty", "General Medicine")
        med_list = [
            {
                "name":         m.get("medicine_name", ""),
                "generic_name": m.get("generic_name", ""),
                "dosage":       m.get("dosage", ""),
                "frequency":    m.get("frequency", ""),
                "duration":     m.get("duration", ""),
                "instructions": m.get("instructions", ""),
                "before_food":  m.get("before_food"),
            }
            for m in medicines
        ]
        pdf_path = generate_prescription_pdf(
            patient_name           = consultation.get("patient_name", "Patient"),
            patient_age            = consultation.get("patient_age") or 0,
            patient_gender         = consultation.get("patient_gender", ""),
            doctor_name            = doctor_name,
            doctor_specialty       = specialty,
            nmc_number             = nmc_number,
            diagnosis              = payload.get("diagnosis", ""),
            medications            = med_list,
            general_instructions   = payload.get("general_instructions", []),
            dietary_advice         = payload.get("dietary_advice", []),
            warning_signs          = payload.get("warning_signs", []),
            follow_up_instructions = payload.get("follow_up_instructions", ""),
            approval_id            = rx_id,
            signature_hash         = sig_hash,
        )
        # Store the file path in prescriptions.pdf_url
        db.client.table("prescriptions").update({"pdf_url": pdf_path}).eq("id", rx_id).execute()
    except Exception as e:
        logger.warning(f"PDF generation failed (non-fatal): {e}")
        pdf_path = None

    # Notify patient
    try:
        profile_id = consultation.get("patient_profile_id")
        if profile_id:
            doctor_name = current_user.get("name", "Your doctor")
            await db.create_user_notification(
                user_id=profile_id,
                notification_type="prescription",
                title="New Prescription Available",
                message=f"Dr. {doctor_name} has issued a prescription for your recent consultation. View it in the Prescriptions tab.",
                priority=2,
            )
    except Exception:
        pass  # Notification failure must not block the response

    logger.info(f"Prescription submitted for consultation {consultation_id}, rx: {rx_number}")
    return {
        "message": "Prescription saved and signed",
        "prescription_id": rx_id,
        "prescription_number": rx_number,
        "signature_hash": sig_hash,
        "download_url": (
            f"{settings.backend_base_url}/prescriptions/{rx_id}/download-pdf"
            if pdf_path else None
        ),
    }


@app.get("/doctor/prescriptions")
async def get_doctor_prescriptions(current_user: Dict = Depends(require_doctor)):
    """Return all prescriptions submitted by this doctor (consultation flow)."""
    try:
        db = DatabaseManager()
        doctor_id = await db.resolve_doctor_id(current_user["sub"])
        result = db.client.table("prescriptions").select(
            "*, prescription_items(*)"
        ).eq("prescribed_by_doctor_id", doctor_id).order("created_at", desc=True).execute()
        prescriptions = result.data or []

        # Enrich with patient name via consultation
        import json as _json
        for rx in prescriptions:
            # Parse JSONB list fields stored as strings
            for field in ("general_instructions", "dietary_advice", "warning_signs"):
                val = rx.get(field)
                if isinstance(val, str):
                    try:
                        rx[field] = _json.loads(val)
                    except Exception:
                        rx[field] = [val]

            patient_name = "Patient"
            consultation_id = rx.get("consultation_id")
            if consultation_id:
                try:
                    c = await db.get_consultation_with_patient(consultation_id)
                    if c:
                        patient_name = c.get("patient_name", "Patient")
                except Exception:
                    pass
            rx["patient_name"] = patient_name

        return {"prescriptions": prescriptions, "count": len(prescriptions)}
    except Exception as e:
        logger.error(f"get_doctor_prescriptions error: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving prescriptions")


@app.get("/doctor/profile")
async def get_doctor_profile(current_user: Dict = Depends(require_doctor)):
    """Return the logged-in doctor's full profile."""
    db = DatabaseManager()
    doctor = await db.get_doctor_full_profile(current_user["sub"])
    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor profile not found.")
    return doctor


@app.put("/doctor/profile")
async def update_doctor_profile_full(
    full_name:        str = Form(default=""),
    email:            str = Form(default=""),
    clinic_name:      str = Form(default=""),
    clinic_address:   str = Form(default=""),
    clinic_phone:     str = Form(default=""),
    consultation_fee: str = Form(default=""),
    available_status: str = Form(default=""),
    experience_years: str = Form(default=""),
    nmc_number:       str = Form(default=""),
    specialties:      str = Form(default=""),
    available_slots:  str = Form(default=""),
    current_user: Dict = Depends(require_doctor),
):
    """Update the logged-in doctor's full profile."""
    db = DatabaseManager()
    updates = {}
    if clinic_name:      updates["clinic_name"]      = clinic_name
    if clinic_address:   updates["clinic_address"]   = clinic_address
    if clinic_phone:     updates["clinic_phone"]      = clinic_phone
    if nmc_number:       updates["nmc_number"]        = nmc_number
    if available_status and available_status in ("available", "busy", "offline", "on_leave"):
        updates["available_status"] = available_status
    if consultation_fee:
        try: updates["consultation_fee"] = float(consultation_fee)
        except ValueError: pass
    if experience_years:
        try: updates["experience_years"] = int(experience_years)
        except ValueError: pass
    if specialties:
        updates["specialties"] = [s.strip() for s in specialties.split(",") if s.strip()]
    if available_slots is not None and available_slots != "":
        try:
            import json as _json
            updates["available_slots"] = _json.loads(available_slots)
        except Exception:
            pass
    success = await db.update_doctor_profile(current_user["sub"], **updates)
    if not success:
        raise HTTPException(status_code=404, detail="Doctor profile not found.")
    # Update first_name / last_name / email in profiles table
    profile_updates = {}
    if full_name and full_name.strip():
        fn_parts = full_name.strip().split(" ", 1)
        profile_updates["first_name"] = fn_parts[0]
        profile_updates["last_name"]  = fn_parts[1] if len(fn_parts) > 1 else ""
    if email and email.strip():
        profile_updates["email"] = email.strip()
    if profile_updates:
        try:
            db.client.table("profiles").update(profile_updates).eq("id", current_user["sub"]).execute()
        except Exception:
            pass
    return {"message": "Profile updated successfully"}


# ─────────────────────────────────────────────────────────────────────────────
# RAZORPAY PAYMENT INTEGRATION
# Requires: pip install razorpay
# .env vars: RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET
# ─────────────────────────────────────────────────────────────────────────────

class PaymentOrderRequest(BaseModel):
    amount:      int    # in paise (₹499 → 49900)
    currency:    str    = "INR"
    doctor_id:   str
    doctor_name: str
    slot:        str
    note:        str    = ""

class PaymentVerifyRequest(BaseModel):
    razorpay_order_id:   str
    razorpay_payment_id: str
    razorpay_signature:  str
    doctor_id:           str
    specialty:           str
    note:                str = ""
    session_id:          str = ""  # if set, update existing consultation instead of creating new
    scheduled_at:        str = ""
    consultation_type:   str = "in_person"


@app.post("/payments/create-order")
@limiter.limit("10/minute")
async def create_payment_order(
    request: Request,
    body: PaymentOrderRequest,
    current_user: Dict = Depends(get_current_user),
):
    """
    Create a Razorpay order.
    Returns order_id, amount, currency, and key_id for the frontend checkout.
    """
    key_id     = os.getenv("RAZORPAY_KEY_ID", "")
    key_secret = os.getenv("RAZORPAY_KEY_SECRET", "")

    # Demo mode — no Razorpay keys configured
    if not key_id or not key_secret:
        demo_order_id = f"order_DEMO_{uuid.uuid4().hex[:12].upper()}"
        logger.info(f"[DEMO] Payment order created: {demo_order_id}")
        return {
            "order_id":    demo_order_id,
            "amount":      body.amount,
            "currency":    body.currency,
            "key_id":      "demo",
            "demo":        True,
            "message":     "Demo mode — Razorpay keys not configured",
        }

    try:
        import razorpay
        client = razorpay.Client(auth=(key_id, key_secret))
        order = client.order.create({
            "amount":   body.amount,
            "currency": body.currency,
            "receipt":  f"receipt_{uuid.uuid4().hex[:8]}",
            "notes": {
                "doctor_id":   body.doctor_id,
                "doctor_name": body.doctor_name,
                "slot":        body.slot,
                "patient_id":  current_user["sub"],
            },
        })
        logger.info(f"Razorpay order created: {order['id']} for patient {current_user['sub']}")
        return {
            "order_id": order["id"],
            "amount":   order["amount"],
            "currency": order["currency"],
            "key_id":   key_id,
            "demo":     False,
        }
    except ImportError:
        raise HTTPException(status_code=500, detail="Razorpay SDK not installed. Run: pip install razorpay")
    except Exception as e:
        logger.error(f"Razorpay order creation failed: {e}")
        raise HTTPException(status_code=500, detail="Payment gateway error. Please try again.")


@app.post("/payment/dev-confirm")
async def dev_confirm_booking(
    request: Request,
    current_user: Dict = Depends(get_current_user),
):
    """Dev-only endpoint: create/confirm a consultation without payment.
    Only works when DEBUG=true in .env."""
    if not settings.DEBUG:
        raise HTTPException(status_code=403, detail="Not available in production.")
    body = await request.json()
    doctor_id        = body.get("doctor_id", "")
    specialty        = body.get("specialty", "general_medicine")
    patient_note     = body.get("patient_note", "")
    scheduled_at     = body.get("scheduled_at", "")
    consultation_type = body.get("consultation_type", "in_person")

    session_id = str(uuid.uuid4())
    db = DatabaseManager()
    await db.create_consultation({
        "id":                session_id,
        "patient_id":        current_user["sub"],
        "doctor_id":         doctor_id or None,
        "specialty":         specialty,
        "patient_note":      patient_note,
        "status":            "scheduled",
        "scheduled_at":      scheduled_at or None,
        "payment_id":        "dev_skip",
        "consultation_type": consultation_type,
        "created_at":        datetime.now().isoformat(),
    })

    # Notify patient
    try:
        specialty_label = specialty.replace("_", " ").title() if specialty else "General Medicine"
        doctor_display = "A doctor"
        if doctor_id:
            try:
                doc = await db.get_doctor_full_profile(doctor_id) or {}
                full_name = (doc.get("full_name") or "").strip()
                if full_name:
                    doctor_display = f"Dr. {full_name}"
            except Exception:
                pass
        await db.create_user_notification(
            user_id=current_user["sub"],
            notification_type="consultation",
            title="Consultation Booked",
            message=f"Your {specialty_label} consultation is confirmed. {doctor_display} has been assigned.",
            priority=2,
        )
    except Exception:
        pass

    return {"status": "ok", "session_id": session_id}


@app.post("/payments/verify")
@limiter.limit("10/minute")
async def verify_payment(
    request: Request,
    body: PaymentVerifyRequest,
    current_user: Dict = Depends(get_current_user),
):
    """
    Verify Razorpay payment signature (HMAC-SHA256).
    On success, creates a consultation session for the patient.
    """
    import hmac
    import hashlib

    key_secret = os.getenv("RAZORPAY_KEY_SECRET", "")

    # Demo mode — skip signature check, go straight to consultation
    if not key_secret or body.razorpay_order_id.startswith("order_DEMO_"):
        logger.info(f"[DEMO] Payment verified for patient {current_user['sub']}")
    else:
        # Verify HMAC signature
        payload   = f"{body.razorpay_order_id}|{body.razorpay_payment_id}"
        expected  = hmac.new(
            key_secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(expected, body.razorpay_signature):
            logger.warning(f"Invalid Razorpay signature for order {body.razorpay_order_id}")
            raise HTTPException(status_code=400, detail="Payment verification failed. Signature mismatch.")

    # Payment verified — update existing or create new consultation session
    try:
        db = DatabaseManager()
        if body.session_id:
            # Update the existing consultation's status to "confirmed" (payment received)
            session_id = body.session_id
            await db.update_consultation(session_id, {
                "status":           "scheduled",
                "payment_id":       body.razorpay_payment_id,
                "payment_order_id": body.razorpay_order_id,
            })
            logger.info(f"Consultation confirmed after payment: {session_id}")
        else:
            # Legacy: create a new consultation session
            session_id = str(uuid.uuid4())
            await db.create_consultation({
                "id":                session_id,
                "patient_id":        current_user["sub"],
                "doctor_id":         body.doctor_id or None,
                "specialty":         body.specialty or "general_medicine",
                "patient_note":      body.note,
                "status":            "scheduled",
                "payment_id":        body.razorpay_payment_id,
                "payment_order_id":  body.razorpay_order_id,
                "scheduled_at":      body.scheduled_at or None,
                "consultation_type": body.consultation_type or "in_person",
                "created_at":        datetime.now().isoformat(),
            })
            logger.info(f"Consultation created after payment: {session_id}")

        # Notify patient
        try:
            specialty_label = (body.specialty or "General Medicine").replace("_", " ").title()
            await db.create_user_notification(
                user_id=current_user["sub"],
                notification_type="consultation",
                title="Consultation Booked",
                message=f"Payment confirmed. Your consultation ({specialty_label}) is booked. A doctor will be assigned shortly.",
                priority=2,
            )
        except Exception:
            pass

        return {
            "success":    True,
            "session_id": session_id,
            "message":    "Payment verified. Consultation created.",
        }
    except Exception as e:
        logger.error(f"Consultation creation after payment failed: {e}")
        raise HTTPException(status_code=500, detail="Payment received but consultation setup failed. Contact support.")


@app.post("/promocode/validate")
@limiter.limit("20/minute")
async def validate_promocode(
    request: Request,
    current_user: Dict = Depends(get_current_user),
):
    """Validate a promo code and return the discount details."""
    body = await request.json()
    code = (body.get("code") or "").strip().upper()
    amount = int(body.get("amount") or 0)   # base amount in rupees

    if not code:
        raise HTTPException(status_code=400, detail="Promo code is required.")

    db_client = DatabaseManager().client
    result = (
        db_client.table("promocodes")
        .select("*")
        .eq("code", code)
        .limit(1)
        .execute()
    )
    row = (result.data or [None])[0]

    if not row:
        raise HTTPException(status_code=404, detail="Invalid promo code.")
    if not row["is_active"]:
        raise HTTPException(status_code=400, detail="This promo code is inactive.")
    if row["expires_at"] and row["expires_at"] < datetime.now(timezone.utc).isoformat():
        raise HTTPException(status_code=400, detail="This promo code has expired.")
    if row["max_uses"] is not None and row["uses_count"] >= row["max_uses"]:
        raise HTTPException(status_code=400, detail="This promo code has reached its usage limit.")

    discount_percent = row["discount_percent"]
    discount_amount  = round(amount * discount_percent / 100)
    final_amount     = max(0, amount - discount_amount)

    return {
        "valid":            True,
        "code":             row["code"],
        "description":      row["description"],
        "discount_percent": discount_percent,
        "discount_amount":  discount_amount,
        "final_amount":     final_amount,
    }


@app.post("/payments/create-hosted-order")
@limiter.limit("10/minute")
async def create_hosted_order(
    request: Request,
    current_user: Dict = Depends(get_current_user),
):
    """
    Create a Razorpay order and return all form fields needed for the
    hosted checkout (form POST to https://api.razorpay.com/v1/checkout/embedded).
    Booking params are encoded in callback_url so the callback endpoint can
    create the consultation after payment verification.
    """
    import razorpay, hmac as _hmac, hashlib, urllib.parse

    body = await request.json()
    amount_paise      = int(body.get("amount", 0))        # already in paise from frontend
    doctor_id         = body.get("doctor_id", "")
    doctor_name       = body.get("doctor_name", "Doctor")
    specialty         = body.get("specialty", "general_medicine")
    scheduled_at      = body.get("scheduled_at", "")
    consultation_type = body.get("consultation_type", "in_person")
    patient_note      = body.get("patient_note", "")

    key_id     = os.getenv("RAZORPAY_KEY_ID", "")
    key_secret = os.getenv("RAZORPAY_KEY_SECRET", "")

    if not key_id or not key_secret:
        raise HTTPException(status_code=503, detail="Payment gateway not configured.")

    try:
        client = razorpay.Client(auth=(key_id, key_secret))
        order = client.order.create({
            "amount":   amount_paise,
            "currency": "INR",
            "notes": {
                "doctor_id":  doctor_id,
                "patient_id": current_user["sub"],
            },
        })
    except Exception as e:
        logger.error(f"Razorpay hosted order creation failed: {e}")
        raise HTTPException(status_code=502, detail=f"Payment gateway error: {str(e)}")

    # Encode booking params into callback_url so the callback can recreate the consultation
    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
    api_base     = os.getenv("API_BASE_URL", "http://localhost:8000")

    booking_params = urllib.parse.urlencode({
        "doctor_id":         doctor_id,
        "specialty":         specialty,
        "scheduled_at":      scheduled_at,
        "consultation_type": consultation_type,
        "patient_note":      patient_note,
        "patient_id":        current_user["sub"],
    })

    callback_url = f"{api_base}/payment/callback?{booking_params}"
    cancel_url   = f"{frontend_url}/book-appointment?payment_error=cancelled"

    # Fetch patient profile for prefill
    db = DatabaseManager()
    profile = await db.get_user_by_id(current_user["sub"])
    patient_name    = _name(profile.get("first_name"), profile.get("last_name")) if profile else ""
    patient_contact = profile.get("phone", "") if profile else ""
    patient_email   = profile.get("email", current_user.get("email", "")) if profile else ""

    return {
        "key_id":           key_id,
        "order_id":         order["id"],
        "amount":           order["amount"],
        "currency":         "INR",
        "name":             "Medivora",
        "description":      f"Consultation with Dr. {doctor_name}",
        "callback_url":     callback_url,
        "cancel_url":       cancel_url,
        "prefill[name]":    patient_name,
        "prefill[contact]": patient_contact,
        "prefill[email]":   patient_email,
    }


@app.post("/payments/verify")
async def verify_payment(request: Request, current_user: Dict = Depends(get_current_user)):
    """
    Called by the frontend after Razorpay checkout.js succeeds.
    Verifies HMAC signature, creates consultation, returns session_id.
    """
    import hmac as _hmac, hashlib

    body                = await request.json()
    razorpay_order_id   = body.get("razorpay_order_id", "")
    razorpay_payment_id = body.get("razorpay_payment_id", "")
    razorpay_signature  = body.get("razorpay_signature", "")
    doctor_id           = body.get("doctor_id", "")
    specialty           = body.get("specialty", "general_medicine")
    scheduled_at        = body.get("scheduled_at") or None
    consultation_type   = body.get("consultation_type", "in_person")
    patient_note        = body.get("patient_note", "")

    key_secret = os.getenv("RAZORPAY_KEY_SECRET", "")
    payload    = f"{razorpay_order_id}|{razorpay_payment_id}"
    expected   = _hmac.new(key_secret.encode(), payload.encode(), hashlib.sha256).hexdigest()

    if not _hmac.compare_digest(expected, razorpay_signature):
        logger.warning(f"verify_payment: invalid signature for order {razorpay_order_id}")
        raise HTTPException(status_code=400, detail="Invalid payment signature")

    try:
        db         = DatabaseManager()
        session_id = str(uuid.uuid4())
        await db.create_consultation({
            "id":                session_id,
            "patient_id":        current_user["sub"],
            "doctor_id":         doctor_id,
            "specialty":         specialty,
            "patient_note":      patient_note,
            "status":            "scheduled",
            "scheduled_at":      scheduled_at,
            "payment_id":        razorpay_payment_id,
            "payment_order_id":  razorpay_order_id,
            "consultation_type": consultation_type,
            "created_at":        datetime.now().isoformat(),
        })
        logger.info(f"Consultation {session_id} created after Razorpay payment {razorpay_payment_id}")
        try:
            await db.create_notification(
                user_id=current_user["sub"],
                notification_type="consultation",
                title="Booking Confirmed",
                message="Your consultation has been booked successfully.",
                priority=2,
            )
        except Exception:
            pass
        return {"session_id": session_id}
    except Exception as e:
        logger.error(f"Consultation creation after verify failed: {e}")
        raise HTTPException(status_code=500, detail="Booking could not be created")


@app.post("/payment/callback")
async def payment_callback(request: Request):
    """
    Browser POST from Razorpay hosted checkout after payment.
    Verifies HMAC signature, creates consultation, redirects to frontend.
    """
    import hmac as _hmac, hashlib, urllib.parse
    from fastapi.responses import RedirectResponse

    form = await request.form()
    razorpay_order_id   = form.get("razorpay_order_id", "")
    razorpay_payment_id = form.get("razorpay_payment_id", "")
    razorpay_signature  = form.get("razorpay_signature", "")

    key_secret   = os.getenv("RAZORPAY_KEY_SECRET", "")
    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")

    # Verify HMAC-SHA256 signature
    payload  = f"{razorpay_order_id}|{razorpay_payment_id}"
    expected = _hmac.new(
        key_secret.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    if not _hmac.compare_digest(expected, razorpay_signature):
        logger.warning(f"Razorpay callback: invalid signature for order {razorpay_order_id}")
        return RedirectResponse(f"{frontend_url}/payment?error=signature_mismatch", status_code=303)

    # Extract booking params from query string
    params            = dict(request.query_params)
    doctor_id         = params.get("doctor_id", "")
    specialty         = params.get("specialty", "general_medicine")
    scheduled_at      = params.get("scheduled_at") or None
    consultation_type = params.get("consultation_type", "in_person")
    patient_note      = params.get("patient_note", "")
    patient_id        = params.get("patient_id", "")

    try:
        db = DatabaseManager()
        session_id = str(uuid.uuid4())
        await db.create_consultation({
            "id":                session_id,
            "patient_id":        patient_id,
            "doctor_id":         doctor_id,
            "specialty":         specialty,
            "patient_note":      patient_note,
            "status":            "scheduled",
            "scheduled_at":      scheduled_at,
            "payment_id":        razorpay_payment_id,
            "payment_order_id":  razorpay_order_id,
            "consultation_type": consultation_type,
            "created_at":        datetime.now().isoformat(),
        })
        logger.info(f"Consultation {session_id} created after Razorpay payment {razorpay_payment_id}")

        try:
            await db.create_notification(
                user_id=patient_id,
                notification_type="consultation",
                title="Booking Confirmed",
                message="Your consultation has been booked successfully.",
                priority=2,
            )
        except Exception:
            pass

        return RedirectResponse(
            f"{frontend_url}/payment?success=true&session_id={session_id}",
            status_code=303,
        )
    except Exception as e:
        logger.error(f"Consultation creation after callback failed: {e}")
        return RedirectResponse(f"{frontend_url}/payment?error=booking_failed", status_code=303)


# ─── Doctor Waitlist ──────────────────────────────────────────────────────────

class DoctorWaitlistRequest(BaseModel):
    name: str
    phone: str

@app.post("/waitlist/doctor")
@limiter.limit("10/minute")
async def join_doctor_waitlist(request: Request, body: DoctorWaitlistRequest):
    """Register a doctor's interest via the landing-page waitlist form."""
    name = body.name.strip()
    phone = body.phone.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Name is required.")
    if not phone:
        raise HTTPException(status_code=422, detail="Phone number is required.")
    try:
        db = DatabaseManager()
        entry = await db.add_to_doctor_waitlist(name=name, phone=phone)
        return {"status": "ok", "id": entry.get("id")}
    except Exception as e:
        logger.error(f"Doctor waitlist insert failed: {e}")
        raise HTTPException(status_code=500, detail="Could not save your details. Please try again.")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from agents.voice_processing_agent import VoiceProcessingAgent
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.debug_mode,
        log_level="info"
    )
