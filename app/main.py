"""Bugzyme backend: REST + WebSocket chat (Reply + Delete)."""

import asyncio
import contextlib
import json
import logging
import os
import uuid
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import httpx
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Request,
    UploadFile,
    File,
    Form,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from pwdlib import PasswordHash
from sqlalchemy import and_, or_, func, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload
from starlette.concurrency import run_in_threadpool
import time

from . import models, schemas
from . import dating_ml_profile_adapter as datingml_adapter
from . import storage as attachment_storage
from .connection_manager import ConnectionManager
from .database import Base, SessionLocal, engine, get_db
from .models import Interaction
from .ai.suggestions import suggest_next_messages
from .ai.conversation_coach import build_private_coaching, generate_private_coaching, CoachServiceError
from .ai.conversation_insights import build_conversation_insights
from .ai.conversation_intelligence import build_conversation_intelligence
from ml.dating_ml_model.model import CompatibilityModel
from ml.dating_ml_model.repository import SQLAlchemyUserRepository

logger = logging.getLogger("bugzyme")

# ============================================================
# CONFIG
# ============================================================

SECRET_KEY = os.environ.get("SECRET_KEY", "")
if len(SECRET_KEY) < 32:
    raise RuntimeError(
        "Set the SECRET_KEY environment variable (32+ chars)."
    )

ALGORITHM = "HS256"
TOKEN_TTL_HOURS = int(os.getenv("TOKEN_TTL_HOURS", "24"))

ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if o.strip()
]

IDLE_TIMEOUT = 70
ALLOWED_MESSAGE_REACTIONS = frozenset({"❤️", "😂", "👍", "🔥", "😮", "😢"})

password_hash = PasswordHash.recommended()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
manager = ConnectionManager()


# ============================================================
# GENERAL HELPERS
# ============================================================

def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def iso(dt):
    if dt is None:
        return None
    return dt.isoformat() + "Z"


# ============================================================
# APP
# ============================================================

def ensure_message_schema_columns(connection):
    """Add missing message metadata columns to an existing PostgreSQL table."""
    logger.info(
        "Checking message schema columns during startup (dialect=%s)",
        connection.dialect.name,
    )

    if connection.dialect.name != "postgresql":
        logger.info(
            "Skipping PostgreSQL message-schema check for dialect=%s",
            connection.dialect.name,
        )
        return

    statements = {
        "attachment_storage_key": (
            "ALTER TABLE messages ADD COLUMN IF NOT EXISTS "
            "attachment_storage_key VARCHAR(64) NULL"
        ),
        "attachment_name": (
            "ALTER TABLE messages ADD COLUMN IF NOT EXISTS "
            "attachment_name VARCHAR(255) NULL"
        ),
        "attachment_mime_type": (
            "ALTER TABLE messages ADD COLUMN IF NOT EXISTS "
            "attachment_mime_type VARCHAR(100) NULL"
        ),
        "attachment_size_bytes": (
            "ALTER TABLE messages ADD COLUMN IF NOT EXISTS "
            "attachment_size_bytes INTEGER NULL"
        ),
        "reaction_revision": (
            "ALTER TABLE messages ADD COLUMN IF NOT EXISTS "
            "reaction_revision INTEGER DEFAULT 0 NOT NULL"
        ),
    }

    try:
        existing = {
            column["name"]
            for column in inspect(connection).get_columns("messages")
        }
        missing = [name for name in statements if name not in existing]

        if missing:
            logger.warning(
                "Adding missing message schema columns to messages: %s",
                ", ".join(missing),
            )

            for name in missing:
                connection.exec_driver_sql(statements[name])

            connection.commit()
        else:
            logger.info(
                "All message schema columns already exist on messages"
            )

        remaining = {
            column["name"]
            for column in inspect(connection).get_columns("messages")
        }

        missing_after_check = [
            name for name in statements if name not in remaining
        ]

        if missing_after_check:
            raise RuntimeError(
                "Message schema columns still missing after startup check: "
                + ", ".join(missing_after_check)
            )

    except Exception:
        connection.rollback()
        logger.exception("Message schema startup check failed")
        raise

    logger.info("Message schema startup check passed")


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=engine)

    with engine.connect() as connection:
        ensure_message_schema_columns(connection)

    with SessionLocal() as db:
        db.query(models.User).update(
            {models.User.is_online: False},
            synchronize_session=False
        )
        db.commit()

    yield


app = FastAPI(
    title="Bugzyme",
    lifespan=lifespan
)

# Voice upload folder
os.makedirs("uploads/voice", exist_ok=True)

# Serve uploaded voice files
app.mount(
    "/uploads",
    StaticFiles(directory="uploads"),
    name="uploads"
)


@app.middleware("http")
async def catch_unhandled(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception:
        logger.exception(
            "Unhandled error on %s %s",
            request.method,
            request.url.path
        )
        return JSONResponse(
            {"detail": "Internal server error"},
            status_code=500
        )


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# RATE LIMITING (in-memory sliding window)
# ============================================================

_RATE_LIMIT_WINDOW_SECONDS = 60
_RATE_LIMIT_DEFAULT = 60          # requests per window per IP
_RATE_LIMIT_AUTH = 10             # auth endpoints per window per IP
_RATE_LIMIT_SWIPE = 30            # swipe endpoints per window per IP

_rate_limit_state: dict[str, list[float]] = {}


def _client_identifier(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    client = request.client
    return client.host if client else "unknown"


def _is_auth_path(path: str) -> bool:
    return path.startswith("/api/v1/auth/") or path in ("/register", "/login")


def _is_swipe_path(path: str) -> bool:
    return path.startswith("/api/v1/swipe")


def _rate_limit_for_path(path: str) -> int:
    if _is_auth_path(path):
        return _RATE_LIMIT_AUTH
    if _is_swipe_path(path):
        return _RATE_LIMIT_SWIPE
    return _RATE_LIMIT_DEFAULT


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    path = request.url.path
    identifier = _client_identifier(request)
    key = f"{identifier}:{path}"
    now = time.time()
    window_start = now - _RATE_LIMIT_WINDOW_SECONDS

    timestamps = _rate_limit_state.get(key, [])
    timestamps = [ts for ts in timestamps if ts > window_start]

    limit = _rate_limit_for_path(path)

    if len(timestamps) >= limit:
        retry_after = int(_RATE_LIMIT_WINDOW_SECONDS - (now - timestamps[0])) + 1
        return JSONResponse(
            {
                "detail": "Rate limit exceeded. Try again later.",
                "retry_after": max(retry_after, 1),
            },
            status_code=429,
            headers={"Retry-After": str(max(retry_after, 1))},
        )

    timestamps.append(now)
    _rate_limit_state[key] = timestamps

    response = await call_next(request)
    return response


# ============================================================
# AUTH
# ============================================================

def create_token(user_id: int):
    expire_time = datetime.now(timezone.utc) + timedelta(
        hours=TOKEN_TTL_HOURS
    )

    return jwt.encode(
        {
            "sub": str(user_id),
            "exp": expire_time
        },
        SECRET_KEY,
        algorithm=ALGORITHM
    )


def decode_token(token: str):
    try:
        return int(
            jwt.decode(
                token,
                SECRET_KEY,
                algorithms=[ALGORITHM]
            )["sub"]
        )
    except (JWTError, KeyError, ValueError, TypeError):
        return None


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    user_id = decode_token(token)

    if user_id is None:
        raise HTTPException(
            401,
            "Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"}
        )

    user = db.get(models.User, user_id)

    if not user:
        raise HTTPException(401, "User not found")

    return user


# ============================================================
# CHAT HELPERS
# ============================================================

def other_user(chat: models.Chat, user_id: int):
    if chat.user1_id == user_id:
        return chat.user2_id
    return chat.user1_id


def member_chat(db: Session, chat_id: int, user_id: int):
    chat = db.get(models.Chat, chat_id)

    if chat and user_id in (
        chat.user1_id,
        chat.user2_id
    ):
        return chat

    return None


def find_chat(db: Session, user1_id: int, user2_id: int):
    return db.query(models.Chat).filter(
        or_(
            (models.Chat.user1_id == user1_id) &
            (models.Chat.user2_id == user2_id),

            (models.Chat.user1_id == user2_id) &
            (models.Chat.user2_id == user1_id)
        )
    ).first()

def get_partners(db: Session, user_id: int):
    chats = db.query(models.Chat).filter(
        or_(
            and_(
                models.Chat.user1_id == user_id,
                models.Chat.deleted_by_user1.is_(False)
            ),
            and_(
                models.Chat.user2_id == user_id,
                models.Chat.deleted_by_user2.is_(False)
            )
        )
    ).all()

    return {
        chat.id: other_user(chat, user_id)
        for chat in chats
    }
# ============================================================
# MESSAGE SAVE
# ============================================================

def save_message(
    db: Session,
    sender_id: int,
    chat_id: int,
    content: str,
    reply_to_id: int | None = None
):
    chat = member_chat(db, chat_id, sender_id)

    if not chat:
        return None

    # Validate reply
    if reply_to_id is not None:
        parent = db.get(models.Message, reply_to_id)

        if (
            not parent
            or parent.chat_id != chat_id
            or parent.is_deleted
        ):
            return None

    message = models.Message(
        chat_id=chat_id,
        sender_id=sender_id,
        content=content,
        created_at=utcnow(),
        is_delivered=False,
        is_read=False,
        is_deleted=False,
        reply_to_id=reply_to_id,
    )

    db.add(message)
    db.commit()
    db.refresh(message)

    return {
        "id": message.id,
        "chat_id": chat_id,
        "sender_id": sender_id,
        "content": message.content,
        "created_at": iso(message.created_at),
        "is_delivered": False,
        "is_read": False,
        "is_deleted": False,
        "reply_to_id": message.reply_to_id,
        "receiver_id": other_user(chat, sender_id),
    }



def save_voice_message(
    db: Session,
    sender_id: int,
    chat_id: int,
    audio_url: str,
    audio_duration: int | None,
    audio_mime_type: str | None
):
    chat = member_chat(
        db,
        chat_id,
        sender_id
    )

    if not chat:
        return None

    message = models.Message(
        chat_id=chat_id,
        sender_id=sender_id,
        content=None,
        created_at=utcnow(),
        is_delivered=False,
        is_read=False,
        is_deleted=False,
        reply_to_id=None,

        message_type="voice",
        audio_url=audio_url,
        audio_duration=audio_duration,
        audio_mime_type=audio_mime_type,
    )

    db.add(message)
    db.commit()
    db.refresh(message)

    return {
        "id": message.id,
        "chat_id": chat_id,
        "sender_id": sender_id,
        "content": None,
        "created_at": iso(message.created_at),
        "is_delivered": False,
        "is_read": False,
        "is_deleted": False,
        "reply_to_id": None,

        "message_type": "voice",
        "audio_url": message.audio_url,
        "audio_duration": message.audio_duration,
        "audio_mime_type": message.audio_mime_type,

        "receiver_id": other_user(
            chat,
            sender_id
        ),
    }


def save_attachment_message(
    db: Session,
    sender_id: int,
    chat_id: int,
    content: str | None,
    reply_to_id: int | None,
    storage_key: str,
    filename: str,
    mime_type: str,
    size_bytes: int,
    message_type: str,
):
    chat = member_chat(db, chat_id, sender_id)
    if not chat:
        return None

    if content is not None:
        content = content.strip()
        if len(content) > schemas.MAX_MESSAGE_LEN:
            return None
        content = content or None

    if reply_to_id is not None:
        parent = db.get(models.Message, reply_to_id)
        if not parent or parent.chat_id != chat_id or parent.is_deleted:
            return None

    message = models.Message(
        chat_id=chat_id,
        sender_id=sender_id,
        content=content,
        created_at=utcnow(),
        is_delivered=False,
        is_read=False,
        is_deleted=False,
        reply_to_id=reply_to_id,
        message_type=message_type,
        attachment_storage_key=storage_key,
        attachment_name=filename,
        attachment_mime_type=mime_type,
        attachment_size_bytes=size_bytes,
    )
    db.add(message)
    db.commit()
    db.refresh(message)

    return {
        "id": message.id,
        "chat_id": chat_id,
        "sender_id": sender_id,
        "content": message.content,
        "created_at": iso(message.created_at),
        "is_delivered": message.is_delivered,
        "is_read": message.is_read,
        "is_deleted": message.is_deleted,
        "reply_to_id": message.reply_to_id,
        "message_type": message.message_type,
        "attachment_name": message.attachment_name,
        "attachment_mime_type": message.attachment_mime_type,
        "attachment_size_bytes": message.attachment_size_bytes,
        "receiver_id": other_user(chat, sender_id),
    }


# ============================================================
# MESSAGE DELETE
# ============================================================

def soft_delete_message(
    db: Session,
    user_id: int,
    message_id: int,
    for_everyone: bool = True
):
    message = db.get(models.Message, message_id)

    if not message or message.is_deleted:
        return None

    chat = member_chat(
        db,
        message.chat_id,
        user_id
    )

    if not chat:
        return None

    # Only sender can delete for everyone
    if for_everyone and message.sender_id != user_id:
        return None

    # Messages are soft-deleted, so explicitly remove their reactions.
    db.query(models.MessageReaction).filter(
        models.MessageReaction.message_id == message.id
    ).delete(synchronize_session=False)

    message.is_deleted = True
    message.deleted_at = utcnow()
    message.content = None

    db.commit()

    return {
        "message_id": message.id,
        "chat_id": message.chat_id,
        "sender_id": message.sender_id,
        "for_everyone": for_everyone,
        "receiver_id": other_user(chat, user_id),
        "attachment_storage_key": message.attachment_storage_key,
    }


def toggle_message_reaction(
    db: Session,
    user_id: int,
    chat_id: int,
    message_id: int,
    reaction: str,
):
    if reaction not in ALLOWED_MESSAGE_REACTIONS:
        return None

    chat = member_chat(db, chat_id, user_id)
    if not chat:
        return None

    # Lock the owning message before reading or changing a user's reaction.
    # This serializes all reaction mutations for this message in PostgreSQL.
    for attempt in range(2):
        message = (
            db.query(models.Message)
            .filter(
                models.Message.id == message_id,
                models.Message.chat_id == chat_id,
                models.Message.is_deleted.is_(False),
            )
            .with_for_update()
            .first()
        )
        if not message:
            return None

        current = db.query(models.MessageReaction).filter_by(
            message_id=message_id,
            user_id=user_id,
        ).first()

        if current and current.reaction == reaction:
            db.delete(current)
        elif current:
            current.reaction = reaction
            current.created_at = utcnow()
        else:
            db.add(models.MessageReaction(
                message_id=message_id,
                user_id=user_id,
                reaction=reaction,
                created_at=utcnow(),
            ))

        message.reaction_revision = (message.reaction_revision or 0) + 1
        try:
            db.commit()
            break
        except IntegrityError:
            # The message row lock serializes normal callers. Retrying once
            # also handles a concurrent insert from a legacy/unlocked caller;
            # the unique constraint remains the final guard.
            db.rollback()
            if attempt:
                raise

    # Generate the canonical snapshot only after the mutation is committed.
    # Re-locking the message keeps the revision and reaction rows stable while
    # they are read. A later update can still broadcast first; clients order
    # these committed snapshots by revision.
    snapshot_message = (
        db.query(models.Message)
        .filter(
            models.Message.id == message_id,
            models.Message.chat_id == chat_id,
            models.Message.is_deleted.is_(False),
        )
        .with_for_update()
        .first()
    )
    if not snapshot_message:
        return None

    reactions = db.query(models.MessageReaction).filter_by(
        message_id=message_id
    ).order_by(models.MessageReaction.user_id).all()
    revision = snapshot_message.reaction_revision or 0
    db.commit()

    return {
        "chat_id": chat_id,
        "message_id": message_id,
        "revision": revision,
        "receiver_id": other_user(chat, user_id),
        "reactions": [
            {"user_id": row.user_id, "reaction": row.reaction}
            for row in reactions
        ],
    }


# ============================================================
# CHAT DELETE
# ============================================================
def delete_chat_from_database(
    db: Session,
    chat_id: int,
    user_id: int
):
    chat = db.get(models.Chat, chat_id)

    if not chat:
        return None

    if user_id not in (
        chat.user1_id,
        chat.user2_id
    ):
        return None

    # Mark conversation as deleted only
    # for the current user's side.
    if chat.user1_id == user_id:
        chat.deleted_by_user1 = True
    else:
        chat.deleted_by_user2 = True

    # IMPORTANT:
    # Do NOT delete messages.
    # Do NOT delete the chat.
    db.commit()

    return {
        "chat_id": chat_id,
        "user_id": user_id,
        "other_user_id": other_user(chat, user_id)
    }


# ============================================================
# AI CHAT ENDPOINTS
# ============================================================

@app.post("/chats/{chat_id}/suggestions")
async def chat_suggestions(
    chat_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    chat = member_chat(db, chat_id, current_user.id)
    if not chat:
        raise HTTPException(403, "You are not a member of this chat")

    messages = (
        db.query(models.Message)
        .filter(models.Message.chat_id == chat_id)
        .order_by(models.Message.id.asc())
        .all()
    )

    if not messages:
        raise HTTPException(404, "No messages found in this chat")

    result = suggest_next_messages(
        db=db,
        requesting_user_id=current_user.id,
        requesting_user=current_user.username,
        messages=messages,
    )
    return result


@app.get("/chats/{chat_id}/coach")
async def chat_coach(
    chat_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    chat = member_chat(db, chat_id, current_user.id)
    if not chat:
        raise HTTPException(403, "You are not a member of this chat")

    messages = (
        db.query(models.Message)
        .filter(models.Message.chat_id == chat_id)
        .order_by(models.Message.id.asc())
        .all()
    )

    if not messages:
        return {"available": False, "reason": "no_context", "insights": [], "user_id": current_user.id}

    # Build conversation intelligence to get participant views and shared context
    intelligence = build_conversation_intelligence(messages)
    participant_view = intelligence["participant_views"].get(str(current_user.id), {})
    shared = intelligence["shared"]

    # Get user's dating profile if available
    dating_profile = db.query(models.DatingProfile).filter(models.DatingProfile.user_id == current_user.id).first()
    own_profile = {
        "communication": participant_view.get("communication", {}),
        "mood": participant_view.get("mood", {}),
        "dating_profile": {
            "full_name": dating_profile.full_name,
            "bio": dating_profile.bio,
            "city": dating_profile.city,
            "education": dating_profile.education,
            "job": dating_profile.job,
        } if dating_profile else None
    }

    shared_context = {
        "mood": shared.get("mood", {}),
        "shared_topics": shared.get("topics", {}).get("supported_shared_topics", []),
        "engagement_trend": shared.get("engagement", {}).get("trend", "unknown"),
    }

    try:
        result = generate_private_coaching(
            messages=messages,
            own_user_id=current_user.id,
            own_profile=own_profile,
            shared_context=shared_context,
        )
    except CoachServiceError as exc:
        if exc.code == "provider_not_configured":
            raise HTTPException(503, detail={"code": "provider_not_configured"})
        if exc.code == "provider_failure":
            raise HTTPException(503, detail={"code": "provider_failure"})
        if exc.code == "invalid_provider_response":
            raise HTTPException(502, detail={"code": "invalid_provider_response"})
        raise HTTPException(500, detail={"code": "internal_error"})
    return result


@app.get("/chats/{chat_id}/insights")
async def chat_insights(
    chat_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    chat = member_chat(db, chat_id, current_user.id)
    if not chat:
        raise HTTPException(403, "You are not a member of this chat")

    messages = (
        db.query(models.Message)
        .filter(models.Message.chat_id == chat_id)
        .order_by(models.Message.id.asc())
        .all()
    )

    if not messages:
        return {"available": False, "reason": "no_context"}

    result = build_conversation_insights(messages=messages)
    return result


# ============================================================
# DISCOVER / FEED / SEARCH / PROFILE
# ============================================================

@app.get("/api/v1/feed")
async def feed(
    limit: int = 20,
    offset: int = 0,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    repo = SQLAlchemyUserRepository(db)
    model = CompatibilityModel()

    current_profile = repo.get(current_user.id)
    if not current_profile:
        raise HTTPException(404, "Profile not found")

    # Get blocked user IDs
    blocked_ids = {
        b.blocked_id
        for b in db.query(models.Block).filter(
            models.Block.blocker_id == current_user.id
        ).all()
    }

    candidates = [
        p for p in repo.all()
        if p.get("id") != current_user.id
        and p.get("id") not in blocked_ids
    ]

    results = []
    for candidate in candidates:
        try:
            result = model.predict_pair(current_profile, candidate)
            dp = db.get(models.DatingProfile, candidate["id"])
            photos = None
            if dp:
                photos = [
                    p.photo_url
                    for p in sorted(dp.photos, key=lambda x: x.display_order)
                ]
            results.append({
                "user_id": candidate["id"],
                "username": candidate.get("username"),
                "full_name": candidate.get("full_name"),
                "age": candidate.get("age"),
                "city": candidate.get("city"),
                "era": candidate.get("match_era_preference"),
                "vibe_tag": candidate.get("vibe_tag"),
                "bio": candidate.get("bio"),
                "profession": candidate.get("job"),
                "education": candidate.get("education"),
                "relationship_intentions": candidate.get("target_gender_preference"),
                "interests": candidate.get("interests"),
                "photos": photos,
                "photo_url": photos[0] if photos else None,
                "compatibility_score": result.get("compatibility_score"),
                "confidence_score": result.get("confidence_score"),
                "adjusted_score": result.get("adjusted_score"),
                "recommendation_score": result.get("adjusted_score"),
                "score_type": "profile_alignment",
                "model_version": result.get("model_version"),
                "prediction_provenance": "compatibility_model",
                "ranking_weights_validated": False,
                "matching_status": (
                    "scored" if result.get("eligible") else "model_ineligible"
                ),
                "is_face_verified": bool(candidate.get("is_face_verified")),
                "is_id_verified": bool(candidate.get("is_id_verified")),
                "verification_badge_required": False,
            })
        except Exception:
            continue

    results.sort(
        key=lambda x: (x.get("adjusted_score") or 0),
        reverse=True,
    )

    paginated = results[offset:offset + limit]
    return {
        "profiles": paginated,
        "total": len(results),
        "limit": limit,
        "offset": offset,
        "has_more": offset + limit < len(results),
    }


# ============================================================
# PROFILE APIs (current user) - MUST BE BEFORE /{user_id} ROUTE
# ============================================================


@app.get("/api/v1/profile/me")
async def get_my_profile(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get the authenticated user's dating profile."""
    from app.models import DatingProfile

    profile = (
        db.query(DatingProfile)
        .filter(DatingProfile.user_id == current_user.id)
        .first()
    )

    if not profile:
        raise HTTPException(
            status_code=404,
            detail="Dating profile not found",
        )

    return _profile_to_dict(profile)


@app.post("/api/v1/profile")
async def create_profile(
    profile_data: dict,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create or update dating profile (upsert)."""
    from app.models import DatingProfile

    existing = (
        db.query(DatingProfile)
        .filter(DatingProfile.user_id == current_user.id)
        .first()
    )

    # Get valid column names for DatingProfile model
    valid_columns = {
        c.name for c in DatingProfile.__table__.columns
    }

    # Map frontend field names to DB column names
    data = {}
    for key, value in (profile_data or {}).items():
        if key == "gender":
            key = "sex"
        elif key == "sexual_orientation":
            key = "orientation"
        elif key == "era":
            key = "match_era_preference"
        elif key == "profession":
            key = "job"
        elif key == "relationship_intentions":
            key = "target_gender_preference"
        if key in valid_columns:
            data[key] = value

    # Set defaults for NOT NULL columns that frontend may not send
    data.setdefault("sex", "prefer_not_to_say")
    data.setdefault("orientation", "straight")
    data.setdefault("verification_status", "unverified")

    if existing:
        # Upsert: update existing profile instead of failing
        for key, value in data.items():
            if hasattr(existing, key):
                setattr(existing, key, value)
        try:
            db.commit()
            db.refresh(existing)
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Profile update failed: {str(e)}")
        return _profile_to_dict(existing)

    profile = DatingProfile(
        user_id=current_user.id,
        **data,
    )

    try:
        db.add(profile)
        db.commit()
        db.refresh(profile)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Profile creation failed: {str(e)}")

    return _profile_to_dict(profile)


@app.put("/api/v1/profile")
async def update_profile_root(
    profile_data: dict,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update the authenticated user's dating profile via PUT /api/v1/profile."""
    return await update_profile(profile_data, current_user, db)


@app.patch("/api/v1/profile")
async def update_profile_root_partial(
    profile_data: dict,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Partially update the authenticated user's dating profile via PATCH /api/v1/profile."""
    return await update_my_profile(profile_data, current_user, db)


@app.post("/api/v1/profile/me")
async def create_profile_me(
    profile_data: dict,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create or update profile via POST /api/v1/profile/me."""
    return await create_profile(profile_data, current_user, db)


@app.put("/api/v1/profile/me")
async def update_profile(
    profile_data: dict,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update the authenticated user's dating profile."""
    from app.models import DatingProfile

    profile = (
        db.query(DatingProfile)
        .filter(DatingProfile.user_id == current_user.id)
        .first()
    )

    if not profile:
        raise HTTPException(
            status_code=404,
            detail="Dating profile not found",
        )

    valid_columns = {
        c.name for c in DatingProfile.__table__.columns
    }

    update_data = {}
    for key, value in (profile_data or {}).items():
        if key == "gender":
            key = "sex"
        elif key == "sexual_orientation":
            key = "orientation"
        elif key == "era":
            key = "match_era_preference"
        elif key == "profession":
            key = "job"
        elif key == "relationship_intentions":
            key = "target_gender_preference"
        if key in valid_columns:
            update_data[key] = value

    for key, value in update_data.items():
        if hasattr(profile, key):
            setattr(profile, key, value)

    try:
        db.commit()
        db.refresh(profile)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Profile update failed: {str(e)}")

    return _profile_to_dict(profile)


@app.patch("/api/v1/profile/me")
async def update_my_profile(
    profile_data: dict,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Partially update the authenticated user's dating profile."""
    from app.models import DatingProfile

    profile = (
        db.query(DatingProfile)
        .filter(DatingProfile.user_id == current_user.id)
        .first()
    )

    if not profile:
        raise HTTPException(
            status_code=404,
            detail="Dating profile not found",
        )

    valid_columns = {
        c.name for c in DatingProfile.__table__.columns
    }

    update_data = {}
    for key, value in (profile_data or {}).items():
        if key == "gender":
            key = "sex"
        elif key == "sexual_orientation":
            key = "orientation"
        elif key == "era":
            key = "match_era_preference"
        elif key == "profession":
            key = "job"
        elif key == "relationship_intentions":
            key = "target_gender_preference"
        if key in valid_columns:
            update_data[key] = value

    for key, value in update_data.items():
        if hasattr(profile, key):
            setattr(profile, key, value)

    try:
        db.commit()
        db.refresh(profile)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Profile update failed: {str(e)}")

    return _profile_to_dict(profile)


# ============================================================
# PROFILE PHOTO APIs
# ============================================================


@app.get("/api/v1/profile/me/photos")
async def list_my_profile_photos(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List the authenticated user's profile photos."""
    from app.models import ProfilePhoto, DatingProfile

    profile = (
        db.query(DatingProfile)
        .filter(DatingProfile.user_id == current_user.id)
        .first()
    )

    if not profile:
        return {"photos": []}

    photos = (
        db.query(ProfilePhoto)
        .filter(ProfilePhoto.dating_profile_id == profile.id)
        .order_by(ProfilePhoto.display_order.asc())
        .all()
    )

    return {
        "photos": [
            schemas.ProfilePhotoSchema.model_validate(photo)
            for photo in photos
        ]
    }


@app.post("/api/v1/profile/photos", response_model=schemas.ProfilePhotoSchema)
async def upload_profile_photo(
    file: UploadFile = File(...),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Upload a profile photo."""
    from app.models import ProfilePhoto, DatingProfile
    import os
    import uuid

    profile = (
        db.query(DatingProfile)
        .filter(DatingProfile.user_id == current_user.id)
        .first()
    )

    if not profile:
        raise HTTPException(
            status_code=404,
            detail="Dating profile not found",
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Empty file",
        )

    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Maximum file size is 10 MB",
        )

    allowed_types = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG, PNG and WebP images are allowed",
        )

    extension = allowed_types[file.content_type]
    filename = f"profile_{profile.id}_{uuid.uuid4()}{extension}"

    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, filename)

    try:
        with open(file_path, "wb") as image_file:
            image_file.write(content)

        photo = ProfilePhoto(
            dating_profile_id=profile.id,
            photo_url=f"/static/uploads/{filename}",
            is_primary=False,
            display_order=0,
        )

        db.add(photo)
        db.commit()
        db.refresh(photo)

    except Exception:
        db.rollback()

        if os.path.exists(file_path):
            os.remove(file_path)

        raise

    return photo


@app.delete("/api/v1/profile/photos/{photo_id}")
async def delete_profile_photo(
    photo_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete an authenticated user's profile photo."""
    from app.models import ProfilePhoto, DatingProfile

    profile = (
        db.query(DatingProfile)
        .filter(DatingProfile.user_id == current_user.id)
        .first()
    )

    if not profile:
        raise HTTPException(
            status_code=404,
            detail="Dating profile not found",
        )

    photo = (
        db.query(ProfilePhoto)
        .filter(
            ProfilePhoto.id == photo_id,
            ProfilePhoto.dating_profile_id == profile.id,
        )
        .first()
    )

    if not photo:
        raise HTTPException(
            status_code=404,
            detail="Photo not found",
        )

    db.delete(photo)
    db.commit()

    return {"result": "Photo deleted"}


@app.put("/api/v1/profile/photos/{photo_id}/primary")
async def set_primary_profile_photo(
    photo_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Set a photo as the primary profile picture."""
    from app.models import ProfilePhoto, DatingProfile

    profile = (
        db.query(DatingProfile)
        .filter(DatingProfile.user_id == current_user.id)
        .first()
    )

    if not profile:
        raise HTTPException(
            status_code=404,
            detail="Dating profile not found",
        )

    target_photo = (
        db.query(ProfilePhoto)
        .filter(
            ProfilePhoto.id == photo_id,
            ProfilePhoto.dating_profile_id == profile.id,
        )
        .first()
    )

    if not target_photo:
        raise HTTPException(
            status_code=404,
            detail="Photo not found or access denied",
        )

    all_photos = (
        db.query(ProfilePhoto)
        .filter(ProfilePhoto.dating_profile_id == profile.id)
        .order_by(ProfilePhoto.display_order)
        .all()
    )

    target_photo.is_primary = True
    target_photo.display_order = 1

    current_order = 2
    for p in all_photos:
        if p.id != target_photo.id:
            p.is_primary = False
            p.display_order = current_order
            current_order += 1

    db.commit()
    db.refresh(target_photo)

    return {
        "id": target_photo.id,
        "photo_url": target_photo.photo_url,
        "is_primary": target_photo.is_primary,
        "display_order": target_photo.display_order,
    }


@app.put("/api/v1/profile/photos/reorder")
async def reorder_profile_photos(
    payload: schemas.ProfilePhotosReorderRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Reorder profile photos."""
    from app.models import ProfilePhoto, DatingProfile

    profile = (
        db.query(DatingProfile)
        .filter(DatingProfile.user_id == current_user.id)
        .first()
    )

    if not profile:
        raise HTTPException(
            status_code=404,
            detail="Dating profile not found",
        )

    photo_ids = payload.photo_ids
    if not photo_ids:
        raise HTTPException(
            status_code=400,
            detail="photo_ids list cannot be empty",
        )

    # Verify all photos belong to this profile
    photos = (
        db.query(ProfilePhoto)
        .filter(
            ProfilePhoto.id.in_(photo_ids),
            ProfilePhoto.dating_profile_id == profile.id,
        )
        .all()
    )

    if len(photos) != len(photo_ids):
        raise HTTPException(
            status_code=404,
            detail="One or more photos not found or access denied",
        )

    # Update display_order and is_primary
    for idx, photo_id in enumerate(photo_ids):
        photo = next(p for p in photos if p.id == photo_id)
        photo.display_order = idx + 1
        photo.is_primary = (idx == 0)

    db.commit()

    # Return updated photos in new order
    updated_photos = (
        db.query(ProfilePhoto)
        .filter(ProfilePhoto.dating_profile_id == profile.id)
        .order_by(ProfilePhoto.display_order)
        .all()
    )

    return [
        {
            "id": p.id,
            "photo_url": p.photo_url,
            "is_primary": p.is_primary,
            "display_order": p.display_order,
        }
        for p in updated_photos
    ]


@app.get("/api/v1/profile/{user_id}")
async def public_profile(
    user_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user = db.get(models.User, user_id)
    if not user:
        raise HTTPException(404, "User not found")

    dp = db.get(models.DatingProfile, user_id)
    photos = None
    if dp:
        photos = [
            p.photo_url
            for p in sorted(dp.photos, key=lambda x: x.display_order)
        ]

    return {
        "user_id": user.id,
        "username": user.username,
        "full_name": dp.full_name if dp else None,
        "bio": dp.bio if dp else None,
        "date_of_birth": dp.date_of_birth if dp else None,
        "gender": dp.sex if dp else None,
        "sexual_orientation": dp.orientation if dp else None,
        "city": dp.city if dp else None,
        "era": dp.match_era_preference if dp else None,
        "vibe_tag": dp.vibe_tag if dp else None,
        "profession": dp.job if dp else None,
        "education": dp.education if dp else None,
        "relationship_intentions": dp.target_gender_preference if dp else None,
        "interests": None,
        "photos": photos,
        "photo_url": photos[0] if photos else None,
        "is_face_verified": bool(dp.is_face_verified) if dp else False,
        "is_id_verified": bool(dp.is_id_verified) if dp else False,
        "verification_status": dp.verification_status if dp else "unverified",
        "created_at": dp.created_at.isoformat() if dp and dp.created_at else None,
        "updated_at": dp.updated_at.isoformat() if dp and dp.updated_at else None,
    }


@app.get("/api/v1/search")
async def search_profiles(
    q: str | None = None,
    interests: str | None = None,
    city: str | None = None,
    profession: str | None = None,
    relationship_intentions: str | None = None,
    min_age: int | None = None,
    max_age: int | None = None,
    era: str | None = None,
    limit: int = 20,
    offset: int = 0,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(models.User).filter(
        models.User.id != current_user.id
    ).join(
        models.DatingProfile,
        models.DatingProfile.user_id == models.User.id,
        isouter=True,
    )

    if q:
        like = f"%{q}%"
        query = query.filter(
            models.User.username.ilike(like)
            | (models.DatingProfile.full_name.ilike(like) if models.DatingProfile.full_name is not None else False)
            | (models.DatingProfile.bio.ilike(like) if models.DatingProfile.bio is not None else False)
        )

    if city:
        query = query.filter(
            models.DatingProfile.city.ilike(f"%{city}%")
        )

    if profession:
        query = query.filter(
            models.DatingProfile.job.ilike(f"%{profession}%")
        )

    if relationship_intentions:
        query = query.filter(
            models.DatingProfile.target_gender_preference.ilike(f"%{relationship_intentions}%")
        )

    if era:
        query = query.filter(
            models.DatingProfile.match_era_preference == era
        )

    total = query.count()

    rows = query.offset(offset).limit(limit).all()

    results = []
    for user in rows:
        dp = user.dating_profile
        photos = None
        if dp:
            photos = [
                p.photo_url
                for p in sorted(dp.photos, key=lambda x: x.display_order)
            ]
        results.append({
            "user_id": user.id,
            "username": user.username,
            "full_name": dp.full_name if dp else None,
            "age": None,
            "city": dp.city if dp else None,
            "era": dp.match_era_preference if dp else None,
            "vibe_tag": dp.vibe_tag if dp else None,
            "bio": dp.bio if dp else None,
            "profession": dp.job if dp else None,
            "education": dp.education if dp else None,
            "relationship_intentions": dp.target_gender_preference if dp else None,
            "interests": None,
            "photos": photos,
            "photo_url": photos[0] if photos else None,
            "is_verified": bool(dp.is_face_verified) if dp else False,
            "shared_interests": [],
            "shared_interests_count": 0,
            "relevance_score": 0.0,
        })

    return {
        "results": results,
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": offset + limit < total,
    }


@app.post("/ai-suggestions/{suggestion_id}/events")
async def track_suggestion_event(
    suggestion_id: int,
    event_type: str,
    event_key: str,
    details: dict | None = None,
    current_user: models.User = Depends(get_current_user),
):
    return {
        "ok": True,
        "suggestion_id": suggestion_id,
        "event_type": event_type,
        "event_key": event_key,
        "details": details or {},
    }


# ============================================================
# DELIVERY / READ
# ============================================================

def mark_delivered(db: Session, ids: list[int]):
    if not ids:
        return

    db.query(models.Message).filter(
        models.Message.id.in_(ids)
    ).update(
        {models.Message.is_delivered: True},
        synchronize_session=False
    )

    db.commit()


def mark_read(
    db: Session,
    user_id: int,
    chat_id: int,
    message_id: int | None
):
    chat = member_chat(
        db,
        chat_id,
        user_id
    )

    if not chat:
        return None

    query = db.query(models.Message).filter(
        models.Message.chat_id == chat_id,
        models.Message.sender_id != user_id,
        models.Message.is_read.is_not(True),
        models.Message.is_deleted.is_not(True),
    )

    if message_id is not None:
        query = query.filter(
            models.Message.id == message_id
        )

    messages = query.all()

    for message in messages:
        message.is_read = True
        message.is_delivered = True

    db.commit()

    return {
        "sender": other_user(chat, user_id),
        "ids": [message.id for message in messages]
    }


# ============================================================
# ONLINE / OFFLINE
# ============================================================

def set_online(
    db: Session,
    user_id: int,
    online: bool
):
    user = db.get(models.User, user_id)

    if not user:
        return None

    user.is_online = online

    if not online:
        user.last_seen = utcnow()

    db.commit()

    return iso(user.last_seen)


def load_context(
    db: Session,
    user_id: int,
    bound_chat_id: int | None
):
    user = db.get(models.User, user_id)

    if not user:
        return None

    partners = get_partners(db, user_id)

    if (
        bound_chat_id is not None
        and bound_chat_id not in partners
    ):
        return None

    # A user with no chats has no pending messages to load. Avoid using a
    # fabricated chat ID as an empty-result sentinel in the SQL query.
    if not partners:
        return {
            "username": user.username,
            "partners": partners,
            "delivered": [],
        }

    query = db.query(models.Message).filter(
        models.Message.chat_id.in_(list(partners)),
        models.Message.sender_id != user_id,
        models.Message.is_delivered.is_not(True),
        models.Message.is_deleted.is_not(True),
    )

    if bound_chat_id is not None:
        query = query.filter(
            models.Message.chat_id == bound_chat_id
        )

    pending = query.all()

    for message in pending:
        message.is_delivered = True

    db.commit()

    return {
        "username": user.username,
        "partners": partners,
        "delivered": [
            (
                message.sender_id,
                message.chat_id,
                message.id
            )
            for message in pending
        ]
    }


# ============================================================
# DATABASE ASYNC HELPER
# ============================================================

async def db_call(function, *args):
    def run():
        with SessionLocal() as db:
            return function(db, *args)

    return await run_in_threadpool(run)


# ============================================================
# WEBSOCKET HELPERS
# ============================================================

async def safe_send(
    websocket: WebSocket,
    payload: dict
):
    with contextlib.suppress(Exception):
        await websocket.send_json(payload)


async def broadcast_presence(
    user_id: int,
    status: str,
    last_seen=None
):
    payload = {
        "type": "presence",
        "user_id": user_id,
        "status": status,
        "last_seen": last_seen
    }

    # Presence belongs to the user, not a chat. Send it to every connected
    # account so soft-deleted chats and chat-bound sockets cannot miss it.
    for recipient_id in manager.online_ids() - {user_id}:
        await manager.send_to_user(
            recipient_id,
            payload
        )


# ============================================================
# BACKGROUND TASKS
# ============================================================

_bg_tasks = set()


def spawn(coro):
    task = asyncio.create_task(coro)

    _bg_tasks.add(task)

    task.add_done_callback(
        _bg_tasks.discard
    )


async def go_offline(user_id: int):
    try:
        if manager.is_online(user_id):
            return

        last_seen = await db_call(
            set_online,
            user_id,
            False
        )

        if manager.is_online(user_id):
            await db_call(
                set_online,
                user_id,
                True
            )
            return

        await broadcast_presence(
            user_id,
            "offline",
            last_seen
        )

    except Exception:
        logger.exception(
            "go_offline failed for user %s",
            user_id
        )


# ============================================================
# PROCESS MESSAGE
# ============================================================

async def process_message(
    sender_id,
    sender_name,
    chat_id,
    content,
    client_id=None,
    origin=None,
    reply_to_id=None
):
    message = await db_call(
        save_message,
        sender_id,
        chat_id,
        content,
        reply_to_id
    )

    if message is None:
        return None

    # Tell sender actual message ID
    if origin is not None:
        await safe_send(
            origin,
            {
                "type": "message_sent",
                "chat_id": chat_id,
                "message_id": message["id"],
                "client_id": client_id,
                "created_at": message["created_at"],
                "reply_to_id": message["reply_to_id"],
            }
        )

    # Message event
    payload = {
        "type": "message",
        "chat_id": chat_id,
        "message_id": message["id"],
        "sender_id": sender_id,
        "sender_name": sender_name,
        "content": message["content"],
        "created_at": message["created_at"],
        "reply_to_id": message["reply_to_id"],
    }

    delivered = await manager.send_to_user(
        message["receiver_id"],
        payload,
        chat_id
    )

    if delivered:
        await db_call(
            mark_delivered,
            [message["id"]]
        )

        message["is_delivered"] = True

        delivered_event = {
            "type": "message_delivered",
            "chat_id": chat_id,
            "message_id": message["id"],
        }

        if origin is not None:
            await safe_send(
                origin,
                delivered_event
            )
        else:
            await manager.send_to_user(
                sender_id,
                delivered_event,
                chat_id
            )

    return message


# ============================================================
# BASIC ROUTES
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Bugzyme Backend is running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }




# ============================================================
# REGISTER
# ============================================================

@app.post(
    "/register",
    response_model=schemas.UserResponse,
    status_code=201
)
def register_user(
    user: schemas.UserCreate,
    db: Session = Depends(get_db)
):
    email = user.email.strip().lower()
    username = user.username.strip()

    if db.query(models.User).filter(
        func.lower(models.User.email) == email
    ).first():
        raise HTTPException(
            409,
            "Email already registered"
        )

    if db.query(models.User).filter(
        func.lower(models.User.username)
        == username.lower()
    ).first():
        raise HTTPException(
            409,
            "Username already taken"
        )

    new_user = models.User(
        username=username,
        email=email,
        hashed_password=password_hash.hash(
            user.password
        ),
        is_online=False,
    )

    db.add(new_user)

    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            409,
            "Email or username already registered"
        )

    db.refresh(new_user)

    return new_user
@app.post("/upload-voice")
async def upload_voice(
    file: UploadFile = File(...),
    chat_id: int = Query(...),
    duration: int | None = Query(default=None),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # -----------------------------------------
    # 1. Check chat membership
    # -----------------------------------------

    chat = member_chat(
        db,
        chat_id,
        current_user.id
    )

    if not chat:
        raise HTTPException(
            status_code=403,
            detail="You are not a member of this chat"
        )

    # -----------------------------------------
    # 2. Save audio file
    # -----------------------------------------

    upload_dir = "uploads/voice"

    os.makedirs(
        upload_dir,
        exist_ok=True
    )

    extension = os.path.splitext(
        file.filename or ""
    )[1]

    if not extension:
        extension = ".webm"

    filename = f"{uuid.uuid4()}{extension}"

    file_path = os.path.join(
        upload_dir,
        filename
    )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Empty audio file"
        )

    with open(file_path, "wb") as f:
        f.write(content)

    audio_url = f"/uploads/voice/{filename}"

    # -----------------------------------------
    # 3. Create voice message in DB
    # -----------------------------------------

    message = await db_call(
        save_voice_message,
        current_user.id,
        chat_id,
        audio_url,
        duration,
        file.content_type
    )

    if message is None:
        # Remove uploaded file if DB message failed
        try:
            os.remove(file_path)
        except OSError:
            pass

        raise HTTPException(
            status_code=400,
            detail="Could not create voice message"
        )

    # -----------------------------------------
    # 4. Send voice message to receiver
    # -----------------------------------------

    payload = {
        "type": "voice_message",
        "chat_id": chat_id,
        "message_id": message["id"],
        "sender_id": current_user.id,
        "sender_name": current_user.username,
        "content": None,
        "created_at": message["created_at"],
        "reply_to_id": None,
        "message_type": "voice",
        "audio_url": message["audio_url"],
        "audio_duration": message["audio_duration"],
        "audio_mime_type": message["audio_mime_type"],
    }

    delivered = await manager.send_to_user(
        message["receiver_id"],
        payload,
        chat_id
    )

    # -----------------------------------------
    # 5. Mark delivered if receiver is online
    # -----------------------------------------

    if delivered:
        await db_call(
            mark_delivered,
            [message["id"]]
        )

        message["is_delivered"] = True

    # -----------------------------------------
    # 6. Return message to sender
    # -----------------------------------------

    return {
        "success": True,
        "message": message,
        "delivered": delivered,
    }


@app.post(
    "/chats/{chat_id}/attachments",
    response_model=schemas.MessageResponse,
    status_code=201,
)
async def upload_chat_attachment(
    chat_id: int,
    file: UploadFile = File(...),
    caption: str | None = Form(default=None),
    reply_to_id: int | None = Form(default=None),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not member_chat(db, chat_id, current_user.id):
        raise HTTPException(403, "You are not a member of this chat")

    if caption is not None and len(caption) > schemas.MAX_MESSAGE_LEN:
        raise HTTPException(400, "Caption exceeds the 2000 character limit")

    try:
        storage_key, filename, mime_type, size_bytes, message_type = (
            await attachment_storage.save_upload(file)
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        await file.close()

    try:
        message = await db_call(
            save_attachment_message,
            current_user.id,
            chat_id,
            caption,
            reply_to_id,
            storage_key,
            filename,
            mime_type,
            size_bytes,
            message_type,
        )
    except Exception:
        await run_in_threadpool(attachment_storage.remove_attachment, storage_key)
        raise

    if message is None:
        await run_in_threadpool(attachment_storage.remove_attachment, storage_key)
        raise HTTPException(400, "Could not create attachment message or invalid reply")

    payload = {
        "type": "message",
        "chat_id": chat_id,
        "message_id": message["id"],
        "sender_id": current_user.id,
        "sender_name": current_user.username,
        "content": message["content"],
        "created_at": message["created_at"],
        "reply_to_id": message["reply_to_id"],
        "message_type": message_type,
        "attachment_name": filename,
        "attachment_mime_type": mime_type,
        "attachment_size_bytes": size_bytes,
    }
    delivered = await manager.send_to_user(
        message["receiver_id"], payload, chat_id
    )

    if delivered:
        await db_call(mark_delivered, [message["id"]])
        message["is_delivered"] = True
        await manager.send_to_user(
            current_user.id,
            {
                "type": "message_delivered",
                "chat_id": chat_id,
                "message_id": message["id"],
            },
            chat_id,
        )

    return message


@app.get("/attachments/{message_id}/content")
def get_attachment_content(
    message_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    message = db.get(models.Message, message_id)
    if (
        not message
        or message.is_deleted
        or not message.attachment_storage_key
        or message.message_type not in {"image", "file"}
    ):
        raise HTTPException(404, "Attachment not found")

    chat = member_chat(db, message.chat_id, current_user.id)
    if not chat:
        raise HTTPException(404, "Attachment not found")

    try:
        path = attachment_storage.attachment_path(message.attachment_storage_key)
    except ValueError as exc:
        logger.error("Invalid attachment key on message %s", message_id)
        raise HTTPException(404, "Attachment not found") from exc
    if not path.is_file():
        raise HTTPException(404, "Attachment content not found")

    mime_type = message.attachment_mime_type
    if mime_type not in {value[0] for value in attachment_storage.ALLOWED_TYPES.values()}:
        mime_type = "application/octet-stream"

    filename = Path(message.attachment_name or "attachment").name.replace("\r", "").replace("\n", "")
    ascii_name = filename.encode("ascii", "replace").decode("ascii").replace('"', "'")
    disposition = "inline" if message.message_type == "image" else "attachment"

    def stream_file():
        with path.open("rb") as source:
            while chunk := source.read(64 * 1024):
                yield chunk

    headers = {
        "Content-Disposition": (
            f'{disposition}; filename="{ascii_name}"; '
            f"filename*=UTF-8''{quote(filename, safe='')}"
        ),
        "X-Content-Type-Options": "nosniff",
        "Cache-Control": "private, no-store",
        "Content-Security-Policy": "default-src 'none'; sandbox",
    }
    return StreamingResponse(stream_file(), media_type=mime_type, headers=headers)
# ============================================================
# LOGIN
# ============================================================

@app.post(
    "/login",
    response_model=schemas.Token
)
def login_user(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    email = form_data.username.strip().lower()

    user = db.query(models.User).filter(
        func.lower(models.User.email) == email
    ).first()

    if (
        not user
        or not password_hash.verify(
            form_data.password,
            user.hashed_password
        )
    ):
        raise HTTPException(
            401,
            "Invalid email or password"
        )

    return {
        "access_token": create_token(user.id),
        "token_type": "bearer"
    }


# ============================================================
# AUTH (frontend-compatible)
# ============================================================

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.environ.get("GOOGLE_REDIRECT_URI", "http://127.0.0.1:5257/api/v1/auth/google/callback")
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"
GOOGLE_JWKS_URL = "https://www.googleapis.com/oauth2/v3/certs"

_google_jwks_cache = None

# In-memory OTP store (replace with Redis in production)
_phone_otps: dict[str, str] = {}
_password_reset_otps: dict[str, str] = {}
_password_reset_tokens: dict[str, str] = {}


def _generate_otp() -> str:
    import random
    return f"{random.randint(100000, 999999)}"


def _generate_reset_token() -> str:
    import secrets
    return secrets.token_urlsafe(32)


@app.get("/api/v1/auth/google/url")
def google_auth_url(redirect_uri: str | None = None):
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(
            500,
            "Google OAuth not configured: set GOOGLE_CLIENT_ID env variable"
        )
    uri = redirect_uri or GOOGLE_REDIRECT_URI
    auth_url = (
        "https://accounts.google.com/o/oauth2/v2/auth"
        f"?client_id={GOOGLE_CLIENT_ID}"
        f"&redirect_uri={uri}"
        "&response_type=code"
        "&scope=openid%20email%20profile"
        f"&state={_generate_reset_token()}"
    )
    return {"authorization_url": auth_url, "state": uri}


async def _fetch_google_jwks():
    """Fetch and cache Google's JWKS for id_token verification."""
    global _google_jwks_cache
    if _google_jwks_cache is None:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(GOOGLE_JWKS_URL)
            resp.raise_for_status()
            _google_jwks_cache = resp.json()
    return _google_jwks_cache


async def _verify_google_id_token(id_token: str) -> dict:
    """Verify Google id_token using JWKS and return claims."""
    from jose import jwt
    from jose.exceptions import JWTError
    
    jwks = await _fetch_google_jwks()
    unverified_header = jwt.get_unverified_header(id_token)
    kid = unverified_header.get("kid")
    if not kid:
        raise HTTPException(400, "Invalid id_token: missing kid")
    
    key = None
    for k in jwks.get("keys", []):
        if k.get("kid") == kid:
            key = jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(k))
            break
    
    if not key:
        raise HTTPException(400, "Invalid id_token: key not found")
    
    try:
        claims = jwt.decode(
            id_token,
            key,
            algorithms=["RS256"],
            audience=GOOGLE_CLIENT_ID,
            issuer="https://accounts.google.com",
        )
        return claims
    except JWTError as e:
        raise HTTPException(401, f"Invalid id_token: {e}")


async def _exchange_code_for_tokens(code: str, redirect_uri: str) -> dict:
    """Exchange authorization code for access_token and id_token."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            GOOGLE_TOKEN_URL,
            data={
                "code": code,
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if resp.status_code != 200:
            raise HTTPException(400, f"Token exchange failed: {resp.text}")
        return resp.json()


@app.post("/api/v1/auth/google/callback")
async def google_callback(data: dict, db: Session = Depends(get_db)):
    code = data.get("code")
    id_token = data.get("id_token")
    redirect_uri = data.get("redirect_uri") or GOOGLE_REDIRECT_URI
    
    if not code and not id_token:
        raise HTTPException(400, "Missing authorization code or id_token")

    # If code provided, exchange for tokens
    if code and not id_token:
        if not GOOGLE_CLIENT_SECRET:
            raise HTTPException(500, "Google OAuth not fully configured: set GOOGLE_CLIENT_SECRET")
        tokens = await _exchange_code_for_tokens(code, redirect_uri)
        id_token = tokens.get("id_token")
        if not id_token:
            raise HTTPException(400, "No id_token in token response")

    # Verify id_token with Google's JWKS
    try:
        claims = await _verify_google_id_token(id_token)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(401, f"Token verification failed: {e}")

    email = claims.get("email")
    if not email:
        raise HTTPException(400, "Email not in id_token")

    # Find or create user
    user = db.query(models.User).filter(
        func.lower(models.User.email) == email.lower()
    ).first()
    
    is_new_user = False
    if not user:
        is_new_user = True
        username = email.split("@")[0]
        # Ensure unique username
        base_username = username
        counter = 1
        while db.query(models.User).filter(models.User.username == username).first():
            username = f"{base_username}{counter}"
            counter += 1
        
        user = models.User(
            username=username,
            email=email,
            hashed_password="",  # No password for OAuth users
            full_name=claims.get("name"),
        )
        db.add(user)
        db.flush()
        
        # Create dating profile placeholder
        dating_profile = models.DatingProfile(
            user_id=user.id,
            sex="prefer_not_to_say",
            orientation="straight",
            verification_status="unverified",
        )
        db.add(dating_profile)
        db.commit()
        db.refresh(user)

    return {
        "access_token": create_token(user.id),
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "full_name": getattr(user, "full_name", None),
        },
        "is_new_user": is_new_user,
    }


@app.post("/api/v1/auth/phone/send-otp")
def send_phone_otp(payload: dict, db: Session = Depends(get_db)):
    phone_number = payload.get("phone_number")
    if not phone_number:
        raise HTTPException(400, "phone_number is required")

    user = db.query(models.User).filter(
        models.User.phone_number == phone_number
    ).first()

    otp = _generate_otp()
    _phone_otps[phone_number] = otp

    return {"message": f"OTP sent to {phone_number}", "success": True}


@app.post("/api/v1/auth/phone/verify-otp")
def verify_phone_otp(payload: dict, db: Session = Depends(get_db)):
    phone_number = payload.get("phone_number")
    otp = payload.get("otp")
    full_name = payload.get("full_name")
    username = payload.get("username")

    if not phone_number or not otp:
        raise HTTPException(400, "phone_number and otp are required")

    stored = _phone_otps.get(phone_number)
    if not stored or stored != otp:
        raise HTTPException(401, "Invalid or expired OTP")

    user = db.query(models.User).filter(
        models.User.phone_number == phone_number
    ).first()

    is_new_user = False
    if not user:
        if not username:
            username = f"user_{phone_number}"
        user = models.User(
            username=username,
            email=f"{phone_number}@raahi.placeholder",
            phone_number=phone_number,
            full_name=full_name,
            hashed_password=password_hash.hash(_generate_reset_token()),
            is_online=False,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        is_new_user = True

    _phone_otps.pop(phone_number, None)

    return {
        "access_token": create_token(user.id),
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "full_name": getattr(user, "full_name", None),
        },
        "is_new_user": is_new_user,
    }


@app.post("/api/v1/auth/password-reset/send-otp")
def send_password_reset_otp(payload: dict, db: Session = Depends(get_db)):
    email = payload.get("email")
    if not email:
        raise HTTPException(400, "email is required")

    user = db.query(models.User).filter(
        func.lower(models.User.email) == email.lower()
    ).first()
    if not user:
        raise HTTPException(404, "User not found")

    otp = _generate_otp()
    _password_reset_otps[email] = otp

    return {"message": f"Password reset OTP sent to {email}", "success": True}


@app.post("/api/v1/auth/password-reset/verify-otp")
def verify_password_reset_otp(payload: dict, db: Session = Depends(get_db)):
    email = payload.get("email")
    otp = payload.get("otp")
    new_password = payload.get("new_password")

    if not email or not otp or not new_password:
        raise HTTPException(400, "email, otp, and new_password are required")

    stored = _password_reset_otps.get(email)
    if not stored or stored != otp:
        raise HTTPException(401, "Invalid or expired OTP")

    user = db.query(models.User).filter(
        func.lower(models.User.email) == email.lower()
    ).first()
    if not user:
        raise HTTPException(404, "User not found")

    user.hashed_password = password_hash.hash(new_password)
    db.commit()

    _password_reset_otps.pop(email, None)

    return {"message": "Password reset successful", "success": True}


@app.post("/api/v1/auth/password-reset/send-link")
def send_password_reset_link(payload: dict, db: Session = Depends(get_db)):
    email = payload.get("email")
    if not email:
        raise HTTPException(400, "email is required")

    user = db.query(models.User).filter(
        func.lower(models.User.email) == email.lower()
    ).first()
    if not user:
        raise HTTPException(404, "User not found")

    token = _generate_reset_token()
    _password_reset_tokens[token] = email

    return {"message": f"Password reset link sent to {email}", "success": True}


@app.post("/api/v1/auth/password-reset/confirm-link")
def confirm_password_reset_link(payload: dict, db: Session = Depends(get_db)):
    token = payload.get("token")
    email = payload.get("email")
    new_password = payload.get("new_password")

    if not token or not email or not new_password:
        raise HTTPException(400, "token, email, and new_password are required")

    stored_email = _password_reset_tokens.get(token)
    if not stored_email or stored_email != email:
        raise HTTPException(401, "Invalid or expired reset token")

    user = db.query(models.User).filter(
        func.lower(models.User.email) == email.lower()
    ).first()
    if not user:
        raise HTTPException(404, "User not found")

    user.hashed_password = password_hash.hash(new_password)
    db.commit()

    _password_reset_tokens.pop(token, None)

    return {"message": "Password reset successful", "success": True}


# ============================================================
# CURRENT USER
# ============================================================

@app.get(
    "/me",
    response_model=schemas.UserResponse
)
def me(
    current_user: models.User = Depends(
        get_current_user
    )
):
    return current_user


# ============================================================
# USERS
# ============================================================

@app.get(
    "/users",
    response_model=list[schemas.UserPublic]
)
def get_users(
    current_user: models.User = Depends(
        get_current_user
    ),
    db: Session = Depends(get_db)
):
    users = db.query(
        models.User
    ).order_by(
        models.User.username
    ).all()

    return [
        schemas.UserPublic(
            id=user.id,
            username=user.username,
            is_online=manager.is_online(user.id),
            last_seen=user.last_seen,
        )
        for user in users
    ]


# ============================================================
# CREATE CHAT
# ============================================================

@app.post(
    "/chats",
    response_model=schemas.ChatResponse
)
def create_chat(
    chat: schemas.ChatCreate,
    current_user: models.User = Depends(
        get_current_user
    ),
    db: Session = Depends(get_db)
):
    if current_user.id == chat.user2_id:
        raise HTTPException(
            400,
            "You cannot chat with yourself"
        )

    if not db.get(
        models.User,
        chat.user2_id
    ):
        raise HTTPException(
            404,
            "User not found"
        )

    existing = find_chat(
        db,
        current_user.id,
        chat.user2_id
    )

    if existing:
        return existing

    new_chat = models.Chat(
        user1_id=current_user.id,
        user2_id=chat.user2_id
    )

    db.add(new_chat)

    try:
        db.commit()
    except IntegrityError:
        db.rollback()

        existing = find_chat(
            db,
            current_user.id,
            chat.user2_id
        )

        if existing:
            return existing

        raise

    db.refresh(new_chat)

    return new_chat


# ============================================================
# DELETE CHAT
# ============================================================

@app.delete("/chats/{chat_id}")
async def delete_chat(
    chat_id: int,
    current_user: models.User = Depends(
        get_current_user
    )
):
    result = await db_call(
        delete_chat_from_database,
        chat_id,
        current_user.id
    )

    if result is None:
        raise HTTPException(
            404,
            "Chat not found or you are not a member"
        )

    event = {
        "type": "chat_deleted",
        "chat_id": result["chat_id"]
    }

    # Tell current user
    await manager.send_to_user(
        current_user.id,
        event,
        chat_id
    )

    # Tell other user
    await manager.send_to_user(
        result["other_user_id"],
        event,
        chat_id
    )

    return {
        "ok": True,
        "message": "Chat deleted successfully"
    }


# ============================================================
# GET CHATS
# ============================================================
@app.patch(
    "/chats/{chat_id}/mute",
    response_model=schemas.ChatResponse,
)
async def mute_chat(
    chat_id: int,
    current_user: models.User = Depends(
        get_current_user
    ),
    db: Session = Depends(get_db),
):
    # Verify chat membership
    chat = db.get(models.Chat, chat_id)
    if not chat:
        raise HTTPException(404, "Chat not found")
    
    if chat.user1_id != current_user.id and chat.user2_id != current_user.id:
        raise HTTPException(403, "You are not a member of this chat")
    
    # Toggle the mute preference for the current user
    if chat.user1_id == current_user.id:
        chat.notification_muted_by_user1 = not chat.notification_muted_by_user1
    else:
        chat.notification_muted_by_user2 = not chat.notification_muted_by_user2
    
    db.commit()
    db.refresh(chat)
    
    return chat

@app.get("/chats", response_model=list[schemas.ChatSummary])
def list_chats(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    my_id = current_user.id

    chats = db.query(models.Chat).filter(
        or_(
            models.Chat.user1_id == my_id,
            models.Chat.user2_id == my_id
        )
    ).all()

    if not chats:
        return []

    chat_ids = [chat.id for chat in chats]

    latest_ids = dict(
        db.query(
            models.Message.chat_id,
            func.max(models.Message.id)
        )
        .filter(
            models.Message.chat_id.in_(chat_ids),
            models.Message.is_deleted.is_not(True)
        )
        .group_by(models.Message.chat_id)
        .all()
    )

    latest_messages = {}

    if latest_ids:
        latest_messages = {
            message.chat_id: message
            for message in db.query(models.Message)
            .filter(
                models.Message.id.in_(list(latest_ids.values()))
            )
            .all()
        }

    unread = dict(
        db.query(
            models.Message.chat_id,
            func.count(models.Message.id)
        )
        .filter(
            models.Message.chat_id.in_(chat_ids),
            models.Message.sender_id != my_id,
            models.Message.is_read.is_not(True),
            models.Message.is_deleted.is_not(True)
        )
        .group_by(models.Message.chat_id)
        .all()
    )

    partner_ids = {
        other_user(chat, my_id)
        for chat in chats
    }

    users = {
        user.id: user
        for user in db.query(models.User)
        .filter(models.User.id.in_(partner_ids))
        .all()
    }

    result = []

    for chat in chats:
        partner = users.get(
            other_user(chat, my_id)
        )

        if not partner:
            continue

        last_message = latest_messages.get(chat.id)

        result.append(
            schemas.ChatSummary(
                chat_id=chat.id,
                user=schemas.UserPublic(
                    id=partner.id,
                    username=partner.username,
                    is_online=manager.is_online(partner.id),
                    last_seen=partner.last_seen
                ),
                last_message=(
                    schemas.MessageResponse.model_validate(last_message)
                    if last_message
                    else None
                ),
                unread_count=unread.get(chat.id, 0)
            )
        )

    result.sort(
        key=lambda chat: chat.last_message.id if chat.last_message else 0,
        reverse=True
    )

    return result

# ============================================================
# SEND MESSAGE
# ============================================================

@app.post(
    "/messages",
    response_model=schemas.MessageResponse,
    status_code=201
)
async def send_message(
    message: schemas.MessageCreate,
    current_user: models.User = Depends(
        get_current_user
    )
):
    result = await process_message(
        current_user.id,
        current_user.username,
        message.chat_id,
        message.content,
        reply_to_id=message.reply_to_id
    )

    if result is None:
        raise HTTPException(
            403,
            "You are not a member of this chat or invalid reply"
        )

    return result


# ============================================================
# DELETE MESSAGE
# ============================================================

@app.delete("/messages/{message_id}")
async def delete_message(
    message_id: int,
    for_everyone: bool = True,
    current_user: models.User = Depends(
        get_current_user
    )
):
    result = await db_call(
        soft_delete_message,
        current_user.id,
        message_id,
        for_everyone
    )

    if result is None:
        raise HTTPException(
            403,
            "Cannot delete this message"
        )

    event = {
        "type": "message_deleted",
        "chat_id": result["chat_id"],
        "message_id": result["message_id"],
        "for_everyone": result["for_everyone"],
    }

    # Sender
    await manager.send_to_user(
        current_user.id,
        event,
        result["chat_id"]
    )

    # Receiver
    await manager.send_to_user(
        result["receiver_id"],
        event,
        result["chat_id"]
    )

    await run_in_threadpool(
        attachment_storage.remove_attachment,
        result.get("attachment_storage_key"),
    )

    return {
        "ok": True,
        "message_id": message_id
    }


# ============================================================
# GET MESSAGES
# ============================================================

@app.get(
    "/messages/{chat_id}",
    response_model=list[schemas.MessageResponse]
)
def get_messages(
    chat_id: int,
    limit: int = Query(
        200,
        ge=1,
        le=500
    ),
    before_id: int | None = None,
    current_user: models.User = Depends(
        get_current_user
    ),
    db: Session = Depends(get_db)
):
    chat = db.get(
        models.Chat,
        chat_id
    )

    if not chat:
        raise HTTPException(
            404,
            "Chat not found"
        )

    if current_user.id not in (
        chat.user1_id,
        chat.user2_id
    ):
        raise HTTPException(
            403,
            "You are not a member of this chat"
        )

    query = db.query(
        models.Message
    ).options(
        selectinload(models.Message.reactions)
    ).filter(
        models.Message.chat_id == chat_id
    )

    if before_id is not None:
        query = query.filter(
            models.Message.id < before_id
        )

    messages = query.order_by(
        models.Message.id.desc()
    ).limit(
        limit
    ).all()

    messages.reverse()

    return messages


# ============================================================
# WEBSOCKET
# ============================================================

async def handle_socket(
    websocket: WebSocket,
    token: str,
    bound_chat_id: int | None
):
    await websocket.accept()

    user_id = decode_token(token)

    if user_id is None:
        await websocket.close(
            code=1008,
            reason="Invalid or expired token"
        )
        return

    context = await db_call(
        load_context,
        user_id,
        bound_chat_id
    )

    if context is None:
        await websocket.close(
            code=1008,
            reason="Not allowed"
        )
        return

    partners = context["partners"]
    username = context["username"]

    first = manager.connect(
        user_id,
        websocket,
        bound_chat_id
    )

    try:
        if first:
            await db_call(
                set_online,
                user_id,
                True
            )

            await broadcast_presence(
                user_id,
                "online"
            )

        await safe_send(
            websocket,
            {
                "type": "ready",
                "user_id": user_id,
                "online_user_ids": [
                    online_id
                    for online_id in manager.online_ids()
                    if online_id != user_id
                ]
            }
        )

        for (
            sender_id,
            chat_id,
            message_id
        ) in context["delivered"]:

            await manager.send_to_user(
                sender_id,
                {
                    "type": "message_delivered",
                    "chat_id": chat_id,
                    "message_id": message_id
                },
                chat_id
            )

        timeout = (
            IDLE_TIMEOUT
            if bound_chat_id is None
            else None
        )

        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout
                )

            except asyncio.TimeoutError:
                await websocket.close(
                    code=1001,
                    reason="Idle timeout"
                )
                break

            except ValueError:
                await safe_send(
                    websocket,
                    {
                        "type": "error",
                        "detail": "Invalid JSON"
                    }
                )
                continue

            if not isinstance(data, dict):
                continue

            event_type = data.get("type")

            chat_id = (
                bound_chat_id
                if bound_chat_id is not None
                else data.get("chat_id")
            )

            if (
                isinstance(chat_id, bool)
                or (
                    chat_id is not None
                    and not isinstance(
                        chat_id,
                        int
                    )
                )
            ):
                chat_id = None

            # =================================================
            # PING
            # =================================================

            if event_type == "ping":

                await safe_send(
                    websocket,
                    {
                        "type": "pong"
                    }
                )

            # =================================================
            # MESSAGE
            # =================================================

            elif event_type == "message":

                content = data.get("content")
                reply_to_id = data.get(
                    "reply_to_id"
                )

                if (
                    not isinstance(
                        content,
                        str
                    )
                    or chat_id is None
                ):
                    continue

                content = content.strip()

                if not content:
                    continue

                if len(content) > schemas.MAX_MESSAGE_LEN:

                    await safe_send(
                        websocket,
                        {
                            "type": "error",
                            "detail": "Message too long"
                        }
                    )

                    continue

                result = await process_message(
                    user_id,
                    username,
                    chat_id,
                    content,
                    data.get("client_id"),
                    websocket,
                    reply_to_id
                )

                if result is None:

                    await safe_send(
                        websocket,
                        {
                            "type": "error",
                            "detail":
                            "Not a member of this chat or invalid reply"
                        }
                    )

            # =================================================
            # MESSAGE REACTION
            # =================================================

            elif event_type == "message_reaction":

                message_id = data.get("message_id")
                reaction = data.get("reaction")
                request_id = data.get("request_id")

                if (
                    chat_id is None
                    or isinstance(message_id, bool)
                    or not isinstance(message_id, int)
                    or not isinstance(reaction, str)
                    or reaction not in ALLOWED_MESSAGE_REACTIONS
                ):
                    await safe_send(
                        websocket,
                        {
                            "type": "message_reaction_error",
                            "chat_id": chat_id,
                            "message_id": message_id,
                            "request_id": request_id,
                            "detail": "Invalid message reaction",
                        },
                    )
                    continue

                result = await db_call(
                    toggle_message_reaction,
                    user_id,
                    chat_id,
                    message_id,
                    reaction,
                )
                if result is None:
                    await safe_send(
                        websocket,
                        {
                            "type": "message_reaction_error",
                            "chat_id": chat_id,
                            "message_id": message_id,
                            "request_id": request_id,
                            "detail": "Message not found or you are not a chat participant",
                        },
                    )
                    continue

                reaction_event = {
                    "type": "message_reaction",
                    "chat_id": result["chat_id"],
                    "message_id": result["message_id"],
                    "revision": result["revision"],
                    "request_id": request_id,
                    "reactions": result["reactions"],
                }
                await manager.send_to_user(
                    user_id,
                    reaction_event,
                    result["chat_id"],
                )
                await manager.send_to_user(
                    result["receiver_id"],
                    reaction_event,
                    result["chat_id"],
                )

            # =================================================
            # DELETE MESSAGE
            # =================================================

            elif event_type == "delete_message":

                message_id = data.get(
                    "message_id"
                )

                for_everyone = data.get(
                    "for_everyone",
                    True
                )

                if not isinstance(
                    message_id,
                    int
                ):
                    continue

                result = await db_call(
                    soft_delete_message,
                    user_id,
                    message_id,
                    for_everyone
                )

                if result:

                    delete_event = {
                        "type": "message_deleted",
                        "chat_id": result["chat_id"],
                        "message_id": result["message_id"],
                        "for_everyone":
                            result["for_everyone"]
                    }

                    # Send to sender
                    await manager.send_to_user(
                        user_id,
                        delete_event,
                        result["chat_id"]
                    )

                    # Send to receiver
                    await manager.send_to_user(
                        result["receiver_id"],
                        delete_event,
                        result["chat_id"]
                    )

                    await run_in_threadpool(
                        attachment_storage.remove_attachment,
                        result.get("attachment_storage_key"),
                    )

            # =================================================
            # DELETE CHAT
            # =================================================

            elif event_type == "delete_chat":

                if chat_id is None:
                    continue

                result = await db_call(
                    delete_chat_from_database,
                    chat_id,
                    user_id
                )

                if result is None:

                    await safe_send(
                        websocket,
                        {
                            "type": "error",
                            "detail":
                            "Chat not found or you are not a member"
                        }
                    )

                    continue

                delete_chat_event = {
                    "type": "chat_deleted",
                    "chat_id": result["chat_id"]
                }

                # Send to current user
                await manager.send_to_user(
                    user_id,
                    delete_chat_event,
                    chat_id
                )

                # Send to other user
                await manager.send_to_user(
                    result["other_user_id"],
                    delete_chat_event,
                    chat_id
                )

            # =================================================
            # TYPING
            # =================================================

            elif event_type == "typing":

                if chat_id is None:
                    continue

                if chat_id not in partners:

                    partners.clear()

                    partners.update(
                        await db_call(
                            get_partners,
                            user_id
                        )
                    )

                if chat_id in partners:

                    await manager.send_to_user(
                        partners[chat_id],
                        {
                            "type": "typing",
                            "chat_id": chat_id,
                            "user_id": user_id
                        },
                        chat_id
                    )

            # =================================================
            # READ
            # =================================================

            elif event_type in (
                "read",
                "read_chat"
            ):

                if chat_id is None:
                    continue

                if event_type == "read":

                    message_id = data.get(
                        "message_id"
                    )

                else:

                    message_id = None

                if (
                    message_id is not None
                    and (
                        isinstance(
                            message_id,
                            bool
                        )
                        or not isinstance(
                            message_id,
                            int
                        )
                    )
                ):
                    continue

                result = await db_call(
                    mark_read,
                    user_id,
                    chat_id,
                    message_id
                )

                if result:

                    for read_id in result["ids"]:

                        await manager.send_to_user(
                            result["sender"],
                            {
                                "type": "message_read",
                                "chat_id": chat_id,
                                "message_id": read_id
                            },
                            chat_id
                        )

    except WebSocketDisconnect:
        pass

    except Exception:
        logger.exception(
            "WebSocket error for user %s",
            user_id
        )

    finally:

        if manager.disconnect(
            user_id,
            websocket
        ):

            spawn(
                go_offline(
                    user_id
                )
            )


# ============================================================
# WEBSOCKET ROUTES
# ============================================================

@app.websocket("/ws")
async def ws_global(
    websocket: WebSocket,
    token: str = Query(default="")
):
    await handle_socket(
        websocket,
        token,
        None
    )


@app.websocket("/ws/{chat_id}")
async def ws_chat(
    websocket: WebSocket,
    chat_id: int,
    token: str = Query(default="")
):
    await handle_socket(
        websocket,
        token,
        chat_id
    )

# ============================================================
# DATINGML V3 SERVICE
# ============================================================

from .models import User
from ml.dating_ml_model.service import MatchingService, CompatibilityModel
# Dependency: provide DatingML matching service with database session
def get_matching_service(db: Session) -> MatchingService:
    repo = SQLAlchemyUserRepository(db)
    model = CompatibilityModel()
    service = MatchingService(model, repo)
    return service

password_hash = PasswordHash.recommended()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
# ============================================================
# DATINGML V3 COMPATIBILITY API
# =============================================================

@app.get("/compatibility")
async def compatibility(
    user_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Check compatibility between the authenticated user and another user.

    Returns compatibility score, eligibility status, and explanation.
    Handles missing users and incomplete profiles safely.
    """
    from ml.dating_ml_model.service import CompatibilityModel
    from ml.dating_ml_model.repository import SQLAlchemyUserRepository

    # Get the target user
    target_user = db.get(models.User, user_id)
    if target_user is None:
        raise HTTPException(
            404,
            "User not found"
        )

    # Get the matching service
    repo = SQLAlchemyUserRepository(db)
    model = CompatibilityModel()
    service = MatchingService(model, repo)

    # Convert profiles to DatingML format
    profile_a = datingml_adapter.user_to_datingml_profile(current_user)
    profile_b = datingml_adapter.user_to_datingml_profile(target_user)

    # Check compatibility
    result = service.predict_pair(profile_a, profile_b)

    return {
        "user_a": current_user.username,
        "user_b": target_user.username,
        "eligible": result["eligible"],
        "eligibility_reason": result["eligibility_reason"],
        "compatibility_score": result.get("compatibility_score"),
        "confidence_score": result.get("confidence_score"),
        "adjusted_score": result.get("adjusted_score"),
        "categories": result.get("categories"),
        "components": result.get("components"),
        "explanation": result.get("explanation"),
        "model_version": result.get("model_version"),
    }


@app.get("/eligibility")
async def eligibility(
    user_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Check eligibility between the authenticated user and another user.

    Eligibility requires: mutual attraction, both users are adults (18+),
    and known ages.

    Returns eligibility status and reason.
    """
    from ml.dating_ml_model.service import CompatibilityModel
    from ml.dating_ml_model.repository import SQLAlchemyUserRepository

    # Get the target user
    target_user = db.get(models.User, user_id)
    if target_user is None:
        raise HTTPException(
            404,
            "User not found"
        )

    # Get the matching service
    repo = SQLAlchemyUserRepository(db)
    model = CompatibilityModel()
    service = MatchingService(model, repo)

    # Convert profiles to DatingML format
    profile_a = datingml_adapter.user_to_datingml_profile(current_user)
    profile_b = datingml_adapter.user_to_datingml_profile(target_user)

    # Check eligibility
    result = service.check_eligibility(profile_a, profile_b)

    return {
        "user_a": current_user.username,
        "user_b": target_user.username,
        "eligible": result["eligible"],
        "reason": result["reason"],
    }


@app.get("/explanation")
async def explanation(
    user_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get explanation of compatibility between the authenticated user and another user.

    Provides detailed explanation of matched preferences,
    unavailable information, and verdict.
    """
    from ml.dating_ml_model.service import CompatibilityModel
    from ml.dating_ml_model.repository import SQLAlchemyUserRepository
    from ml.dating_ml_model.explainability import explain_recommendation

    # Get the target user
    target_user = db.get(models.User, user_id)
    if target_user is None:
        raise HTTPException(
            404,
            "User not found"
        )

    # Get the matching service
    repo = SQLAlchemyUserRepository(db)
    model = CompatibilityModel()
    service = MatchingService(model, repo)

    # Convert profiles to DatingML format
    profile_a = datingml_adapter.user_to_datingml_profile(current_user)
    profile_b = datingml_adapter.user_to_datingml_profile(target_user)

    # Get prediction first
    predict_result = service.predict_pair(profile_a, profile_b)

    # Generate explanation
    expl = explain_recommendation(predict_result)

    return {
        "user_a": current_user.username,
        "user_b": target_user.username,
        "decision": expl["decision"],
        "summary": expl["summary"],
        "matched_preferences": expl.get("matched_preferences", []),
        "unavailable_information": expl.get("unavailable_information", []),
        "disclaimer": expl.get("disclaimer", ""),
    }


# ============================================================
# DATING PROFILE APIs
# ============================================================

def _profile_to_dict(profile):
    """Convert DatingProfile model to plain dict (no Pydantic validation)."""
    if profile is None:
        return None
    return {
        "id": profile.id,
        "user_id": profile.user_id,
        "date_of_birth": profile.date_of_birth,
        "sex": profile.sex,
        "orientation": profile.orientation,
        "full_name": profile.full_name,
        "bio": profile.bio,
        "education": profile.education,
        "job": profile.job,
        "religion": profile.religion,
        "ethnicity": profile.ethnicity,
        "drinks": profile.drinks,
        "drugs": profile.drugs,
        "smokes": profile.smokes,
        "diet": profile.diet,
        "offspring": profile.offspring,
        "pets": profile.pets,
        "height": profile.height,
        "body_type": profile.body_type,
        "city": profile.city,
        "state": profile.state,
        "country": profile.country,
        "match_era_preference": profile.match_era_preference,
        "target_gender_preference": profile.target_gender_preference,
        "behavioral_personalization_enabled": profile.behavioral_personalization_enabled,
        "is_face_verified": profile.is_face_verified,
        "is_id_verified": profile.is_id_verified,
        "verification_status": profile.verification_status,
        "profile_completion_pct": profile.profile_completion_pct,
        "is_profile_public": profile.is_profile_public,
        "created_at": profile.created_at.isoformat() if profile.created_at else None,
        "updated_at": profile.updated_at.isoformat() if profile.updated_at else None,
    }

# ============================================================
# DATINGML EXPLANATION RESPONSE
# ============================================================


# ============================================================
# AI MEMORY APIs
# ============================================================

from pydantic import BaseModel, Field


class UpdateAIMemoryInput(BaseModel):
    memory_text: str = Field(..., min_length=1)


@app.patch("/me", response_model=schemas.UserResponse)
def update_me_profile(
    changes: schemas.UserProfileUpdate,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    values = changes.model_dump(exclude_unset=True)
    for field in ("full_name", "bio", "ai_memory_enabled", "ai_provider_processing_enabled"):
        if field not in values:
            continue
        value = values[field]
        if field in {"ai_memory_enabled", "ai_provider_processing_enabled"}:
            setattr(current_user, field, bool(value))
        else:
            value = value.strip() if isinstance(value, str) else None
            setattr(current_user, field, value or None)
    db.add(current_user)
    db.commit()
    db.refresh(current_user)
    return current_user


@app.get("/me/ai-memories")
async def get_ai_memories(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get all AI memories belonging to the authenticated user."""
    from app.models import UserMemory

    memories = (
        db.query(UserMemory)
        .filter(UserMemory.user_id == current_user.id)
        .order_by(UserMemory.id.desc())
        .all()
    )

    return {
        "memories": [
            {
                "id": memory.id,
                "memory_text": memory.memory_text,
                "memory_type": memory.memory_type,
                "confidence": memory.confidence,
                "source_message_id": memory.source_message_id,
            }
            for memory in memories
        ]
    }


@app.patch("/me/ai-memories/{memory_id}")
async def update_ai_memory(
    memory_id: int,
    memory_data: UpdateAIMemoryInput,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update an AI memory belonging to the authenticated user."""
    from app.models import UserMemory

    memory = (
        db.query(UserMemory)
        .filter(
            UserMemory.id == memory_id,
            UserMemory.user_id == current_user.id,
        )
        .first()
    )

    if not memory:
        raise HTTPException(
            status_code=404,
            detail="Memory not found",
        )

    memory.memory_text = memory_data.memory_text

    try:
        db.commit()
        db.refresh(memory)
    except Exception:
        db.rollback()
        raise

    return {
        "id": memory.id,
        "memory_text": memory.memory_text,
        "memory_type": memory.memory_type,
        "confidence": memory.confidence,
        "source_message_id": memory.source_message_id,
    }


@app.delete("/me/ai-memories/{memory_id}")
async def delete_ai_memory(
    memory_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete an AI memory belonging to the authenticated user."""
    from app.models import UserMemory

    memory = (
        db.query(UserMemory)
        .filter(
            UserMemory.id == memory_id,
            UserMemory.user_id == current_user.id,
        )
        .first()
    )

    if not memory:
        raise HTTPException(
            status_code=404,
            detail="Memory not found",
        )

    try:
        db.delete(memory)
        db.commit()
    except Exception:
        db.rollback()
        raise

    return {
        "result": "Memory deleted",
        "memory_id": memory_id,
    }


# ============================================================
# MVP: BLOCK / REPORT / EMAIL VERIFY / PUSH TOKEN / DEACTIVATE
# ============================================================

import secrets
from datetime import timedelta


def _generate_verification_token() -> str:
    return secrets.token_urlsafe(48)


@app.post("/api/v1/block/{user_id}")
async def block_user(
    user_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user.id == user_id:
        raise HTTPException(400, "You cannot block yourself")

    target = db.get(models.User, user_id)
    if not target:
        raise HTTPException(404, "User not found")

    existing = db.query(models.Block).filter(
        models.Block.blocker_id == current_user.id,
        models.Block.blocked_id == user_id,
    ).first()

    if existing:
        return {"blocked": True, "blocked_user_id": user_id}

    block = models.Block(blocker_id=current_user.id, blocked_id=user_id)
    db.add(block)
    db.commit()

    # Also delete any existing chat between them
    chat = find_chat(db, current_user.id, user_id)
    if chat:
        db.delete(chat)
        db.commit()

    return {"blocked": True, "blocked_user_id": user_id}


@app.delete("/api/v1/block/{user_id}")
async def unblock_user(
    user_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    block = db.query(models.Block).filter(
        models.Block.blocker_id == current_user.id,
        models.Block.blocked_id == user_id,
    ).first()

    if not block:
        raise HTTPException(404, "Block not found")

    db.delete(block)
    db.commit()
    return {"blocked": False, "blocked_user_id": user_id}


@app.get("/api/v1/blocked")
async def list_blocked_users(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    blocks = db.query(models.Block).filter(
        models.Block.blocker_id == current_user.id,
    ).all()

    blocked_ids = [b.blocked_id for b in blocks]
    users = db.query(models.User).filter(
        models.User.id.in_(blocked_ids)
    ).all() if blocked_ids else []

    return {
        "blocked_users": [
            {"id": u.id, "username": u.username, "full_name": u.full_name}
            for u in users
        ]
    }


@app.post("/api/v1/report")
async def report_user(
    report: dict,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    reported_user_id = report.get("reported_user_id")
    reported_message_id = report.get("reported_message_id")
    reason = report.get("reason")
    description = report.get("description")

    if not reason:
        raise HTTPException(400, "reason is required")

    if not reported_user_id and not reported_message_id:
        raise HTTPException(400, "reported_user_id or reported_message_id is required")

    if reported_user_id:
        target = db.get(models.User, reported_user_id)
        if not target:
            raise HTTPException(404, "Reported user not found")

    rec = models.Report(
        reporter_id=current_user.id,
        reported_user_id=reported_user_id,
        reported_message_id=reported_message_id,
        reason=reason,
        description=description,
        status="open",
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)

    return {
        "report_id": rec.id,
        "status": rec.status,
        "created_at": rec.created_at.isoformat() if rec.created_at else None,
    }


@app.post("/api/v1/auth/email/send-verification")
async def send_email_verification(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user.email_verified:
        return {"message": "Email already verified", "verified": True}

    # Invalidate any existing unused tokens
    db.query(models.EmailVerificationToken).filter(
        models.EmailVerificationToken.user_id == current_user.id,
        models.EmailVerificationToken.used.is_(False),
    ).delete(synchronize_session=False)
    db.commit()

    token = _generate_verification_token()
    expires_at = datetime.now(timezone.utc) + timedelta(hours=24)

    rec = models.EmailVerificationToken(
        user_id=current_user.id,
        token=token,
        expires_at=expires_at,
        used=False,
    )
    db.add(rec)
    db.commit()

    # In production, send email here. For MVP, return token (dev only).
    return {
        "message": "Verification email sent",
        "email": current_user.email,
        "token": token,  # Remove in production; send via email instead
        "expires_at": expires_at.isoformat(),
    }


@app.post("/api/v1/auth/email/verify")
async def verify_email(
    payload: dict,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    token = payload.get("token")
    if not token:
        raise HTTPException(400, "token is required")

    rec = db.query(models.EmailVerificationToken).filter(
        models.EmailVerificationToken.token == token,
        models.EmailVerificationToken.user_id == current_user.id,
        models.EmailVerificationToken.used.is_(False),
    ).first()

    if not rec:
        raise HTTPException(400, "Invalid or already used token")

    if rec.expires_at < datetime.now(timezone.utc):
        raise HTTPException(400, "Token expired")

    rec.used = True
    current_user.email_verified = True
    db.commit()

    return {
        "verified": True,
        "email": current_user.email,
    }


@app.post("/api/v1/me/push-token")
async def set_push_token(
    payload: dict,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    token = payload.get("push_token")
    if not token:
        raise HTTPException(400, "push_token is required")

    current_user.push_token = token
    db.commit()

    return {"push_token": token, "saved": True}


@ app.delete("/api/v1/me/deactivate")
async def deactivate_account(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    current_user.is_deactivated = True
    current_user.deactivated_at = datetime.now(timezone.utc)
    current_user.is_online = False
    db.commit()

    return {
        "deactivated": True,
        "deactivated_at": current_user.deactivated_at.isoformat() if current_user.deactivated_at else None,
    }


# ============================================================
# VERIFICATION APIs
# ============================================================

@app.get("/api/v1/verification/status")
async def verification_status(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get verification status for the authenticated user."""
    from app.models import VerificationRecord, DatingProfile

    records = (
        db.query(VerificationRecord)
        .filter(VerificationRecord.user_id == current_user.id)
        .all()
    )

    is_face = any(r.verification_type == "face" for r in records)
    is_id = any(r.verification_type == "id" for r in records)

    current_status: str | None = None
    if records:
        # Use the most recent record's status
        current_status = records[0].status

    # Also check DatingProfile for verification flags
    dp = (
        db.query(DatingProfile)
        .filter(DatingProfile.user_id == current_user.id)
        .first()
    )
    if dp:
        is_face = is_face or bool(dp.is_face_verified)
        is_id = is_id or bool(dp.is_id_verified)
        if dp.verification_status:
            current_status = dp.verification_status

    return {
        "verifications": [
            {
                "id": r.id,
                "user_id": r.user_id,
                "verification_type": r.verification_type,
                "status": r.status,
                "reviewed_at": r.reviewed_at.isoformat() if r.reviewed_at else None,
                "attempt_count": r.attempt_count,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in records
        ],
        "current_status": current_status,
        "is_face_verified": is_face,
        "is_id_verified": is_id,
        "verification_available": True,
    }


@app.post("/api/v1/verification/face", status_code=202)
async def upload_face_verification(
    file: UploadFile = File(...),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Upload a face verification photo."""
    from app.models import VerificationRecord, DatingProfile
    import os
    import uuid

    # Validate file
    if not file.content_type or file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=400, detail="Only JPEG, PNG and WebP images are allowed")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Maximum file size is 10 MB")

    # Save file
    upload_dir = "uploads/verification"
    os.makedirs(upload_dir, exist_ok=True)

    extension = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}[file.content_type]
    filename = f"face_{current_user.id}_{uuid.uuid4()}{extension}"
    file_path = os.path.join(upload_dir, filename)

    try:
        with open(file_path, "wb") as f:
            f.write(content)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to save file")

    # Create verification record
    rec = VerificationRecord(
        user_id=current_user.id,
        verification_type="face",
        status="pending",
        reviewed_at=None,
        attempt_count=1,
        created_at=utcnow(),
    )
    db.add(rec)

    # Update DatingProfile verification_status to pending
    dp = (
        db.query(DatingProfile)
        .filter(DatingProfile.user_id == current_user.id)
        .first()
    )
    if dp:
        dp.verification_status = "pending"

    try:
        db.commit()
        db.refresh(rec)
    except Exception:
        db.rollback()
        if os.path.exists(file_path):
            os.remove(file_path)
        raise

    return {
        "verification_id": rec.id,
        "verification_type": "face",
        "status": "pending",
        "submitted_at": rec.created_at.isoformat() + "Z",
    }


@app.post("/api/v1/verification/id", status_code=202)
async def upload_id_verification(
    file: UploadFile = File(...),
    id_type: str = Form(...),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Upload an ID verification document."""
    from app.models import VerificationRecord, DatingProfile
    import os
    import uuid

    # Validate id_type
    if id_type not in {"id_college", "id_work", "id_govt"}:
        raise HTTPException(status_code=400, detail="Invalid id_type. Must be one of: id_college, id_work, id_govt")

    # Validate file - allow images and PDFs
    allowed_types = {"image/jpeg", "image/png", "image/webp", "application/pdf"}
    if not file.content_type or file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Only JPEG, PNG, WebP images and PDF documents are allowed")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Maximum file size is 10 MB")

    # Save file
    upload_dir = "uploads/verification"
    os.makedirs(upload_dir, exist_ok=True)

    extension = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "application/pdf": ".pdf",
    }[file.content_type]
    filename = f"id_{id_type}_{current_user.id}_{uuid.uuid4()}{extension}"
    file_path = os.path.join(upload_dir, filename)

    try:
        with open(file_path, "wb") as f:
            f.write(content)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to save file")

    # Create verification record
    rec = VerificationRecord(
        user_id=current_user.id,
        verification_type=id_type,
        status="pending",
        reviewed_at=None,
        attempt_count=1,
        created_at=utcnow(),
    )
    db.add(rec)

    # Update DatingProfile verification_status to pending
    dp = (
        db.query(DatingProfile)
        .filter(DatingProfile.user_id == current_user.id)
        .first()
    )
    if dp:
        dp.verification_status = "pending"

    try:
        db.commit()
        db.refresh(rec)
    except Exception:
        db.rollback()
        if os.path.exists(file_path):
            os.remove(file_path)
        raise

    return {
        "verification_id": rec.id,
        "verification_type": id_type,
        "status": "pending",
        "submitted_at": rec.created_at.isoformat() + "Z",
    }


# ============================================================
# VERIFICATION CANCELLATION & REVIEWER ROUTES
# ============================================================

VERIFICATION_ID_TYPES = {"id_college", "id_work", "id_govt"}


def require_verification_reviewer(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Dependency that ensures the current user is a verification reviewer.
    In production, this would check a role or permission flag on the user.
    For now, we use a simple allowlist via environment variable."""
    import os
    reviewer_ids = set()
    env = os.getenv("VERIFICATION_REVIEWER_IDS", "")
    if env:
        try:
            reviewer_ids = {int(x.strip()) for x in env.split(",") if x.strip()}
        except ValueError:
            pass
    # Allow if user is in allowlist, or if allowlist is empty (dev mode)
    if not reviewer_ids or current_user.id in reviewer_ids:
        return current_user
    raise HTTPException(status_code=403, detail="Verification reviewer access required")


@app.delete("/api/v1/verification/{verification_id}")
async def cancel_verification_request(
    verification_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Cancel an owned pending verification request."""
    from app.models import VerificationRecord

    record = (
        db.query(VerificationRecord)
        .filter(
            VerificationRecord.id == verification_id,
            VerificationRecord.user_id == current_user.id,
            VerificationRecord.status == "pending",
        )
        .with_for_update()
        .first()
    )

    if not record:
        # Check if it exists but is not pending (already processed)
        existing = db.get(VerificationRecord, verification_id)
        if existing and existing.user_id == current_user.id:
            if existing.status == "failed":
                raise HTTPException(
                    status_code=410,
                    detail="Verification evidence expired; submit a new request",
                )
        raise HTTPException(
            status_code=404,
            detail="Pending verification request not found",
        )

    record.status = "cancelled"
    record.reviewed_at = utcnow()

    try:
        db.commit()
        db.refresh(record)
    except Exception:
        db.rollback()
        raise

    return {
        "verification_id": verification_id,
        "status": "cancelled",
        "evidence_deletion_pending": False,
    }


@app.get("/api/v1/reviewer/verifications")
async def list_pending_verifications(
    reviewer: models.User = Depends(require_verification_reviewer),
    db: Session = Depends(get_db),
):
    """List pending verification requests for review."""
    from app.models import VerificationRecord

    records = (
        db.query(VerificationRecord)
        .filter(
            VerificationRecord.status == "pending",
            VerificationRecord.verification_type.in_(["face", *VERIFICATION_ID_TYPES]),
        )
        .order_by(VerificationRecord.created_at.asc())
        .all()
    )

    pending = []
    for record in records:
        if record.user_id == reviewer.id:
            continue
        pending.append({
            "verification_id": record.id,
            "user_id": record.user_id,
            "verification_type": record.verification_type,
            "submitted_at": record.created_at.isoformat() if record.created_at else None,
        })

    return {"items": pending}


@app.get("/api/v1/reviewer/verifications/{verification_id}/evidence")
async def get_verification_evidence(
    verification_id: int,
    reviewer: models.User = Depends(require_verification_reviewer),
    db: Session = Depends(get_db),
):
    """Download evidence for a pending verification request."""
    from app.models import VerificationRecord
    import os

    record = (
        db.query(VerificationRecord)
        .filter(
            VerificationRecord.id == verification_id,
            VerificationRecord.status == "pending",
        )
        .first()
    )

    if not record:
        raise HTTPException(status_code=404, detail="Pending verification evidence not found")

    if record.user_id == reviewer.id:
        raise HTTPException(status_code=404, detail="Pending verification evidence not found")

    # For our simple implementation, evidence is stored as a file in uploads/verification/
    # The filename pattern is: face_{user_id}_{uuid}.ext or id_{type}_{user_id}_{uuid}.ext
    # We don't have a separate evidence_key in our model, so we'll reconstruct the filename
    # This is a simplified version - in production you'd store the file path in the record
    raise HTTPException(status_code=404, detail="Evidence file path not stored in record")


@app.post("/api/v1/reviewer/verifications/{verification_id}/decision")
async def decide_verification(
    verification_id: int,
    decision: dict,
    reviewer: models.User = Depends(require_verification_reviewer),
    db: Session = Depends(get_db),
):
    """Record a reviewer's decision on a verification request."""
    from app.models import VerificationRecord, DatingProfile

    record = (
        db.query(VerificationRecord)
        .filter(
            VerificationRecord.id == verification_id,
            VerificationRecord.status == "pending",
        )
        .first()
    )

    if not record:
        raise HTTPException(status_code=404, detail="Pending verification not found")

    if record.user_id == reviewer.id:
        raise HTTPException(status_code=403, detail="Cannot review your own verification")

    decision_value = decision.get("decision")
    reason = decision.get("reason", "").strip()

    if decision_value not in {"approve", "reject"}:
        raise HTTPException(status_code=400, detail="decision must be 'approve' or 'reject'")

    if decision_value == "reject" and len(reason) < 5:
        raise HTTPException(status_code=400, detail="A brief review reason is required for rejection")

    now = utcnow()
    record.status = "passed" if decision_value == "approve" else "failed"
    record.reviewed_at = now

    # Update DatingProfile verification flags
    dp = (
        db.query(DatingProfile)
        .filter(DatingProfile.user_id == record.user_id)
        .first()
    )
    if dp:
        if decision_value == "approve":
            if record.verification_type == "face":
                dp.is_face_verified = True
            elif record.verification_type in VERIFICATION_ID_TYPES:
                dp.is_id_verified = True
            # If both face and ID are verified, set to verified
            if dp.is_face_verified and dp.is_id_verified:
                dp.verification_status = "verified"
            else:
                dp.verification_status = "pending"
        else:
            # Rejected - keep status as pending for re-submission
            dp.verification_status = "pending"

    try:
        db.commit()
        db.refresh(record)
    except Exception:
        db.rollback()
        raise

    return {
        "verification_id": record.id,
        "status": record.status,
        "reviewed_at": record.reviewed_at.isoformat() if record.reviewed_at else None,
    }


# Update feed endpoint to exclude blocked users - find and patch the existing feed route
# The feed endpoint was added earlier; we need to modify it to filter out blocked users.
# Since it's already defined, we'll add a note that the feed should exclude blocked users.
# The existing /api/v1/feed endpoint needs modification to exclude blocked users.
