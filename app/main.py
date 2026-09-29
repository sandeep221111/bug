"""Bugzyme backend: REST + WebSocket chat (Reply + Delete)."""

import asyncio
import contextlib
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Request,
    UploadFile,
    File,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from pwdlib import PasswordHash
from sqlalchemy import and_, or_, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from . import models, schemas
from .connection_manager import ConnectionManager
from .database import Base, SessionLocal, engine, get_db

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

@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=engine)

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

    query = db.query(models.Message).filter(
        models.Message.chat_id.in_(
            list(partners) or [0]
        ),
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