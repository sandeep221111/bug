from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

EMAIL_PATTERN = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
USERNAME_PATTERN = r"^[A-Za-z0-9_.-]+$"
MAX_MESSAGE_LEN = 2000


class _FromOrm(BaseModel):
    model_config = ConfigDict(from_attributes=True)


def _none_to_false(v):
    return bool(v)


# ---------- users ----------
class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=30, pattern=USERNAME_PATTERN)
    email: str = Field(max_length=254, pattern=EMAIL_PATTERN)
    password: str = Field(min_length=8, max_length=128)


class UserPublic(_FromOrm):
    id: int
    username: str
    is_online: bool = False
    last_seen: datetime | None = None

    _fix_online = field_validator("is_online", mode="before")(_none_to_false)


class UserResponse(UserPublic):
    email: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ---------- chats / messages ----------
class ChatCreate(BaseModel):
    user2_id: int


class ChatResponse(_FromOrm):
    id: int
    user1_id: int
    user2_id: int


class MessageCreate(BaseModel):
    chat_id: int
    content: str = Field(min_length=1, max_length=MAX_MESSAGE_LEN)
    reply_to_id: int | None = None

    @field_validator("content")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Message cannot be empty")
        return v


class MessageResponse(_FromOrm):
    id: int
    chat_id: int
    sender_id: int
    content: str | None = None
    created_at: datetime | None = None
    is_delivered: bool = False
    is_read: bool = False
    is_deleted: bool = False
    reply_to_id: int | None = None

    # Voice message support
    message_type: str = "text"
    audio_url: str | None = None
    audio_duration: int | None = None
    audio_mime_type: str | None = None

    # Photo and document attachment metadata (never includes a storage path).
    attachment_name: str | None = None
    attachment_mime_type: str | None = None
    attachment_size_bytes: int | None = None

    _fix_delivered = field_validator(
        "is_delivered",
        mode="before"
    )(_none_to_false)

    _fix_read = field_validator(
        "is_read",
        mode="before"
    )(_none_to_false)

    _fix_deleted = field_validator(
        "is_deleted",
        mode="before"
    )(_none_to_false)

    _fix_delivered = field_validator("is_delivered", mode="before")(_none_to_false)
    _fix_read = field_validator("is_read", mode="before")(_none_to_false)
    _fix_deleted = field_validator("is_deleted", mode="before")(_none_to_false)


class MessageDelete(BaseModel):
    for_everyone: bool = True


class ChatSummary(BaseModel):
    chat_id: int
    user: UserPublic
    last_message: MessageResponse | None = None
    unread_count: int = 0
