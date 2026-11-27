from datetime import datetime
from typing import Literal, List, Optional

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


class UserUpdate(BaseModel):
    full_name: str | None = Field(default=None, max_length=80)
    bio: str | None = Field(default=None, max_length=500)
    activity_status_enabled: bool | None = Field(default=None)
    ai_memory_enabled: bool | None = Field(default=None)

    @field_validator("full_name", mode="before")
    @classmethod
    def _format_full_name(cls, v: object) -> object:
        if v is None:
            return v
        return v.strip()[:80]

    @field_validator("bio", mode="before")
    @classmethod
    def _format_bio(cls, v: object) -> object:
        if v is None:
            return v
        return v.strip()[:500]


class UserPublic(_FromOrm):
    id: int
    username: str
    is_online: bool = False
    last_seen: datetime | None = None

    _fix_online = field_validator("is_online", mode="before")(_none_to_false)


class UserResponse(UserPublic):
    email: str


class UserProfileUpdate(BaseModel):
    full_name: str | None = Field(default=None, max_length=80)
    bio: str | None = Field(default=None, max_length=500)
    ai_memory_enabled: bool | None = None
    ai_provider_processing_enabled: bool | None = None


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


class MessageReactionResponse(_FromOrm):
    user_id: int
    reaction: str


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
    reactions: list[MessageReactionResponse] = Field(default_factory=list)
    reaction_revision: int = 0

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


# -----------------------------------------------------------------
#  DatingProfileInput – what the UI sends when creating/updating
# -----------------------------------------------------------------
class DatingProfileInput(BaseModel):
    date_of_birth: str | None = Field(default=None, description="ISO date string, e.g. 1995-03-15")
    sex: str = Field(..., pattern="^(male|female|nonbinary|other|prefer_not_to_say)$")
    orientation: str = Field(..., pattern="^(straight|gay|bisexual)$")
    full_name: str | None = Field(default=None, max_length=80)
    bio: str | None = Field(default=None, max_length=500)
    education: str | None = Field(default=None, max_length=120)
    job: str | None = Field(default=None, max_length=120)
    religion: str | None = Field(default=None, max_length=80)
    ethnicity: str | None = Field(default=None, max_length=80)
    drinks: str | None = Field(default=None, max_length=50)
    drugs: str | None = Field(default=None, max_length=50)
    smokes: str | None = Field(default=None, max_length=50)
    diet: str | None = Field(default=None, max_length=50)
    offspring: str | None = Field(default=None, max_length=80)
    pets: str | None = Field(default=None, max_length=80)
    height: float | None = Field(default=None, ge=48, le=96)
    body_type: str | None = Field(default=None, pattern="^(thin|skinny|average|fit|jacked|curvy|a little extra|full figured|overweight)$")
    city: str | None = Field(default=None, max_length=100)
    state: str | None = Field(default=None, max_length=100)
    country: str | None = Field(default=None, max_length=100)
    match_era_preference: str | None = Field(default=None, pattern="^(genz|zillenial|millennial)$")
    target_gender_preference: str | None = Field(default=None, pattern="^(male|female|nonbinary|other|prefer_not_to_say)$")
    behavioral_personalization_enabled: bool | None = Field(default=None)

    class Config:
        from_attributes = True


class DatingProfileSchema(BaseModel):
    id: int
    user_id: int
    date_of_birth: str | None = None
    sex: str | None = None
    orientation: str | None = None
    full_name: str | None = None
    bio: str | None = None
    education: str | None = None
    job: str | None = None
    religion: str | None = None
    ethnicity: str | None = None
    drinks: str | None = None
    drugs: str | None = None
    smokes: str | None = None
    diet: str | None = None
    offspring: str | None = None
    pets: str | None = None
    height: float | None = None
    body_type: str | None = None
    city: str | None = None
    state: str | None = None
    country: str | None = None
    match_era_preference: str | None = None
    target_gender_preference: str | None = None
    behavioral_personalization_enabled: bool = False
    is_face_verified: bool = False
    is_id_verified: bool = False
    verification_status: str = "unverified"

    profile_completion_pct: int
    is_profile_public: bool

    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DatingProfilePublicSchema(BaseModel):
    id: int
    user_id: int
    username: str
    full_name: str | None = None
    age: int | None = None
    city: str | None = None
    state: str | None = None
    country: str | None = None
    bio: str | None = None
    profile_completion_pct: int
    is_profile_public: bool
    is_face_verified: bool
    is_id_verified: bool

    model_config = ConfigDict(from_attributes=True)


class ProfilePhotoSchema(BaseModel):
    id: int
    photo_url: str
    is_primary: bool = False
    display_order: int = 0

    model_config = ConfigDict(from_attributes=True)


class ProfilePhotosReorderRequest(BaseModel):
    photo_ids: list[int]

class MatchItemSchema(BaseModel):
    id: int
    chat_id: int | None = None
    user_a_id: int
    user_b_id: int
    matched_at: datetime
    status: Literal["active", "unmatched", "blocked"]
    compatibility_score: int | None = None
    compatibility_score_status: Literal["legacy_placeholder"]
    created_at: datetime


class MatchesResponseSchema(BaseModel):
    matches: List[MatchItemSchema]
    total: int
    limit: int
    offset: int
    has_more: bool


class VerificationRecordSchema(BaseModel):
    id: int
    user_id: int
    verification_type: str  # "face" | "id"
    status: Literal["pending", "passed", "failed", "cancelled"]
    reviewed_at: datetime | None = None
    attempt_count: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class VerificationStatusSchema(BaseModel):
    verifications: List[VerificationRecordSchema]
    current_status: str | None = None
    is_face_verified: bool
    is_id_verified: bool
    verification_available: bool = True


class VerificationSubmissionSchema(BaseModel):
    verification_id: int
    verification_type: str
    status: Literal["pending"] = "pending"
    submitted_at: datetime = Field(default_factory=datetime.utcnow)


class ProfileCompletionSchema(BaseModel):
    total_fields: int = 22
    filled_fields: int
    completion_pct: float
    missing_fields: List[str] = []


class SwipeInput(BaseModel):
    target_user_id: int
    action: Literal["like", "pass", "superlike"]


class SwipeResponseSchema(BaseModel):
    id: int
    user_id: int
    target_user_id: int
    action: Literal["like", "pass", "superlike"]
    created_at: datetime
    is_match: bool = False


class MyProfileResponse(BaseModel):
    profile: DatingProfileSchema
    completion: ProfileCompletionSchema


class UserPublicProfile(BaseModel):
    id: int
    username: str
    is_profile_public: bool
    is_face_verified: bool
    is_id_verified: bool
    age: int | None = None
    bio: str | None = None
    city: str | None = None
    state: str | None = None
    country: str | None = None
    bio: str | None = None
# -----------------------------------------------------------------
#  DatingProfileInput – what the UI sends when creating/updating
# -----------------------------------------------------------------
