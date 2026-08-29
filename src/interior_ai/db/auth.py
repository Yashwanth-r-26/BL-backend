"""Accounts.

The service ran without any notion of a user: every endpoint was public and a
design was addressed only by possession of its session id. That is fine for a
single-operator console and wrong for a phone app, where "my designs" has to
mean something across reinstalls and devices.

Two tables, and deliberately no more:

**users** -- an email, a bcrypt hash, a display name. No roles, no profile, no
verification flow. Adding columns for features that do not exist yet is how a
schema becomes a museum.

**edit_sessions.user_id** -- a nullable owner. Nullable because every session
created before this existed has no owner and must keep working, and because the
console still creates sessions with no user at all. Ownership is therefore a
claim a session *may* carry, not a requirement, and the endpoints treat an
unowned session exactly as they did before.
"""

from __future__ import annotations

import os
import re
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import DateTime, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from .catalogue import Base, JSONType  # noqa: F401 -- shared declarative base


class UserRow(Base):
    """One account."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    # Stored lower-cased and stripped; the unique index is what actually
    # prevents two accounts differing only by capitalisation.
    email: Mapped[str] = mapped_column(String(320), nullable=False, unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(80), nullable=False, default="")
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    active: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


# ------------------------------------------------------------------ hashing

#: bcrypt truncates silently at 72 bytes. Rejecting longer input is better than
#: accepting a password of which only the first 72 bytes are ever checked.
MAX_PASSWORD_BYTES = 72
MIN_PASSWORD_LENGTH = 8


def hash_password(password: str) -> str:
    import bcrypt

    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("ascii")


def verify_password(password: str, hashed: str) -> bool:
    import bcrypt

    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("ascii"))
    except Exception:
        # A malformed stored hash must read as "wrong password", never as a
        # 500 that tells an attacker the row exists.
        return False


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s.]+\.[^@\s]+$")


def normalise_email(email: str) -> str:
    return (email or "").strip().lower()


def validate_email(email: str) -> str | None:
    """Returns a problem, or None when the address is acceptable."""
    if not email:
        return "Enter an email address."
    if len(email) > 320:
        return "That email address is too long."
    if not _EMAIL_RE.match(email):
        return "That does not look like an email address."
    return None


def validate_password(password: str) -> str | None:
    if not password:
        return "Enter a password."
    if len(password) < MIN_PASSWORD_LENGTH:
        return f"Use at least {MIN_PASSWORD_LENGTH} characters."
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        return "That password is too long."
    return None


# ------------------------------------------------------------------- tokens

#: Long-lived on purpose. There is no refresh endpoint and no revocation list;
#: a phone app that logs you out every hour with no way back is worse, for this
#: threat model, than a token that lasts a month.
TOKEN_TTL_DAYS = 30
_ALGORITHM = "HS256"


def token_secret() -> str:
    """The signing key.

    Falls back to a per-process random value when unset, which means tokens
    stop working on restart. That is the correct failure: it is loud in
    development and it refuses to sign anything with a well-known constant in
    production.
    """
    configured = os.getenv("AUTH_SECRET") or os.getenv("SECRET_KEY")
    if configured:
        return configured
    global _EPHEMERAL_SECRET
    try:
        return _EPHEMERAL_SECRET
    except NameError:
        _EPHEMERAL_SECRET = uuid.uuid4().hex + uuid.uuid4().hex
        return _EPHEMERAL_SECRET


def issue_token(user_id: str) -> tuple[str, int]:
    """Returns (token, seconds until expiry)."""
    import jwt

    now = datetime.now(timezone.utc)
    expires = now + timedelta(days=TOKEN_TTL_DAYS)
    payload = {"sub": user_id, "iat": int(now.timestamp()), "exp": int(expires.timestamp())}
    token = jwt.encode(payload, token_secret(), algorithm=_ALGORITHM)
    return token, int((expires - now).total_seconds())


def read_token(token: str) -> str | None:
    """The user id in a valid token, or None. Never raises."""
    import jwt

    try:
        payload = jwt.decode(token, token_secret(), algorithms=[_ALGORITHM])
    except Exception:
        return None
    subject = payload.get("sub")
    return subject if isinstance(subject, str) and subject else None
