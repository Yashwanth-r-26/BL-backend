"""Saved products, and the OAuth hand-back table.

Both are small enough that a module of their own is mostly about keeping the
catalogue module from growing a third unrelated concern.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from .catalogue import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class FavouriteRow(Base):
    """One saved product, keyed by (account, sku).

    A composite primary key rather than a surrogate id: saving the same
    product twice is not a second row, it is the same fact stated again, and
    the database is the right place to enforce that.
    """

    __tablename__ = "user_favourites"

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True, index=True)
    sku: Mapped[str] = mapped_column(String(128), primary_key=True)
    #: Stamped in Python, not by the database. "Newest first" is this
    #: endpoint's contract, and SQL's CURRENT_TIMESTAMP resolves to the second
    #: on SQLite -- two products saved in the same second tie, and the order
    #: comes out arbitrary. A client-side timestamp carries microseconds.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now, server_default=func.now()
    )


class OAuthExchangeRow(Base):
    """A single-use code handed to the app after a Google sign-in.

    Exists because the token itself must not travel in a deep-link URL. Rows
    are short-lived and marked spent on first use; a self-contained signed
    value could not be, since anything self-contained is replayable until it
    expires.
    """

    __tablename__ = "oauth_exchanges"

    code: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
