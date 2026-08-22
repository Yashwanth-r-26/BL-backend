"""Environment loading.

``os.getenv`` reads only the *process* environment, so a ``.env`` file sitting
in the repo is invisible until something loads it. This module does that once,
early, for every entry point -- the API server, Alembic, and the CLI scripts --
so configuration lives in one file instead of being re-exported into every new
shell.

Precedence, deliberately: **real environment variables win over ``.env``**.
That is what ``load_dotenv`` does by default and it is the right way round --
a value you set explicitly for one run (a different database, a test key)
should not be silently overridden by a file you forgot was there. The reverse
would make debugging miserable.

The search walks upward from the current directory, so running from a
subdirectory still finds the repo's ``.env``.
"""

from __future__ import annotations

import os
from pathlib import Path

_LOADED = False

#: Set ``INTERIOR_AI_SKIP_DOTENV=1`` to ignore any ``.env`` -- useful in CI and
#: containers where configuration arrives purely as real environment variables
#: and a stray file would be a surprise.
SKIP_VAR = "INTERIOR_AI_SKIP_DOTENV"


def find_dotenv(start: Path | None = None) -> Path | None:
    """Nearest ``.env`` at or above ``start`` (default: current directory)."""
    here = (start or Path.cwd()).resolve()
    for directory in (here, *here.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None


def load_env(*, force: bool = False) -> Path | None:
    """Load ``.env`` into the process environment. Idempotent.

    Returns the file that was loaded, or ``None`` when there was nothing to
    load. Never raises: a missing or malformed ``.env`` must not stop the
    application from starting on real environment variables.
    """
    global _LOADED
    if _LOADED and not force:
        return None
    if os.getenv(SKIP_VAR) == "1":
        _LOADED = True
        return None

    path = find_dotenv()
    if path is None:
        _LOADED = True
        return None

    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dependency is declared
        _LOADED = True
        return None

    # override=False: an explicitly-set variable beats the file.
    load_dotenv(path, override=False)
    _LOADED = True
    return path


def describe_env() -> dict[str, object]:
    """Which important settings are present, without revealing their values.

    Reporting whether a key is configured is useful; printing the key is not.
    """
    def status(name: str) -> str:
        value = os.getenv(name)
        if not value:
            return "not set"
        return f"set ({len(value)} chars)"

    return {
        "dotenv_file": str(find_dotenv() or "none found"),
        "DATABASE_URL": status("DATABASE_URL"),
        "GEMINI_API_KEY": status("GEMINI_API_KEY"),
        "GEMINI_MODEL": os.getenv("GEMINI_MODEL", "default"),
        "GEMINI_IMAGE_MODEL": os.getenv("GEMINI_IMAGE_MODEL", "default"),
        "GEMINI_DETECT_MODEL": os.getenv("GEMINI_DETECT_MODEL", "default"),
    }