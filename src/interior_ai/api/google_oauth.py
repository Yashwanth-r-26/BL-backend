"""Google sign-in, done server-side.

Google will not accept a custom scheme as a redirect URI, and a phone app has
nothing else to offer -- in Expo Go the app's own redirect is
``exp://127.0.0.1:8081/--/...``. Native Google Sign-In sidesteps this with
platform SDKs, but those need custom native code and therefore cannot run in
Expo Go at all.

So the redirect comes **here** instead, to an ordinary https URL that Google is
happy with, and the backend hands the result back to the app across a deep
link. The thing handed over is a single-use code rather than the session token:
a token in a URL survives in logs, shell history and anything that records
where a browser went.

    app  ──► /auth/google/start ──► Google consent
                                      │
         ◄── deep link ?code=… ◄── /auth/google/callback
    app  ──► POST /auth/google/exchange ──► bearer token

Nothing here runs without ``GOOGLE_CLIENT_ID`` and ``GOOGLE_CLIENT_SECRET``.
Absent them every route answers 503 saying so, rather than failing somewhere
inside a redirect where the reason would be invisible.
"""

from __future__ import annotations

import os
import secrets
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
JWKS_URI = "https://www.googleapis.com/oauth2/v3/certs"
ISSUERS = ("https://accounts.google.com", "accounts.google.com")

#: Long enough to survive a slow deep-link hand-off, short enough that a code
#: left in a browser history is worthless by the time anyone finds it.
EXCHANGE_TTL_SECONDS = 120
#: The signed state that rides through Google and back.
STATE_TTL_SECONDS = 600


def client_id() -> str:
    return os.getenv("GOOGLE_CLIENT_ID", "")


def client_secret() -> str:
    return os.getenv("GOOGLE_CLIENT_SECRET", "")


def configured() -> bool:
    return bool(client_id() and client_secret())


def public_base_url() -> str:
    """Where Google should send the browser back to.

    Must match a redirect URI registered on the OAuth client exactly, so it is
    configuration rather than something derived from the request -- a proxy
    rewriting Host would otherwise produce a URL Google rejects.
    """
    return os.getenv("PUBLIC_BASE_URL", "").rstrip("/")


def redirect_uri() -> str:
    return f"{public_base_url()}/auth/google/callback"


# ------------------------------------------------------------------- state


def encode_state(app_redirect: str, secret: str) -> str:
    """Sign the app's deep link so it survives the round trip untampered.

    Google echoes `state` back verbatim, which makes it the only channel for
    remembering where to return to. Signed because an attacker who could edit
    it could point the hand-off at a link of their choosing.
    """
    import jwt

    now = datetime.now(timezone.utc)
    return jwt.encode(
        {
            "redirect": app_redirect,
            "nonce": secrets.token_urlsafe(12),
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(seconds=STATE_TTL_SECONDS)).timestamp()),
        },
        secret,
        algorithm="HS256",
    )


def decode_state(state: str, secret: str) -> str | None:
    """The app redirect carried by a valid state, or None."""
    import jwt

    try:
        payload = jwt.decode(state, secret, algorithms=["HS256"])
    except Exception:
        return None
    target = payload.get("redirect")
    return target if isinstance(target, str) and target else None


def authorisation_url(state: str) -> str:
    return f"{AUTH_ENDPOINT}?" + urlencode(
        {
            "client_id": client_id(),
            "redirect_uri": redirect_uri(),
            "response_type": "code",
            "scope": "openid email profile",
            "state": state,
            # `select_account` so a shared device does not silently sign in
            # whoever used it last.
            "prompt": "select_account",
        }
    )


# ------------------------------------------------------------------ tokens


class GoogleError(Exception):
    """Anything that went wrong talking to Google, in words worth showing."""


def exchange_code(code: str, *, transport=None) -> dict:
    """Trade an authorisation code for Google's token response."""
    payload = {
        "code": code,
        "client_id": client_id(),
        "client_secret": client_secret(),
        "redirect_uri": redirect_uri(),
        "grant_type": "authorization_code",
    }
    if transport is not None:
        return transport(payload)

    import httpx

    try:
        response = httpx.post(TOKEN_ENDPOINT, data=payload, timeout=20.0)
    except Exception as exc:  # pragma: no cover - network
        raise GoogleError(f"could not reach Google: {exc}") from exc
    if response.status_code >= 400:
        raise GoogleError(f"Google refused the code: {response.text[:200]}")
    return response.json()


def verify_id_token(id_token: str, *, verifier=None) -> dict:
    """Validate Google's ID token and return its claims.

    Verified properly -- signature against Google's published keys, audience
    against our own client id, issuer against Google's. Decoding without
    checking those would accept a token minted by anybody for anybody, which
    is the whole attack.
    """
    if verifier is not None:
        return verifier(id_token)

    import jwt
    from jwt import PyJWKClient

    try:
        signing_key = PyJWKClient(JWKS_URI).get_signing_key_from_jwt(id_token)
        claims = jwt.decode(
            id_token,
            signing_key.key,
            algorithms=["RS256"],
            audience=client_id(),
            issuer=list(ISSUERS),
            # Google's clock and ours are not the same clock. Without leeway a
            # token issued a second "in the future" is rejected outright, which
            # would fail intermittently and look like nothing at all.
            leeway=30,
        )
    except Exception as exc:
        raise GoogleError(f"that Google sign-in could not be verified: {exc}") from exc
    return claims


def identity(claims: dict) -> tuple[str, str, str]:
    """``(google_sub, email, display_name)`` from verified claims."""
    sub = str(claims.get("sub") or "")
    email = str(claims.get("email") or "").strip().lower()
    if not sub or not email:
        raise GoogleError("Google did not return an email address for that account")
    if claims.get("email_verified") is False:
        raise GoogleError("that Google account's email address is not verified")
    name = str(claims.get("name") or "").strip()[:80]
    return sub, email, name


# --------------------------------------------------------------- hand-back


def new_exchange_code() -> tuple[str, datetime]:
    return secrets.token_urlsafe(32), datetime.now(timezone.utc) + timedelta(
        seconds=EXCHANGE_TTL_SECONDS
    )


def append_code(app_redirect: str, code: str) -> str:
    """Attach the one-time code to the app's deep link."""
    joiner = "&" if "?" in app_redirect else "?"
    return f"{app_redirect}{joiner}{urlencode({'code': code})}"


def append_error(app_redirect: str, message: str) -> str:
    """Send a failure back to the app rather than into the browser.

    Whatever goes wrong here happens inside an in-app browser the person
    cannot debug: a 400 leaves them staring at raw JSON with no way forward
    but the back gesture. Handing the reason to the app means it can say
    something and let them try again.
    """
    joiner = "&" if "?" in app_redirect else "?"
    return f"{app_redirect}{joiner}{urlencode({'error': message[:300]})}"
