# frontend/auth.py
from __future__ import annotations

import os
import json
import time
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

import streamlit as st
from supabase import Client

# ---------------- Session keys ----------------
SESSION_KEY = "supa_session"        # trimmed session dict kept in memory
EXP_KEY     = "supa_session_exp"    # soft UI expiry timestamp (ISO)
LAST_EMAIL_KEY = "last_email"       # remembered email for prefill

# ---------------- Local token store (dev) ----------------
# In production, prefer HTTP-only cookies or server storage.
TOKEN_STORE_PATH = os.path.expanduser("~/.ccp_auth.json")


def _persist_tokens(
    access_token: Optional[str],
    refresh_token: Optional[str],
    expires_at_unix: Optional[int],
    email: Optional[str] = None,
) -> None:
    """Persist tokens for silent restore across app restarts."""
    if not access_token or not refresh_token:
        return
    data = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "email": email or "",
        "expires_at": int(expires_at_unix or (time.time() + 3600)),
        "saved_at": int(time.time()),
    }
    try:
        with open(TOKEN_STORE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass


def _load_persisted_tokens() -> Optional[Dict[str, Any]]:
    try:
        if os.path.exists(TOKEN_STORE_PATH):
            with open(TOKEN_STORE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return None
    return None


def _clear_persisted_tokens() -> None:
    try:
        if os.path.exists(TOKEN_STORE_PATH):
            os.remove(TOKEN_STORE_PATH)
    except Exception:
        pass


def _remembered_email() -> str:
    if st.session_state.get(LAST_EMAIL_KEY):
        return st.session_state[LAST_EMAIL_KEY]
    data = _load_persisted_tokens() or {}
    return data.get("email", "")


# ---------------- Core session helpers (robust + token storage) ----------------
def _save_session(sess: Any, minutes: int = 120) -> None:
    """
    Persist a trimmed Supabase session in Streamlit session_state
    AND to a small local file for silent restore across app restarts.
    Works for dict- or object-based sessions.
    """
    # Extract user + tokens from either dict or object
    if isinstance(sess, dict):
        u = sess.get("user") or {}
        access_token = sess.get("access_token")
        refresh_token = sess.get("refresh_token")
        # prefer real expires_in if present, else fallback
        expires_in = int(sess.get("expires_in") or 3600)
        user_id = u.get("id")
        user_email = u.get("email")
    else:
        u = getattr(sess, "user", None)
        access_token = getattr(sess, "access_token", None)
        refresh_token = getattr(sess, "refresh_token", None)
        expires_in = int(getattr(sess, "expires_in", 3600))
        if isinstance(u, dict):
            user_id = u.get("id")
            user_email = u.get("email")
        else:
            user_id = getattr(u, "id", None)
            user_email = getattr(u, "email", None)

    trimmed = {
        "user": {"id": user_id, "email": user_email},
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_in": expires_in,
    }
    st.session_state[SESSION_KEY] = trimmed
    # Soft UI expiry; Supabase handles real token expiry/refresh
    st.session_state[EXP_KEY] = (datetime.utcnow() + timedelta(minutes=minutes)).isoformat()

    if user_email:
        st.session_state[LAST_EMAIL_KEY] = user_email

    _persist_tokens(
        access_token,
        refresh_token,
        int(time.time()) + int(expires_in or 3600),
        email=user_email,
    )


def _expired() -> bool:
    exp = st.session_state.get(EXP_KEY)
    if not exp:
        return True
    return datetime.utcnow() >= datetime.fromisoformat(exp)


def get_user() -> Optional[Dict[str, Any]]:
    sess = st.session_state.get(SESSION_KEY)
    if not sess or _expired():
        return None
    return sess.get("user")


def get_user_id() -> Optional[str]:
    user = get_user()
    return user.get("id") if user else None


def get_access_token() -> Optional[str]:
    sess = st.session_state.get(SESSION_KEY) or {}
    return sess.get("access_token")


# ---------------- Silent restore on app load ----------------
def bootstrap_session(supa: Client) -> None:
    """
    If tokens exist in memory or on disk, initialize/refresh the Supabase session.
    Call once in app.py right after creating the client.
    """
    # 1) In-memory
    sess = st.session_state.get(SESSION_KEY) or {}
    at = sess.get("access_token")
    rt = sess.get("refresh_token")
    if at and rt:
        try:
            supa.auth.set_session(at, rt)
            try:
                supa.auth.refresh_session()
            except Exception:
                pass
            return
        except Exception:
            pass  # fall through to disk path

    # 2) Disk
    tok = _load_persisted_tokens()
    if not tok:
        return
    at = tok.get("access_token")
    rt = tok.get("refresh_token")
    if not at or not rt:
        return
    try:
        supa.auth.set_session(at, rt)
        cur = supa.auth.get_session()
        # supabase-py v2 returns object-like session
        if getattr(cur, "user", None):
            normalized = {
                "user": {
                    "id": getattr(cur.user, "id", None),
                    "email": getattr(cur.user, "email", None),
                },
                "access_token": getattr(cur, "access_token", None),
                "refresh_token": getattr(cur, "refresh_token", None),
                "expires_in": int(getattr(cur, "expires_in", 3600)),
            }
            _save_session(normalized)
    except Exception:
        _clear_persisted_tokens()


def sign_out(supa: Client) -> None:
    """Global sign-out (revokes refresh token) + clear local state."""
    try:
        supa.auth.sign_out()
    except Exception:
        pass
    for k in (SESSION_KEY, EXP_KEY, LAST_EMAIL_KEY):
        st.session_state.pop(k, None)
    _clear_persisted_tokens()
    # ensure any parent-controlled modal closes
    st.session_state["auth_open"] = False
    st.rerun()

# ---------------- Email + Password helpers ----------------
def sign_up_password(supa: Client, email: str, password: str) -> bool:
    """
    Create a new user with email+password.
    If email confirmations are disabled in Supabase, returns a session immediately.
    Otherwise, Supabase will send a confirmation email with a redirect back to this app.
    """
    res = supa.auth.sign_up({
        "email": email,
        "password": password,
        "options": {
            "email_redirect_to": "http://localhost:8501"  # IMPORTANT: match your Streamlit URL
        }
    })
    sess = getattr(res, "session", None)
    if sess:
        _save_session(sess)
        # close a parent modal if it exists
        st.session_state["auth_open"] = False
        return True
    st.success("Account created. Check your email to confirm, then sign in.")
    return False


def sign_in_password(supa: Client, email: str, password: str) -> bool:
    """Sign in with email+password and persist the session."""
    res = supa.auth.sign_in_with_password({"email": email, "password": password})
    sess = getattr(res, "session", None)
    if not sess:
        return False
    _save_session(sess)
    # close a parent modal if it exists
    st.session_state["auth_open"] = False
    return True


# ---------------- Modal (popup) version with blur ----------------
try:
    _dialog_fn = st.dialog  # type: ignore[attr-defined]
except AttributeError:
    _dialog_fn = None


def _inject_blur_overlay() -> None:
    st.markdown(
        """
        <style>
          [data-testid="stAppViewContainer"] .main {
            filter: blur(4px);
            pointer-events: none;
            user-select: none;
          }
          [data-testid="stSidebar"] .block-container {
            filter: blur(4px);
            pointer-events: none;
            user-select: none;
          }
          .auth-dim::before {
            content: "";
            position: fixed; inset: 0;
            background: rgba(0,0,0,0.45);
            z-index: 9998;
          }
          [data-testid="stModal"] { z-index: 9999 !important; }
        </style>
        <div class="auth-dim"></div>
        """,
        unsafe_allow_html=True,
    )


def render_auth_modal(supa: Client) -> None:
    """
    Simple email + password auth modal.
    - Sign in with email & password
    - Create account if new
    - Remembers email locally
    """
    if get_user():
        return  # already signed in

    # Darken/blur background
    _inject_blur_overlay()
    title = "Sign in to continue"

    def _ui():
        st.caption("Use your email and password to sign in. Or create an account below.")
        email = st.text_input("Email", key="pw_email", value=_remembered_email())
        password = st.text_input("Password", key="pw_pass", type="password")
        remember = st.checkbox("Remember my email on this device", value=bool(_remembered_email()))

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Sign in", use_container_width=True, disabled=not (email and password)):
                try:
                    ok = sign_in_password(supa, email, password)
                    if ok:
                        if remember:
                            st.session_state[LAST_EMAIL_KEY] = email
                        st.success("Signed in!")
                        st.session_state["auth_open"] = False
                        st.rerun()
                    else:
                        st.error("Invalid email or password.")
                except Exception as e:
                    st.error(f"Sign-in failed: {e}")
        with col2:
            if st.button("Create account", use_container_width=True, disabled=not (email and password)):
                try:
                    ok = sign_up_password(supa, email, password)
                    if ok:
                        if remember:
                            st.session_state[LAST_EMAIL_KEY] = email
                        st.success("Account created & signed in!")
                        st.session_state["auth_open"] = False
                        st.rerun()
                except Exception as e:
                    st.error(f"Sign-up failed: {e}")

    # Show as dialog if supported, else fallback card
    if _dialog_fn:
        @_dialog_fn(title, width="small")  # type: ignore[misc]
        def _dlg():
            _ui()
        _dlg()
    else:
        st.markdown("<div style='height:10vh'></div>", unsafe_allow_html=True)
        card = st.container(border=True)
        with card:
            st.markdown("### " + title)
            _ui()

