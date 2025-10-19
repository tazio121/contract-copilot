# frontend/auth.py
from __future__ import annotations

import os
import json
import time
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

import streamlit as st
from supabase import Client
import streamlit.components.v1 as components

# ---------------- Session keys ----------------
SESSION_KEY = "supa_session"        # trimmed session dict kept in memory
EXP_KEY     = "supa_session_exp"    # soft UI expiry timestamp (ISO)
LAST_EMAIL_KEY = "last_email"       # remembered email for prefill
OTP_VERIFIED_KEY = "otp_verified"

# ---------------- Local token store (dev) ----------------
# In production, prefer HTTP-only cookies or server storage.
TOKEN_STORE_PATH = os.path.expanduser("~/.ccp_auth.json")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:8501")

def render_forgot_password(supa: Client) -> None:
    st.markdown("### Reset your password")
    email_prefill = st.session_state.get("reset_email_prefill", _remembered_email())
    email = st.text_input("Your account email", key="reset_email", value=email_prefill)

    if st.button("Send reset link", use_container_width=True, disabled=not email):
        try:
            # Supabase v2: options={"redirect_to": ...}
            supa.auth.reset_password_for_email(email, options={"redirect_to": FRONTEND_ORIGIN})
            st.success("If that email exists, a reset link has been sent.")
            st.session_state.pop("show_reset_form", None)
        except Exception as e:
            st.error(f"Couldn't send reset link: {e}")

    if st.button("Back to sign in", type="secondary", use_container_width=True):
        st.session_state.pop("show_reset_form", None)
        st.rerun()


def render_new_password(supa: Client) -> None:
    st.markdown("### Set a new password")
    pw1 = st.text_input("New password", type="password", key="npw1")
    pw2 = st.text_input("Confirm new password", type="password", key="npw2")

    if st.button("Update password", use_container_width=True, disabled=not (pw1 and pw2)):
        if pw1 != pw2:
            st.error("Passwords do not match.")
            return
        try:
            # Update password
            supa.auth.update_user({"password": pw1})

            # Hand off a one-time banner for the next screen (belt & braces)
            st.session_state["__pwreset_flash"] = "Your password has been updated. Please sign in."
            try:
                # Replace URL params with just ?flash=pwreset (survives sign-out)
                st.query_params.clear()
                st.query_params["flash"] = "pwreset"
            except Exception:
                st.experimental_set_query_params(flash="pwreset")

            # (Optional) remember email for prefill on sign-in
            if st.session_state.get("pw_email"):
                st.session_state[LAST_EMAIL_KEY] = st.session_state["pw_email"]

            # Stop forcing the reset UI
            st.session_state.pop("show_new_pw", None)

            # Sign out so we land on the Sign-in page
            try:
                supa.auth.sign_out()
            except Exception:
                pass

            # Clear auth session keys (keep LAST_EMAIL_KEY so it can prefill)
            for k in (SESSION_KEY, EXP_KEY):
                st.session_state.pop(k, None)

            # Rerender → app will show Sign-in
            st.rerun()
        except Exception as e:
            st.error(f"Update failed: {e}")


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
    Otherwise, instructs user to sign in manually after confirming.
    """
    # NOTE: Make sure this origin matches your Supabase Auth "Site URL" and the URL you open.
    res = supa.auth.sign_up({
        "email": email,
        "password": password,
        "options": {"email_redirect_to": "http://localhost:8501/?confirmed=1"}
    })
    sess = getattr(res, "session", None)
    if sess:
        _save_session(sess)
        st.session_state["auth_open"] = False
        return True

    # confirmations ON: show one-time hint and refresh
    st.session_state["signup_message"] = True
    st.rerun()   # <-- NOT experimental_rerun


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


# ---------------- Auth modal (container; no overlays) ----------------
def render_auth_modal(supa: Client) -> None:
    """
    Email+password auth modal with clean Supabase recovery flow (token_hash in query).
    Idempotent: verify once, then keep the user on the reset form until submit.
    """

    # default flags
    if "show_reset_form" not in st.session_state:
        st.session_state["show_reset_form"] = False

    # ---- Parse query params (helper + values) ----
    qp = st.query_params

    def _first(key: str):
        v = qp.get(key)
        if isinstance(v, list):
            return v[0] if v else None
        return v

    type_q     = (_first("type") or "").lower()
    token_hash = _first("token_hash")
    next_path  = _first("next") or ""

    # (legacy tokens if present; unused in the new flow)
    at = _first("access_token")
    rt = _first("refresh_token")

    # ---- Recovery: verify ONCE then clean URL (even on reruns) ----
    try:
        if type_q == "recovery" and token_hash:
            if not st.session_state.get(OTP_VERIFIED_KEY):
                # First time: verify and persist the session
                res = supa.auth.verify_otp({"token_hash": token_hash, "type": "recovery"})
                sess = getattr(res, "session", None) or (res.get("session") if isinstance(res, dict) else None)
                if sess:
                    _save_session(sess)
                st.session_state[OTP_VERIFIED_KEY] = True
                st.session_state["show_new_pw"] = True

            # Always clean the URL so future reruns don't see token_hash again
            clean_to = next_path or "/"
            components.html(
                "<script>history.replaceState({}, document.title, '/');</script>",
                height=1,  # non-zero so it renders
            )
    except Exception as e:
        st.warning(f"Recovery link handling issue: {e}")

    # ---- If flagged, render the new-password screen BEFORE any early 'signed-in' return ----
    if st.session_state.get("show_new_pw"):
        render_new_password(supa)
        return

    # ---- Forgot-password view ----
    if st.session_state.get("show_reset_form"):
        render_forgot_password(supa)
        return

    # ---- If already logged in (and not resetting), don't render auth ----
    if get_user():
        return

     # ---- Sign-in UI ----
    title = "Sign in to continue"

    def _ui():
        # Optional one-time notice after signup (if you use it)
        if st.session_state.get("signup_message"):
            st.markdown(
                """
                <div style="
                    background:#0e1117; border:1px solid #2e7d32; color:#d0ffd6;
                    padding:12px 16px; border-radius:6px; margin-bottom:12px;
                    position:relative; font-size:14px;">
                    <strong>Account created</strong><br>
                    Check your inbox and click the confirmation link to activate your account.
                    <span style="position:absolute;top:4px;right:8px;cursor:pointer;color:#9ccc9c;"
                          onclick="this.parentElement.style.display='none'">✖</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.session_state.pop("signup_message", None)

        # ✅ Password-reset banner (one-time)
        msg = st.session_state.pop("__pwreset_flash", None)
        if not msg:
            try:
                v = st.query_params.get("flash")
                if isinstance(v, list):
                    v = v[0] if v else None
                if v == "pwreset":
                    msg = "Your password has been updated. Please sign in."
                    try:
                        del st.query_params["flash"]  # clean the flag
                    except Exception:
                        pass
            except Exception:
                pass
        if msg:
            st.success(msg, icon="✅")

        # --- Header + inputs ---
        st.markdown(f"### {title}")
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
                        components.html("<script>history.replaceState({}, document.title, '/');</script>", height=1)
                        st.rerun()
                    else:
                        st.error("Invalid email or password.")
                except Exception as e:
                    st.error(f"Sign-in failed: {e}")

            if st.button("Forgot Password?", use_container_width=True):
                st.session_state["show_reset_form"] = True
                st.session_state["reset_email_prefill"] = email
                st.rerun()

        with col2:
            if st.button("Create account", use_container_width=True, disabled=not (email and password)):
                try:
                    ok = sign_up_password(supa, email, password)
                    if ok:
                        if remember:
                            st.session_state[LAST_EMAIL_KEY] = email
                        st.success("Account created & signed in!")
                        st.session_state["auth_open"] = False
                        components.html("<script>history.replaceState({}, document.title, '/');</script>", height=1)
                        st.rerun()
                    else:
                        st.session_state["signup_message"] = True
                        if remember:
                            st.session_state[LAST_EMAIL_KEY] = email
                        st.rerun()
                except Exception as e:
                    st.error(f"Sign-up failed: {e}")

    _ui()
    # --------- END INLINE UI ----------