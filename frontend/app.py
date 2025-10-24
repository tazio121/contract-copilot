# =============================================================================
# Contract Co-Pilot — Streamlit Frontend (Tab Persistence Fixed)
# =============================================================================
from __future__ import annotations

# ---- Imports ----------------------------------------------------------------
import os, io, json, time, requests
from datetime import datetime
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components
from json import dumps

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8787")

@st.cache_data(ttl=30)  # ping at most once every 30s per session
def ping_backend() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=1.5)
        return r.ok
    except Exception:
        return False
    
 # ==== History: load/save/toggle/delete/add ====================================
from pathlib import Path
HISTORY_PATH = Path("history.json")

def load_history() -> list[dict]:
    try:
        return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []

def _save_history(items: list[dict]) -> None:
    try:
        HISTORY_PATH.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

def add_history_entry(*, kind: str, title: str, payload: dict | str = None, meta: dict = None, fav: bool=False):
    """Append one line to history and keep only the latest 100."""
    items = st.session_state.get("history") or load_history()
    from datetime import datetime
    entry = {
        "type": kind,                      # "text", "text_detailed", "pdf_quick", "pdf_detailed"
        "title": (title or "Untitled").strip(),
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M"),  # short ts for sidebar
        "payload": payload if payload is not None else {},
        "meta": meta or {},
        "fav": bool(fav),
    }
    items.insert(0, entry)
    items = items[:100]
    st.session_state["history"] = items
    _save_history(items)

def toggle_fav(i: int):
    items = st.session_state.get("history", [])
    if 0 <= i < len(items):
        items[i]["fav"] = not items[i].get("fav", False)
        _save_history(items)

def delete_history_item(i: int):
    items = st.session_state.get("history", [])
    if 0 <= i < len(items):
        del items[i]
        st.session_state["history"] = items
        _save_history(items)

# Ensure session has history loaded on first run
st.session_state.setdefault("history", load_history())   

# ---- Branding paths ----
API_BASE = "http://140.238.88.228:8787"
STATIC_BASE = f"{API_BASE}/static"
LOGO_URL = f"{STATIC_BASE}/ccp-logo.png?v=1"
AVATAR_DEFAULT = f"{STATIC_BASE}/avatars/ccp-white-blue.png"

# Sidebar mini-logo CSS
st.markdown(f"""
<style>
[data-testid="stSidebar"] {{
  position: relative;
}}
[data-testid="stSidebar"]::before {{
  content: "";
  position: absolute;
  top: 14px; left: 16px;
  width: 28px; height: 28px;
  background: url('{LOGO_URL}') no-repeat center/contain;
  opacity: .95;
}}
</style>
""", unsafe_allow_html=True)        

# ---- Page Setup --------------------------------------------------------------
st.set_page_config(page_title="Contract Co-Pilot", page_icon="🧾", layout="centered")


# ---- One-shot rerun guard (global) ------------------------------------------
st.session_state.setdefault("_do_rerun", False)
if st.session_state._do_rerun:
    st.session_state._do_rerun = False
    st.rerun()

# ---- Auth (matches your auth.py) --------------------------------------------
from supa import get_supa
from auth import render_auth_modal, get_user_id, bootstrap_session, sign_out



# ---- Static/branding helpers --------------------------------------------------
from pathlib import Path
ASSETS = Path(__file__).parent / "static"  # local fallback, not required

BACKEND_BASE = os.environ.get("BACKEND_BASE_URL", "http://127.0.0.1:8787").rstrip("/")
LOGO_URL = f"{BACKEND_BASE}/static/ccp-logo.png"  # served by FastAPI
FAVICON_URL = f"{BACKEND_BASE}/static/favicon-32.png"  # optional

# Mini “memory” for profile + history (file-based so it survives restarts)
STORE = Path.home() / ".contract-copilot"
STORE.mkdir(exist_ok=True)
PROFILE_FILE = STORE / "profile.json"
HISTORY_FILE = STORE / "history.json"

def _load_json(path: Path, default):
    try:
        return json.loads(path.read_text("utf-8"))
    except Exception:
        return default

def _save_json(path: Path, data):
    try:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

def load_profile():
    data = _load_json(PROFILE_FILE, {})
    # sensible defaults
    return {
        "display_name": data.get("display_name", "New Pilot"),
        "email": data.get("email", st.session_state.get("user_email", "")),
        "avatar_key": data.get("avatar_key", "wp-black"),
        "show_reasons": data.get("show_reasons", True),
        "default_mode": data.get("default_mode", "quick"),  # quick|detailed
    }

def save_profile(p):
    _save_json(PROFILE_FILE, p)

def load_history():
    h = _load_json(HISTORY_FILE, [])
    # keep most recent first
    return sorted(h, key=lambda x: x.get("ts",""), reverse=True)[:50]

def add_history(entry: dict):
    h = load_history()
    h.insert(0, entry)  # newest first
    # clamp to 50
    h = h[:50]
    _save_json(HISTORY_FILE, h)

# Avatars you have in static/avatars/
AVATAR_CHOICES = {
    "wp-black":  f"{BACKEND_BASE}/static/avatars/wp-black.png",
    "wp-white":  f"{BACKEND_BASE}/static/avatars/wp-white.png",
    "wp-blue":   f"{BACKEND_BASE}/static/avatars/wp-blue.png",
    # add more here as you upload them
}

# Sidebar mini icon (same approach as in the playbook UI notes)
st.markdown(f"""
<style>
  .ccp-hero {{ text-align:center; margin:6px 0 8px 0; }}
  .ccp-hero h1 {{ margin:0; line-height:1.1; }}
  .ccp-hero .sub {{ color:#888; margin:4px 0 0 0; }}
  [data-testid="stSidebar"] {{ position:relative; }}
  [data-testid="stSidebar"]::before {{
    content:""; position:absolute; top:12px; left:16px; width:20px; height:20px;
    background-image:url('{LOGO_URL}');
    background-size:contain; background-repeat:no-repeat; opacity:0.9;
  }}
  div[role="tablist"] {{ justify-content:center !important; }} /* center tabs */
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Feature Gates / Popover Helper
# =============================================================================
HAS_POPOVER = bool(getattr(st, "popover", None))

def _popover_accessible():
    """Return a popover context manager compatible with multiple Streamlit versions, with no visible label."""
    if not HAS_POPOVER:
        raise RuntimeError("st.popover not available")
    try:
        return st.popover("", icon=None, use_container_width=False)
    except TypeError:
        return st.popover("", use_container_width=False)

# Back-compat alias so existing calls don't break
_popover_blank = _popover_accessible

def stick_to_tab(idx_or_name):
    """Back-compat shim: select a tab by index or name using stateful nav."""
    if isinstance(idx_or_name, int):
        i = max(0, min(idx_or_name, len(TAB_NAMES) - 1))
        name = TAB_NAMES[i]
    else:
        name = idx_or_name if idx_or_name in TAB_NAMES else TAB_NAMES[0]
    _set_tab(name)
    return name

# =============================================================================
# State defaults (stable keys)
# =============================================================================
for k, v in {
    "td_input": "",
    "td_result": None,
    "td_report_bytes": None,
    "td_busy": False,
    "quick_input": "",
    "quick_result": None,
    "quick_report_bytes": None,
    "quick_busy": False,
    "prefill_text_detailed": "",
    "text_busy": False,
    "text_d_busy": False,
    "pdf_busy": False,
    "pdf_d_busy": False,
    "pdfq_busy": False,
    "pdfq_result": None,
    "pdfq_report_bytes": None,
    "pdfq_uploader_key": 0,
    "pdfd_busy": False,
    "pdfd_result": None,
    "pdfd_report_bytes": None,
}.items():
    st.session_state.setdefault(k, v)

# =============================================================================
# Config & Constants
# =============================================================================
BACKEND  = os.getenv("BACKEND", "http://127.0.0.1:8787")
LOGO_URL = f"{BACKEND}/static/ccp-logo.png"

AVATAR_BASE = f"{BACKEND}/static/avatars"
AVATAR_CHOICES = [
    ("white-black", "Wp · Black text"),
    ("white-red",   "Wp · Red text"),
    ("white-blue",  "Wp · Blue text"),
    ("white-yellow","Wp · Yellow text"),
    ("black-red",   "Bp · Red text"),
    ("black-blue",  "Bp · Blue text"),
    ("black-yellow","Bp · Yellow text"),
    ("black-white", "Bp · White text"),
]
def avatar_url_from_choice(choice_slug: str) -> str:
    return f"{AVATAR_BASE}/ccp-{choice_slug}.png"

MAX_CHARS  = 20_000
MAX_PDF_MB = 10

# =============================================================================
# Prefs (local JSON)
# =============================================================================
PREFS_PATH = os.path.expanduser("~/.ccp_prefs.json")

def _load_prefs() -> dict:
    try:
        if os.path.exists(PREFS_PATH):
            with open(PREFS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _save_prefs(prefs: dict) -> None:
    try:
        with open(PREFS_PATH, "w", encoding="utf-8") as f:
            json.dump(prefs, f, indent=2)
    except Exception:
        pass

if "prefs" not in st.session_state:
    st.session_state["prefs"] = _load_prefs() or {
        "display_name":  "New Pilot",
        "default_mode":  "Quick",
        "risk_badges":   True,
        "avatar_choice": "black-white",
    }

def _prefs() -> dict:
    return st.session_state["prefs"]

def _risk_on() -> bool:
    return bool(_prefs().get("risk_badges", True))


# =============================================================================
# Helper — Compact renderer for Upload PDF (Quick)
# (uses the same .cc-card vibe as your Detailed tab)
# =============================================================================

def _looks_like_preamble(s: str) -> bool:
    """Heuristic: scream-case, very long, lots of WHEREAS/ALL CAPS → preamble, not a summary."""
    if not s: return False
    t = s.strip()
    # % uppercase letters
    letters = [c for c in t if c.isalpha()]
    if letters:
        upper_ratio = sum(1 for c in letters if c.isupper()) / max(1, len(letters))
        if upper_ratio > 0.6 and len(t) > 300:
            return True
    # common preamble tokens
    bad_markers = ("WHEREAS", "AGREEMENT", "THIS MASTER", "RECITALS", "WITNESSETH")
    return any(m in t[:220].upper() for m in bad_markers)

def _qp_pick_summary(res: dict) -> str | None:
    """
    Prefer a concise summary in this order:
      1) res['overall']['summary'] from DETAILED (if present, not preamble)
      2) res['summary'] from QUICK (but reject if it looks like preamble)
      3) Synthesize from clause summaries (top 3 one-liners)
      4) Fallback: _extract_overall_summary(res) if nothing else
    """
    if not isinstance(res, dict):
        return None

    # 1) Detailed overall.summary
    o = res.get("overall") or {}
    s = o.get("summary")
    if isinstance(s, str) and s.strip() and not _looks_like_preamble(s):
        return s.strip()

    # 2) Quick summary (reject preamble)
    s2 = res.get("summary")
    if isinstance(s2, str) and s2.strip() and not _looks_like_preamble(s2):
        return s2.strip()

    # 3) Synthesize from clauses
    clauses = res.get("clauses") or []
    bullets = []
    for c in clauses[:6]:
        line = (c.get("summary") or c.get("notes") or "").strip()
        if not line:
            # fall back to first sentence of text
            txt = (c.get("text") or c.get("raw_text") or "").strip()
            if txt:
                import re
                line = re.split(r'(?<=[\.\!\?])\s+', txt)[0][:280].strip()
        if line:
            bullets.append(line)
        if len(bullets) >= 3:
            break
    if bullets:
        return " ".join(bullets)

    # 4) Last resort: your existing extractor
    try:
        s3 = _extract_overall_summary(res)
        if isinstance(s3, str) and s3.strip() and not _looks_like_preamble(s3):
            return s3.strip()
    except Exception:
        pass
    return None

# --- local helper: paragraphize + sentence wrapping (UPDATED) ---
def _qp_to_paragraphs(text: str | None) -> str:
    """
    Turn a single blob into readable <p> paragraphs.
    Try blank lines → single newlines → punctuation (., !, ?) → em dashes/semicolons.
    Group sentences to ~250-300 chars per paragraph.
    """
    import html as _html, re
    t = (text or "").strip()
    if not t:
        return ""

    # First try blank-line then single newline splits
    blocks = [p.strip() for p in t.replace("\r","").split("\n\n") if p.strip()]
    if len(blocks) <= 1:
        blocks = [p.strip() for p in t.split("\n") if p.strip()]

    if len(blocks) <= 1:
        # No natural breaks → sentence-level splitting
        t = re.sub(r"[ \t]+", " ", t)
        # Also split on em dashes / semicolons as soft breaks
        t = re.sub(r"[—–]+", ". ", t)       # em/en dashes → period + space
        t = re.sub(r";\s*", ". ", t)        # semicolons → period + space
        sents = re.split(r'(?<=[\.\!\?])\s+', t)

        # group sentences into compact paragraphs
        blocks, cur = [], ""
        for s in sents:
            s = s.strip()
            if not s:
                continue
            if len(cur) + len(s) > 280 and cur:
                blocks.append(cur)
                cur = s
            else:
                cur = (cur + " " + s).strip() if cur else s
        if cur:
            blocks.append(cur)

    return "".join(f"<p>{_html.escape(p)}</p>" for p in blocks)

# =============================================================================
# Helpers — small utilities
# =============================================================================
from datetime import datetime  # ensure present
# (also ensure you have: import io, from typing import Optional, and components imported earlier)

# ====================== Helpers — Recents (unified) ======================
def add_history_item(entry_type: str, title: str, payload=None, meta: dict | None = None, limit: int = 200) -> None:
    """
    Add a history entry (newest first) and trim to `limit`.
    entry_type: "text", "text_detailed", "pdf", "pdf_detailed"
    title: short label (file name or first 60 chars of text)
    payload: dict/str result to reopen later
    meta: optional flags, e.g. {"report": True}
    """
    hist = st.session_state.setdefault("history", [])

    entry = {
        "type": entry_type,
        "title": (title or "Untitled").strip(),
        "ts": datetime.utcnow().isoformat(timespec="seconds"),
        "payload": payload,
        "fav": False,
        "meta": meta or {},
    }

    # simple de-dupe by (type, title)
    hist = [e for e in hist if not (e.get("type") == entry["type"] and e.get("title") == entry["title"])]
    hist.insert(0, entry)

    # trim
    if len(hist) > limit:
        del hist[limit:]

    st.session_state["history"] = hist

    # persist if you have a saver
    try:
        _save_history(hist)
    except Exception:
        pass

# Back-compat alias so any old calls keep working
add_recent_entry = add_history_item

# ---- REMOVE this whole function if present (old, separate store) ------------
# def _add_recent(kind: str, title: str, payload: str = "", meta: dict | None = None):
#     ...

# ====================== Other small helpers (keep) ======================
def _warn_near_limit(count: int, max_chars: int, threshold: float = 0.9):
    try:
        if max_chars and count >= int(max_chars * threshold):
            st.info(f"You're near the limit: {count} / {max_chars} characters.")
    except Exception:
        pass

def copy_btn(text: str, label: str, key: str):
    import base64 as _b
    enc = _b.b64encode((text or "").encode("utf-8")).decode("ascii")
    html = f"""
    <button style="padding:6px 10px;border:1px solid #333;border-radius:6px;cursor:pointer;background:#0e1117;color:#ddd;"
            onclick="navigator.clipboard.writeText(atob('{enc}'));this.innerText='Copied!';setTimeout(()=>this.innerText='{label}',1200)">
      {label}
    </button>
    """
    try:
        components.html(html, height=38, key=key)
    except TypeError:
        components.html(html, height=38)

def _retry_button(label: str, key: str):
    if st.button(label, key=key):
        st.rerun()

def _pdf_page_count(file) -> Optional[int]:
    try:
        from pypdf import PdfReader
        data = file.getvalue() if hasattr(file, "getvalue") else getattr(file, "read", lambda: b"")()
        if not data:
            return None
        bio = io.BytesIO(data)
        return len(PdfReader(bio).pages)
    except Exception:
        return None

def _file_size_mb(file) -> float:
    try:
        n = file.size
    except Exception:
        try:
            n = len(file.getvalue())
        except Exception:
            n = 0
    return round((n or 0) / (1024 * 1024), 2)

def _too_big(file) -> bool:
    try:
        return (file.size or 0) > MAX_PDF_MB * 1024 * 1024
    except Exception:
        try:
            return len(file.getvalue()) > MAX_PDF_MB * 1024 * 1024
        except Exception:
            return False

def _notice(msg: str, position: str = "top-center"):
    pos_css = {
        "top-right":    "top: 18px; right: 18px;",
        "top-center":   "top: 18px; left: 50%; transform: translateX(-50%);",
        "bottom-right": "bottom: 18px; right: 18px;",
        "bottom-left":  "bottom: 18px; left: 18px;",
    }.get(position, "top: 18px; right: 18px;")
    st.markdown(
        f"""
        <div style="
          position: fixed; {pos_css}
          z-index: 9999;
          background: #0e1117;
          color: #fff;
          border: 1px solid #333;
          border-radius: 8px;
          padding: 10px 14px;
          box-shadow: 0 6px 18px rgba(0,0,0,0.25);
          font-size: 14px;">
          <span>{msg}</span>
          <span style="margin-left:12px; cursor:pointer; color:#aaa;"
                onclick="this.parentElement.style.display='none'">✖</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

def _clear_auth_overlay():
    st.markdown("""
    <style>
      [data-testid="stAppViewContainer"] .main { filter:none !important; pointer-events:auto !important; }
      [data-testid="stSidebar"] .block-container { filter:none !important; pointer-events:auto !important; }
      .auth-dim::before { display:none !important; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
<style>
/* Neutralize any transform-based placement so we can control with top/left */
[data-testid="stSidebar"] div[role="dialog"][data-baseweb="popover"],
html body div[role="dialog"][data-baseweb="popover"]{
  transform: none !important;
  will-change: top,left !important;
  position: fixed !important;           /* we control it explicitly */
  z-index: 9999 !important;
  max-height: calc(100vh - 32px) !important;
  overflow-y: auto !important;
  overflow-x: hidden !important;
  background:#12151d !important;
  border:1px solid #2a2f37 !important;
  border-radius:12px !important;
  box-shadow:0 10px 28px rgba(0,0,0,.35) !important;
  padding:8px 10px !important;
  min-width: 260px !important;
  max-width: 320px !important;
  backdrop-filter: saturate(140%) blur(8px);
}

/* Menu internals */
[data-testid="stSidebar"] .menu-item .stButton > button{
  width: 100%;
  height: 36px !important;
  display: flex; align-items: center; justify-content: flex-start; gap: 8px;
  background: transparent !important; color: #e7e7ea !important;
  border: 0 !important; border-radius: 8px !important; padding: 8px 10px !important;
  font-weight: 500 !important; letter-spacing: .01em; box-shadow: none !important;
}
[data-testid="stSidebar"] .menu-item .stButton > button:hover{ background:#2b2d36 !important; }
[data-testid="stSidebar"] .menu-item .stButton > button:active{ background:#343541 !important; }
.menu-hr{ height:1px; background:#2f3340; border:0; margin:6px 2px; }
[data-testid="stSidebar"] .menu-rename .stTextInput > div > div > input{
  background:#1c1e24 !important; color:#f1f1f2 !important; border:1px solid #313543 !important;
  border-radius:8px !important; font-size:14px !important; height:34px !important; padding:0 10px !important;
}
[data-testid="stSidebar"] .menu-rename .stButton > button{
  height:36px !important; background:#10a37f !important; color:#fff !important;
  border:0 !important; border-radius:8px !important; font-weight:600 !important;
}
[data-testid="stSidebar"] .menu-rename .stButton > button:hover{ background:#0e8e6f !important; }

/* Sidebar basics */
[data-testid="stSidebar"] .block-container { padding-top: 56px; padding-bottom: 8px; }
[data-testid="stSidebar"]::before {
  content: ""; position: absolute; top: 12px; left: 16px; width: 28px; height: 28px;
  background: url('REPLACE_LOGO_URL') no-repeat center / contain; opacity: .95;
}
.side-label {
  font-size: .80rem; letter-spacing: .06em; text-transform: uppercase;
  color: #8a94a6; margin: 0 0 6px 0;
}
.side-hr { height: 1px; background: #262a33; border: 0; margin: 8px 0; }
.side-pilot {
  display: grid; grid-template-columns: 40px 1fr; gap: 10px; align-items: center;
  margin: 6px 0 2px 0;
}
.side-pilot img { width: 40px; height: 40px; border-radius: 9999px; }
.side-pilot .name { font-weight: 700; line-height: 1.1; color: #e5e7eb; }
.side-pilot .by   { color: #8a94a6; font-size: .85rem; margin-top: 2px; }

/* Recent list tidy */
[data-testid="stSidebar"] .side-list .element-container{ margin:0 !important; padding:0 !important; }
[data-testid="stSidebar"] .side-list [data-testid="stHorizontalBlock"]{ gap:0 !important; margin:0 !important; }
[data-testid="stSidebar"] .side-list [data-testid="column"]{ padding-left:0 !important; padding-right:0 !important; }
.side-row{ display:flex; align-items:center; justify-content:space-between; gap:8px; padding:1px 0 !important; border-bottom:1px solid #2a2d36; }
.side-row:first-child{ border-top:1px solid #2a2d36; }
.side-title{ margin:0 !important; line-height:1.25; font-size:inherit; color:#e5e7eb; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:220px; }
.side-meta{ margin:0 !important; line-height:1.15; font-size:inherit; color:#8a94a6; }
.side-title .star, .side-title .star-fav { display:inline-block; margin-right:6px; }
.side-title .star     { color:#fff; }
.side-title .star-fav { color:#FFC107; font-weight:900; }

/* Invisible ⋯ trigger */
[data-testid="stSidebar"] [data-testid="stPopover"]{
  position: relative; display:inline-flex; width: 24px; height: 24px;
}
[data-testid="stSidebar"] [data-testid="stPopover"] *{
  background: transparent !important; border: 0 !important; box-shadow: none !important; color: transparent !important;
}
[data-testid="stSidebar"] [data-testid="stPopover"] > button,
[data-testid="stSidebar"] [data-testid="stPopover"] [role="button"]{
  width: 24px !important; height: 24px !important; padding: 0 !important; margin: 0 !important; min-width: 0 !important; border-radius: 0 !important;
  font-size: 0 !important; opacity: 0 !important; outline: none !important;
}
[data-testid="stSidebar"] [data-testid="stPopover"] svg{ display:none !important; }
[data-testid="stSidebar"] [data-testid="stPopover"]::after{
  content:"⋯"; position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
  font-size:20px; line-height:1; color:#cbd5e1; opacity:0; transition:opacity .12s ease; pointer-events:none;
}
[data-testid="stSidebar"] [data-testid="stPopover"]:hover::after{ opacity:1; }

/* Header & tabs */
.ccp-hero { text-align: center; margin: 6px 0 8px 0; }
.ccp-hero h1 { margin: 0; padding: 0; font-size: 2.2rem; line-height: 1.1; }
.ccp-hero .sub { color: #888; margin: 4px 0 0 0; line-height: 1.3; }
div[role="tablist"] { justify-content: center !important; margin-top: 6px !important; margin-bottom: 8px !important; }

/* Small chip style used in multiple places */
.chips span{
  display:inline-block; padding:2px 8px; margin:2px 6px 2px 0;
  border:1px solid #2a2f37; background:#0f131a; border-radius:999px;
  font-size:12px; opacity:.9;
}
</style>
""".replace("REPLACE_LOGO_URL", LOGO_URL), unsafe_allow_html=True)

components.html("""
<script>
(() => {
  'use strict';
  if (window.__ccpPopoverMgr_v8) return;
  window.__ccpPopoverMgr_v8 = true;

  const PADS = { top: 32, bottom: 16, left: 16, right: 16 };
  const GAP  = 8;
  const FLIP_THRESHOLD = 72;

  let lastTriggerEl = null;
  let lastPointer   = { x: null, y: null };

  function remember(ev){
    lastPointer.x = ev.clientX ?? null;
    lastPointer.y = ev.clientY ?? null;
    const t = ev.target.closest('[data-testid="stPopover"]')
          || ev.target.closest('.side-ellipsis .stButton>button')
          || ev.target.closest('button[aria-haspopup="dialog"]')
          || ev.target.closest('button');
    if (t) lastTriggerEl = t;
  }
  function rememberFocus(ev){
    const t = (ev.target && ev.target.closest)
      ? (ev.target.closest('[data-testid="stPopover"]')
         || ev.target.closest('.side-ellipsis .stButton>button')
         || ev.target.closest('button[aria-haspopup="dialog"]')
         || ev.target.closest('button'))
      : null;
    if (t) lastTriggerEl = t;
  }

  function getAnchorRect(){
    if (lastTriggerEl && document.body.contains(lastTriggerEl)) {
      return lastTriggerEl.getBoundingClientRect();
    }
    if (lastPointer.x != null && lastPointer.y != null) {
      const x = lastPointer.x, y = lastPointer.y;
      return { left:x, right:x, top:y, bottom:y, width:0, height:0 };
    }
    const sb = document.querySelector('[data-testid="stSidebar"]');
    return sb ? sb.getBoundingClientRect()
              : { left:PADS.left, right:PADS.left, top:PADS.top, bottom:PADS.top, width:0, height:0 };
  }

  function getPlacementNode(pop){
    if (!pop) return null;
    const inner = pop.querySelector('[data-baseweb="popover"]');
    return inner || pop;
  }

  function neutralize(node){
    if (!node) return;
    node.style.setProperty('position', 'fixed', 'important');
    node.style.setProperty('transform', 'none', 'important');
    node.style.setProperty('inset', 'auto', 'important');
    node.style.setProperty('right', 'auto', 'important');
    node.style.setProperty('bottom','auto', 'important');
    node.style.setProperty('z-index','9999', 'important');
    node.style.setProperty('overflow-y','auto','important');
    node.style.setProperty('will-change','top,left','important');
  }

  function clampXY(left, top, w, h, vw, vh){
    left = Math.max(PADS.left, Math.min(left, vw - PADS.right - w));
    top  = Math.max(PADS.top,  Math.min(top,  vh - PADS.bottom - h));
    return { left, top };
  }

  function place(pop, pass=0){
    if (!pop || !document.body.contains(pop)) return;
    neutralize(pop);
    const node = getPlacementNode(pop);
    if (node !== pop) neutralize(node);

    const vw = innerWidth, vh = innerHeight;

    const maxW = Math.max(180, vw - (PADS.left + PADS.right));
    node.style.setProperty('max-width',  maxW + 'px', 'important');

    let r0 = node.getBoundingClientRect();
    const w = Math.min(r0.width, maxW);

    const a = getAnchorRect();
    const belowTop = a.bottom + GAP;
    const availBelow = vh - PADS.bottom - belowTop;
    const availAbove = (a.top - GAP) - PADS.top;

    let placeAbove = false;
    if (availBelow < FLIP_THRESHOLD && availAbove > availBelow) {
      placeAbove = true;
    }

    const minH = 120;
    const targetMaxH = Math.max(minH, placeAbove ? availAbove : availBelow);
    node.style.setProperty('max-height', targetMaxH + 'px', 'important');

    let left = Math.min(a.right + GAP, vw - PADS.right - w);
    let top  = placeAbove ? (a.top - GAP - Math.max(minH, Math.min(r0.height, targetMaxH)))
                          : belowTop;

    let r1 = node.getBoundingClientRect();
    let h0 = Math.min(r1.height, targetMaxH);
    ({ left, top } = clampXY(left, top, w, h0, vw, vh));
    node.style.setProperty('left', left + 'px', 'important');
    node.style.setProperty('top',  top  + 'px', 'important');

    if (pass < 2) { requestAnimationFrame(()=>place(pop, pass+1)); return; }

    let r2 = node.getBoundingClientRect();
    const h = Math.min(r2.height, targetMaxH);
    ({ left, top } = clampXY(left, top, w, h, vw, vh));
    node.style.setProperty('left', left + 'px', 'important');
    node.style.setProperty('top',  top  + 'px', 'important');
  }

  function observePop(pop){
    try {
      if (pop.__ccpObs) return;
      const obs = new MutationObserver(() => {
        requestAnimationFrame(() => place(pop));
      });
      obs.observe(pop, {
        attributes: true,
        attributeFilter: ['style','class','data-placement','data-popper-placement']
      });
      pop.__ccpObs = obs;
      pop.addEventListener('animationend',  () => place(pop), { passive:true });
      pop.addEventListener('transitionend', () => place(pop), { passive:true });
    } catch(e) {}
  }

  function repositionAll(){
    document.querySelectorAll('div[role="dialog"][data-baseweb="popover"]').forEach(pop=>{
      requestAnimationFrame(()=>{ place(pop); observePop(pop); });
    });
  }

  addEventListener('pointerdown', remember, { passive:true, capture:true });
  addEventListener('click',       remember, { passive:true, capture:true });
  addEventListener('keydown',     rememberFocus, true);
  addEventListener('focusin',     rememberFocus, true);

  const sidebar = document.querySelector('[data-testid="stSidebar"]');
  if (sidebar) sidebar.addEventListener('scroll', repositionAll, { passive:true });
  addEventListener('scroll', repositionAll, { passive:true });
  addEventListener('resize', repositionAll, { passive:true });

  const mo = new MutationObserver(muts => {
    muts.forEach(m => {
      m.addedNodes.forEach(node => {
        if (node.nodeType !== 1) return;
        if (node.matches?.('div[role="dialog"][data-baseweb="popover"]')) {
          requestAnimationFrame(() => { place(node); observePop(node); });
        }
        node.querySelectorAll?.('div[role="dialog"][data-baseweb="popover"]').forEach(p=>{
          requestAnimationFrame(() => { place(p); observePop(p); });
        });
      });
    });
  });
  mo.observe(document.body, { childList: true, subtree: true });

  repositionAll();
})();
</script>
""", height=0)
st.session_state["_js_injected_v8"] = True

# =============================================================================
# Auth Bootstrap & Gates
# =============================================================================
_S = get_supa()

# bootstrap session (cookies/token → session_state)
try:
    bootstrap_session(_S)
except Exception:
    pass

# small helpers
def _qp_first(name: str, default: str = ""):
    v = st.query_params.get(name, default)
    if isinstance(v, list):
        return v[0] if v else default
    return v or default

uid = get_user_id()
otp_verified = bool(st.session_state.get("otp_verified"))
has_recovery_query = (_qp_first("type") == "recovery") and bool(_qp_first("token_hash"))

# Open auth modal when needed; stop rendering until user completes auth
needs_auth = (
    st.session_state.get("show_new_pw")
    or (not uid and has_recovery_query and not otp_verified)
    or (not uid)
)

if needs_auth:
    render_auth_modal(_S)
    if get_user_id():
        st.session_state["_do_rerun"] = True
    st.stop()

# --- Email confirmed notice (once) ---
confirmed = st.query_params.get("confirmed")
if (isinstance(confirmed, list) and "1" in confirmed) or confirmed == "1":
    st.session_state["confirm_notice"] = True
    try:
        del st.query_params["confirmed"]
    except Exception:
        pass

if st.session_state.pop("confirm_notice", False):
    st.success("✅ Email confirmed — you're in!")

_clear_auth_overlay()

# =============================================================================
# History (single source of truth)
# =============================================================================
HISTORY_PATH = os.path.expanduser("~/.ccp_history.json")

def _load_history() -> list[dict]:
    try:
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data[:20]
    except Exception:
        pass
    return []

def _save_history(items: list[dict]) -> None:
    try:
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(items[:20], f, indent=2, ensure_ascii=False)
    except Exception:
        pass

if "history" not in st.session_state:
    st.session_state["history"] = _load_history()

def add_history(entry: dict) -> None:
    entry = dict(entry)
    entry.setdefault("fav", False)
    entry["ts"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    existing = [
        e for e in st.session_state.get("history", [])
        if not (e.get("type") == entry.get("type") and e.get("title") == entry.get("title"))
    ]
    items = [entry] + existing[:19]
    st.session_state["history"] = items
    _save_history(items)

def toggle_fav(idx: int) -> None:
    try:
        st.session_state["history"][idx]["fav"] = not st.session_state["history"][idx].get("fav", False)
        _save_history(st.session_state["history"])
    except Exception:
        pass

def delete_history_item(idx: int) -> None:
    try:
        items = st.session_state.get("history", [])
        if 0 <= idx < len(items):
            items.pop(idx)
            st.session_state["history"] = items
            _save_history(st.session_state["history"])
            st.rerun()
    except Exception:
        pass

# =============================================================================
# Sidebar (account, pilot, recent)
# =============================================================================

# --- Avatar catalog + helpers (keep near top-level, before the sidebar) ---
STATIC_BASE = "http://140.238.88.228:8787/static"  # served by FastAPI

AVATAR_CHOICES = {
    "white-blue":   "ccp-white-blue.png",
    "white-red":    "ccp-white-red.png",
    "white-yellow": "ccp-white-yellow.png",
    "white-black":  "ccp-white-black.png",
    "black-blue":   "ccp-black-blue.png",
    "black-red":    "ccp-black-red.png",
    "black-yellow": "ccp-black-yellow.png",
    "black-white":  "ccp-black-white.png",
}

def avatar_url_from_choice(choice: str) -> str:
    filename = AVATAR_CHOICES.get(choice) or AVATAR_CHOICES["white-blue"]
    return f"{STATIC_BASE}/avatars/{filename}?v=2"

def _prefs():
    # Your existing prefs loader might already exist; keep this as a safe fallback
    return st.session_state.get("_profile") or {"display_name": "New Pilot", "avatar_choice": "white-blue"}

def load_history():
    # Use your real history loader if you have one
    return st.session_state.get("_recent_history", [])

# --- Sidebar render ---
with st.sidebar:
    # SECTION: Account
    st.markdown('<div class="side-label">Account</div>', unsafe_allow_html=True)
    if st.button("Sign out", use_container_width=True, key="sidebar_signout_btn"):
        sign_out(_S)
        st.stop()
    st.markdown('<hr class="side-hr">', unsafe_allow_html=True)

    # SECTION: Pilot (avatar + name)
    st.markdown('<div class="side-label">Pilot</div>', unsafe_allow_html=True)

    prof = _prefs()
    # Accept either avatar_choice or avatar_key from older saves
    avatar_choice = prof.get("avatar_choice") or prof.get("avatar_key") or "white-blue"
    pilot_name    = prof.get("display_name", "New Pilot")
    avatar_url    = avatar_url_from_choice(avatar_choice)

    st.markdown(
        f"""
        <div class="side-pilot">
          <img src="{avatar_url}" alt="avatar">
          <div>
            <div class="name">{pilot_name}</div>
            <div class="by">Created by Contract Co-Pilot</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<hr class="side-hr">', unsafe_allow_html=True)

    # SECTION: Recent
    st.markdown('<div class="side-label">Recent</div>', unsafe_allow_html=True)
    recent = load_history()
    if not recent:
        st.caption("No entries yet.")
    else:
        # render up to 10 items
        for item in recent[:10]:
            title = item.get("title", "(untitled)")
            ts    = (item.get("ts","") or "")[:16].replace("T"," ")
            st.markdown(f"• {title}  \n<small>{ts}</small>", unsafe_allow_html=True)

# =============================================================================
# Sidebar — Recent entries (uses popover when available)
# =============================================================================
# Create the placeholder inside the sidebar and remember it for later updates
with st.sidebar:
    st.markdown('<div class="side-label">Recent</div>', unsafe_allow_html=True)
    RECENT_BOX = st.empty()
    st.session_state["RECENT_BOX"] = RECENT_BOX
    # Ensure history list exists
    st.session_state.setdefault("history", [])

# Safely get the placeholder outside the sidebar (e.g., during reruns)
RECENT_BOX = st.session_state.get("RECENT_BOX") or st.sidebar.empty()


def _recent_type_label(t: str) -> str:
    t = (t or "").lower()
    return {
        "text": "Text",
        "text_detailed": "Text (Detailed)",
        "pdf_quick": "PDF",
        "pdf": "PDF",                    # legacy/compat
        "pdf_detailed": "PDF (Detailed)",
    }.get(t, (t or "Item").title())


def _render_recent_menu(i: int, t: str, title: str, e: dict) -> None:
    # --- OPEN (tab-specific actions) ---
    if t in ("text", "text_detailed"):
        st.markdown('<div class="menu-item">', unsafe_allow_html=True)
        if st.button("↗ Open in Text (Detailed)", key=f"open_{i}", use_container_width=True):
            payload = e.get("payload", "") or ""
            st.session_state["prefill_text_detailed"] = (payload[:15000] if isinstance(payload, str) else "")
            jump_tab("Text (Detailed)")
        st.markdown('</div>', unsafe_allow_html=True)

    elif t in ("pdf", "pdf_quick"):
        st.markdown('<div class="menu-item">', unsafe_allow_html=True)
        if st.button("↗ Open in Upload PDF (Quick)", key=f"open_pdfq_{i}", use_container_width=True):
            payload = e.get("payload") or {}
            try:
                st.session_state["pdfq_result"] = payload or {}
                st.session_state["pdfq_done"] = True
            except Exception:
                pass
            jump_tab("Upload PDF")
        st.markdown('</div>', unsafe_allow_html=True)

    elif t == "pdf_detailed":
        st.markdown('<div class="menu-item">', unsafe_allow_html=True)
        if st.button("↗ Open in PDF (Detailed)", key=f"open_pdfd_{i}", use_container_width=True):
            payload = e.get("payload") or {}
            st.session_state["pdfd_result"] = payload
            st.session_state["pdfd_done"] = True
            st.session_state["__scroll_pdfd"] = True
            # if you persist report bytes in meta, you could restore here:
            # rb = e.get("meta", {}).get("report_bytes")
            # if rb: st.session_state["pdfd_report_bytes"] = rb
            jump_tab("PDF (Detailed)")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- FAVORITE / UNFAVORITE ---
    st.markdown('<div class="menu-item">', unsafe_allow_html=True)
    fav_label = "★ Unfavorite" if e.get("fav") else "☆ Favorite"
    if st.button(fav_label, key=f"fav_{i}", use_container_width=True):
        try:
            toggle_fav(i)
            st.rerun()
        except Exception:
            st.warning("Couldn't update favorite.")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- COPY TITLE ---
    st.markdown('<div class="menu-item">', unsafe_allow_html=True)
    if st.button("⧉ Copy title", key=f"copy_{i}", use_container_width=True):
        try:
            st.session_state[f"_copied_{i}"] = title
            st.toast("Title copied")
        except Exception:
            st.warning("Couldn't copy title.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="menu-hr"></div>', unsafe_allow_html=True)

    # --- RENAME ---
    st.markdown('<div class="menu-rename">', unsafe_allow_html=True)
    _, col_mid, _ = st.columns([1, 9, 1], vertical_alignment="center")
    with col_mid:
        st.markdown('<div style="text-align:center;margin:0 0 6px;">Rename</div>', unsafe_allow_html=True)
        new_title = st.text_input(
            "Rename entry",
            value=title,
            key=f"rename_in_{i}",
            placeholder="New title…",
            label_visibility="collapsed",
        )
        if st.button("Save name", key=f"rename_btn_{i}", use_container_width=True):
            try:
                st.session_state["history"][i]["title"] = (new_title or "Untitled").strip()
                _save_history(st.session_state["history"])
                st.toast("Renamed")
                st.rerun()
            except Exception:
                st.warning("Couldn't rename.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="menu-hr"></div>', unsafe_allow_html=True)

    # --- DELETE (danger) ---
    st.markdown('<div class="menu-item">', unsafe_allow_html=True)
    if st.button("🗑 Delete", key=f"del_{i}", use_container_width=True):
        try:
            delete_history_item(i)
        except Exception:
            st.warning("Couldn't delete.")
    st.markdown('</div>', unsafe_allow_html=True)


def render_recent_sidebar(box):
    st.session_state.setdefault("menu_idx", -1)
    entries = st.session_state.get("history", [])

    with box.container():
        if not entries:
            st.caption("No entries yet.")
            return

        st.markdown('<div class="side-list">', unsafe_allow_html=True)

        for i, e in enumerate(entries):
            t = (e.get("type") or "text")
            kind = _recent_type_label(t)
            is_fav = bool(e.get("fav"))
            star_char  = "★" if is_fav else "☆"
            star_class = "star-fav" if is_fav else "star"
            title = e.get("title", "Untitled")
            ts = e.get("ts", "—")

            c1, c2 = st.columns([1, 0.12], vertical_alignment="center")

            with c1:
                st.markdown('<div class="side-row">', unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <div class="side-left">
                      <div class="side-title">
                        <span class="{star_class}">{star_char}</span> {title}
                      </div>
                      <div class="side-meta">{ts} · {kind}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown('</div>', unsafe_allow_html=True)

            with c2:
                st.markdown('<div class="side-ellipsis">', unsafe_allow_html=True)
                if HAS_POPOVER:
                    with _popover_accessible():
                        _render_recent_menu(i, t, title, e)
                else:
                    if st.button(" ", key=f"more_{i}", use_container_width=False):
                        st.session_state["menu_idx"] = -1 if st.session_state["menu_idx"] == i else i
                        st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

            if (not HAS_POPOVER) and st.session_state.get("menu_idx") == i:
                with st.container():
                    st.markdown('<div class="fallback-pop">', unsafe_allow_html=True)
                    _render_recent_menu(i, t, title, e)
                    st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        if st.button("Clear history", key="clear_history_btn", use_container_width=True):
            st.session_state["history"] = []
            _save_history([])
            try:
                if os.path.exists(HISTORY_PATH):
                    os.remove(HISTORY_PATH)
            except Exception:
                pass
            st.rerun()


# Render sidebar list
render_recent_sidebar(RECENT_BOX)

# =============================================================================
# Header
# =============================================================================
st.markdown(
    """
    <div class="ccp-hero">
      <h1>Contract Co-Pilot</h1>
      <div class="sub">Upload a contract to get plain-English summaries, risk alerts, tags &amp; entities.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

@st.cache_data(ttl=30)
def _api_health(base: str) -> tuple[bool, str]:
    try:
        r = requests.get(f"{base}/health", timeout=3)
        ok = (r.status_code == 200)
        txt = (r.text or "").strip()[:80]
        return ok, (txt if txt else "OK" if ok else "Unhealthy")
    except Exception as e:
        return False, f"{type(e).__name__}"

ok, msg = _api_health(BACKEND)
st.caption("🟢 Online" if ok else "🔴 Offline")

# =============================================================================
# NAV — HTML underline tabs (links; target=_self; decode ?tab=)
# =============================================================================
from urllib.parse import unquote_plus

TAB_NAMES = ["Paste Text", "Text (Detailed)", "Upload PDF", "PDF (Detailed)", "Profile"]

def _norm(x: str) -> str:
    return x if x in TAB_NAMES else TAB_NAMES[0]

# Source of truth = ?tab=... (decode + to space)
_tab_qp = st.query_params.get("tab")
if isinstance(_tab_qp, list): _tab_qp = _tab_qp[0]
active = _norm(unquote_plus(_tab_qp) if _tab_qp else st.session_state.get("_tab", TAB_NAMES[0]))
st.session_state["_tab"] = active
TAB = active  # <-- use in your routing below

# --- CSS (underline style like your screenshot) ---
st.markdown("""
<style>
#ccp-tabs .tabs{
  display:flex; align-items:center; justify-content:center;
  gap:18px; margin:6px 0 12px 0; padding-bottom:6px;
  list-style:none; border-bottom:1px solid rgba(255,255,255,0.10); padding-left:0;
}
#ccp-tabs .tabs li{ margin:0; padding:0; }
#ccp-tabs .tabs a{
  text-decoration:none; display:inline-block; padding:8px 10px; line-height:1.2;
  color:#c7c7c7; font-weight:600; letter-spacing:.02em;
  border-bottom:2px solid transparent; transition:color .12s, border-color .12s;
}
#ccp-tabs .tabs a:hover{ color:#fff; border-bottom-color:rgba(255,255,255,0.22); }
#ccp-tabs .tabs a.active{ color:#fff; border-bottom-color:#ff4d4d; }
</style>
""", unsafe_allow_html=True)

# --- Render nav as links that reload same window (no nested forms) ---
def enc(s: str) -> str:
    # minimal encoder good enough for our labels
    return s.replace(" ", "%20").replace("(", "%28").replace(")", "%29")

items = []
for t in TAB_NAMES:
    cls = "active" if t == active else ""
    items.append(f"<li><a class='{cls}' href='?tab={enc(t)}' target='_self' rel='nofollow'>{t}</a></li>")

st.markdown(f"<div id='ccp-tabs'><ul class='tabs'>{''.join(items)}</ul></div>", unsafe_allow_html=True)

# Helper for handlers to **stay on the same tab** after submit/reset
def _set_tab(name: str):
    name = _norm(name)
    st.query_params["tab"] = name
    st.session_state["_tab"] = name
    st.rerun()

def lock_tab(name: str):
    # keep tab without triggering a rerun (safe inside submit/reset handlers)
    if name in TAB_NAMES:
        st.session_state["_tab"] = name
        st.query_params["tab"] = name  # keep URL in sync

def jump_tab(name: str):
    # only use for actual navigation clicks, not in submit/reset
    if name in TAB_NAMES:
        st.session_state["_tab"] = name
        st.query_params["tab"] = name
        st.rerun()    
 
def reset_and_rerun(tab_name: str):
    # keep current tab and trigger a full refresh
    lock_tab(tab_name)
    st.rerun()        

# =============================================================================
# Shared Helper Functions (used across tabs)
# =============================================================================
def _extract_overall_summary(res: dict) -> str:
    o = (res or {}).get("overall", {}) or {}
    for k in ("summary", "overview", "overall_summary", "notes"):
        v = o.get(k) or res.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    parts = []
    for c in (res or {}).get("clauses", [])[:3]:
        s = c.get("summary") or c.get("notes") or ""
        if isinstance(s, str) and s.strip():
            parts.append(s.strip())
    return " ".join(parts) if parts else "—"

def _fmt_score(score):
    try:
        v = float(score)
        return f" ({int(v)}/100)" if v > 0 else ""
    except Exception:
        return ""

def _get_clauses(result: dict):
    if not isinstance(result, dict):
        return []
    return (result.get("clauses") or result.get("sections") or [])

def _level_str(x):
    return x.title() if isinstance(x, str) else str(x)

def _badge_class(level):
    if not isinstance(level, str):
        return "low"
    l = level.lower()
    if "high" in l:
        return "high"
    if "medium" in l:
        return "medium"
    return "low"

# =============================================================================
# Tab: Paste Text (Quick)
# =============================================================================
if TAB == "Paste Text":
    status_placeholder = st.empty()

    # ---- one-shot toast driver (shows toast on rerun) ----
    _toast_next = st.session_state.pop("__toast_next", None)
    if _toast_next:
        st.toast(_toast_next.get("msg", "Done ✓"), icon=_toast_next.get("icon", "✅"))
        st.markdown(f"""
        <script>
          setTimeout(() => {{
            const el = document.querySelector('[data-testid="stNotificationContent"]');
            if (el) el.style.transition = 'opacity 1s ease';
            if (el) el.style.opacity = '1';
          }}, 50);
          setTimeout(() => {{
            const el = document.querySelector('[data-testid="stNotificationContent"]');
            if (el) el.style.opacity = '0';
          }}, {_toast_next.get("ms", 5000)});
        </script>
        """, unsafe_allow_html=True)

    # --- styles for quick summary/risk cards (same look & feel as other tabs) ---
    st.markdown("""
    <style>
      #tq-summary { border:1px solid #2a2f37; border-radius:12px; padding:12px 14px; background:#0f131a; margin-bottom:12px; }
      #tq-summary .ttl  { font-weight:600; margin:0 0 8px 0; }
      #tq-summary .body { margin:2px 0 0 0; line-height:1.5; }
      #tq-summary .body p { margin:0 0 10px 0; }
      #tq-risk { border:1px solid #2a2f37; border-radius:12px; padding:12px 14px; background:#0f131a; margin:10px 0 12px 0; }
      #tq-risk .header { display:flex; align-items:center; gap:10px; }
      #tq-risk .title  { font-weight:600; margin:0; }
      #tq-risk .score  { font-size:12px; opacity:.85; margin-top:2px; }
      #tq-risk .badge  { display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; font-weight:600; line-height:1; }
      #tq-risk .badge.low    { background:#10331e; color:#61d092; border:1px solid #1a6c3f; }
      #tq-risk .badge.medium { background:#2e2a12; color:#ffd166; border:1px solid #806b1a; }
      #tq-risk .badge.high   { background:#3a1515; color:#ff7b7b; border:1px solid #8b1f1f; }
      #tq-risk ul.reasons    { margin:8px 0 0 0; padding-left:18px; }
      #tq-risk ul.reasons li { margin:0 0 4px 0; }

      .chips span{
        display:inline-block; padding:2px 8px; margin:2px 6px 2px 0;
        border:1px solid #2a2f37; background:#0f131a; border-radius:999px;
        font-size:12px; opacity:.9;
      }
    </style>
    """, unsafe_allow_html=True)

    # ---- state (QUICK TEXT) ----
    st.session_state.setdefault("quick_input", "")
    st.session_state.setdefault("quick_busy", False)
    st.session_state.setdefault("quick_result", None)
    st.session_state.setdefault("quick_report_bytes", None)
    # history de-dupe guard (lives only this session; avoids double add on rerun)
    st.session_state.setdefault("_history_guard", set())

    # ---- form ----
    with st.form("form_text_quick", clear_on_submit=False):
        st.session_state.quick_input = st.text_area(
            "Paste a contract clause or short section:",
            value=st.session_state.quick_input,
            height=180,
            placeholder="Paste text here…",
            key="quick_text_area",
        )
        count = len(st.session_state.quick_input or "")
        st.caption(f"{count} / {MAX_CHARS} characters")
        _warn_near_limit(count, MAX_CHARS, threshold=0.9)

        submit_quick = st.form_submit_button(
            "Analyze Text",
            key="quick_analyze_btn",
            use_container_width=True,
            disabled=st.session_state.quick_busy,
        )

    # ---- submit ----
    if submit_quick:
        lock_tab("Paste Text")
        if not (st.session_state.quick_input or "").strip():
            st.warning("Please paste some text.")
        else:
            st.session_state.quick_busy = True
            status = status_placeholder.status("Analyzing…", state="running", expanded=True)
            t0 = time.perf_counter()
            try:
                payload = {"text": st.session_state.quick_input}

                status.write("Sending text to /analyze_text …")
                r = requests.post(f"{BACKEND}/analyze_text", json=payload, timeout=45)
                r.raise_for_status()
                st.session_state.quick_result = r.json()

                # Optional: printable report
                status.write("Building printable report …")
                try:
                    r2 = requests.post(f"{BACKEND}/report_text_detailed", json=payload, timeout=60)
                    r2.raise_for_status()
                    st.session_state.quick_report_bytes = r2.content
                except Exception:
                    st.session_state.quick_report_bytes = None

                # ---- Push to history (success only, before rerun) ----
                hist_id = f"text-{int(time.time())}"
                if hist_id not in st.session_state._history_guard:
                    title_txt = (st.session_state.quick_input[:60] + "…") if st.session_state.quick_input else "Text"

                    # NEW: standard history entry (persists to history.json)
                    try:
                        if "add_history_entry" in globals():
                            # Prefer a reasonable title from result if present
                            _title_from_result = None
                            if isinstance(st.session_state.quick_result, dict):
                                _title_from_result = (
                                    st.session_state.quick_result.get("overall", {}).get("title")
                                    or st.session_state.quick_result.get("title")
                                )
                            add_history_entry(
                                kind="text",
                                title=_title_from_result or title_txt or "Quick text analysis",
                                payload={"result": st.session_state.get("quick_result")},
                                meta={"report": bool(st.session_state.get("quick_report_bytes"))} if "meta" in add_history_entry.__code__.co_varnames else None  # safe if meta not supported
                            )
                    except Exception:
                        pass

                    # EXISTING: legacy/local history paths (kept intact)
                    try:
                        if "add_history_item" in globals():
                            add_history_item(
                                entry_type="text",
                                title=title_txt,
                                payload=st.session_state.get("quick_result"),
                                meta={"report": bool(st.session_state.get("quick_report_bytes"))},
                            )
                        else:
                            # fallback to a simpler local history function if present
                            if "add_history" in globals():
                                add_history({
                                    "id": hist_id,
                                    "kind": "text",
                                    "title": title_txt,
                                    "ts": datetime.utcnow().isoformat()
                                })
                    except Exception:
                        pass

                    st.session_state._history_guard.add(hist_id)

                # Schedule a visible toast on the next run (so rerun won't clear it)
                st.session_state["_select_tab"] = "Paste Text"
                st.session_state["__toast_next"] = {
                    "msg": "Quick analysis ready ✓",
                    "icon": "✅",
                    "ms": 5000
                }

                status.update(label=f"Done in {time.perf_counter()-t0:.1f}s ✓", state="complete")
                status_placeholder.empty()
                st.caption(f"Done in {time.perf_counter()-t0:.1f}s ✓")
                st.rerun()

            except Exception as e:
                status.update(label="Analysis failed", state="error")
                st.error(f"Quick analyze failed: {e}")
                st.session_state.quick_result = None
                st.session_state.quick_report_bytes = None
            finally:
                st.session_state.quick_busy = False

    # ---- render quick results ----
    qres = st.session_state.get("quick_result")
    if qres:
        # ---- summary ----------------------------------------------------------
        summary_text = _extract_overall_summary(qres) or qres.get("summary") or "—"
        if summary_text and summary_text != "—":
            st.markdown('<div id="tq-summary"><div class="ttl">Summary</div><div class="body">', unsafe_allow_html=True)
            st.write(summary_text)
            st.markdown('</div></div>', unsafe_allow_html=True)

        # ---- robust overall extraction ---------------------------------------
        overall = qres.get("overall") or qres.get("risk") or {}
        if not overall:
            overall = {
                "level": qres.get("risk_level"),
                "risk_level": qres.get("risk_level"),
                "score": qres.get("risk_score"),
                "risk_score": qres.get("risk_score"),
                "reasons": qres.get("risk_reasons") or [],
                "risk_reasons": qres.get("risk_reasons") or [],
            }

        lvl_raw = overall.get("level") or overall.get("risk_level") or "—"
        lvl = _level_str(lvl_raw)
        score = overall.get("score", overall.get("risk_score"))
        reasons = overall.get("reasons") or overall.get("risk_reasons") or []
        lvl_class = _badge_class(lvl)

        top = reasons[:3]
        extra = reasons[3:]
        reasons_html = "<ul class='reasons'>" + "".join(f"<li>{r}</li>" for r in top) + "</ul>" if top else ""

        st.markdown(f"""
        <div id="tq-risk">
          <div class="header">
            <span class="badge {lvl_class}">{lvl}</span>
            <div>
              <div class="title">Overall Risk</div>
              <div class="score">{'' if score in (None,'—') else f'Score {score}/100'}</div>
            </div>
          </div>
          {reasons_html}
        </div>
        """, unsafe_allow_html=True)

        if extra:
            with st.expander(f"Show {len(extra)} more reason(s)"):
                for r in extra[:6]:
                    st.write(f"• {r}")

        # ---- optional entities ------------------------------------------------
        ents = qres.get("entities") or []
        names = []
        for e in ents:
            if isinstance(e, str):
                names.append(e.strip())
            elif isinstance(e, dict):
                txt = (e.get("text") or e.get("name") or "").strip()
                if txt:
                    names.append(txt)
        names = sorted({n for n in names if n})
        if names:
            with st.expander(f"Entities ({len(names)})"):
                st.markdown('<div class="chips">' + " ".join(f"<span>{n}</span>" for n in names[:40]) + "</div>", unsafe_allow_html=True)

        # --- sticky bottom actions --------------------------------------------
        with st.container():
            st.markdown('<div class="sticky-actions">', unsafe_allow_html=True)
            col_dl, col_reset = st.columns([4,1], vertical_alignment="center")

            with col_dl:
                has_report = st.session_state.get("quick_report_bytes") is not None
                if has_report:
                    st.download_button(
                        "⬇️ Download report (HTML)",
                        data=st.session_state.quick_report_bytes,
                        file_name="contract-report_text_quick.html",
                        mime="text/html",
                        key="tq_download_btn",
                        use_container_width=True,
                    )
                else:
                    st.button("⬇️ Download report (HTML)",
                              key="tq_download_btn_disabled",
                              disabled=True,
                              use_container_width=True)

            with col_reset:
                if st.button("Reset", key="quick_reset_btn", help="Clear this result"):
                    lock_tab("Paste Text")
                    for k in ("quick_input", "quick_result", "quick_report_bytes", "quick_busy"):
                        st.session_state.pop(k, None)
                    reset_and_rerun("Paste Text")
            st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# =================
# Tab: Text (Detailed)
# =============================================================================
elif TAB == "Text (Detailed)":
    st.markdown("### Text (Detailed)")
    status_placeholder = st.empty()

    # --- tab-local defaults ---
    st.session_state.setdefault("td_busy", False)
    st.session_state.setdefault("td_result", None)
    st.session_state.setdefault("td_report_bytes", None)
    st.session_state.setdefault("td_input", "")
    # de-dupe guard for history across immediate reruns
    st.session_state.setdefault("_history_guard", set())

    with st.form("form_text_detailed", clear_on_submit=False):
        st.session_state.td_input = st.text_area(
            "Paste multi-clause text",
            value=st.session_state.get("prefill_text_detailed", "") or st.session_state.td_input,
            height=220,
            key="td_input_area",
            help="Use paragraphs/blank lines for separate clauses.",
        )
        count = len(st.session_state.td_input or "")
        st.caption(f"{count} / {MAX_CHARS} characters")
        _warn_near_limit(count, MAX_CHARS, threshold=0.9)

        submit_td = st.form_submit_button(
            "Analyze (Detailed)",
            key="td_analyze_btn",
            use_container_width=True,
            disabled=st.session_state.td_busy,
        )

    if submit_td:
        lock_tab("Text (Detailed)")  # keep user on this tab throughout
        if not (st.session_state.td_input or "").strip():
            st.warning("Please paste some text.")
        else:
            st.session_state.td_busy = True
            status_placeholder.empty()  # ensure single status instance
            status = status_placeholder.status("Analyzing…", state="running", expanded=True)
            t0 = time.perf_counter()
            try:
                payload = {"text": st.session_state.td_input}

                status.write("Sending text to /analyze_text_detailed …")
                r = requests.post(f"{BACKEND}/analyze_text_detailed", json=payload, timeout=90)
                r.raise_for_status()
                st.session_state.td_result = r.json()

                status.write("Building printable report …")
                try:
                    r2 = requests.post(f"{BACKEND}/report_text_detailed", json=payload, timeout=90)
                    r2.raise_for_status()
                    st.session_state.td_report_bytes = r2.content
                except Exception:
                    st.session_state.td_report_bytes = None

                # --- record in Recent (success only, before rerun) ---
                full_text = st.session_state.td_input or ""
                first_line = (full_text.splitlines() or [""])[0].strip()
                safe_title = first_line if first_line else "Text (Detailed)"
                title = (safe_title[:60] + ("…" if len(safe_title) > 60 else ""))

                hist_id = f"td-{int(time.time())}"
                if hist_id not in st.session_state._history_guard:
                    # NEW: standard history entry (persists to history.json)
                    try:
                        if "add_history_entry" in globals():
                            _overall_title = None
                            _clauses_count = 0
                            try:
                                _overall_title = ((st.session_state.td_result or {}).get("overall", {}) or {}).get("title")
                                _clauses_count = len(_get_clauses(st.session_state.td_result or {}))
                            except Exception:
                                pass
                            add_history_entry(
                                kind="text_detailed",
                                title=_overall_title or title or "Text (Detailed)",
                                payload={"result": st.session_state.get("td_result")},
                                meta={
                                    "report": bool(st.session_state.get("td_report_bytes")),
                                    "clauses": _clauses_count,
                                } if "meta" in add_history_entry.__code__.co_varnames else None  # safe if meta not supported
                            )
                    except Exception:
                        pass

                    # EXISTING: legacy/local history paths (kept intact)
                    try:
                        if "add_history_item" in globals():
                            add_history_item(
                                entry_type="text_detailed",
                                title=title,
                                payload=st.session_state.get("td_result"),
                                meta={
                                    "report": bool(st.session_state.get("td_report_bytes")),
                                    "clauses": len(_get_clauses(st.session_state.td_result or {})),
                                },
                            )
                        elif "add_history" in globals():
                            add_history({
                                "id": hist_id,
                                "kind": "text_detailed",
                                "title": title,
                                "ts": datetime.utcnow().isoformat()
                            })
                    except Exception:
                        pass
                    st.session_state._history_guard.add(hist_id)

                # Prepare toast + tab persistence, then rerun
                st.session_state["_select_tab"] = "Text (Detailed)"
                st.session_state["__toast_next"] = {
                    "msg": "Detailed analysis ready ✓",
                    "icon": "✅",
                    "ms": 5000
                }

                elapsed = time.perf_counter() - t0
                status.update(label=f"Done in {elapsed:.1f}s ✓", state="complete")
                st.rerun()

            except requests.HTTPError as http_err:
                resp = http_err.response
                status.update(label="Analysis failed", state="error")
                st.error(f"Error contacting backend: {getattr(resp, 'status_code', '')} {getattr(resp, 'reason', '')}")
                try:
                    st.caption((resp.text or "")[:400])
                except Exception:
                    pass
                st.session_state.td_result = None
                st.session_state.td_report_bytes = None
            except Exception as e:
                status.update(label="Analysis failed", state="error")
                if "Error contacting backend" not in str(e):
                    st.error(f"Analysis failed: {e}")
                st.session_state.td_result = None
                st.session_state.td_report_bytes = None
            finally:
                status_placeholder.empty()
                st.session_state.td_busy = False

    if st.session_state.td_result:
        overall = st.session_state.td_result.get("overall", {}) or {}
        clauses = list(_get_clauses(st.session_state.td_result))

        st.markdown("""
        <style>
          #td-summary, #td-risk {
            border:1px solid #2a2f37; border-radius:12px; background:#0f131a;
            padding:12px 14px; margin:10px 0 12px 0;
          }
          #td-summary .ttl{ font-weight:600; margin:0 0 8px 0; }
          #td-risk .header { display:flex; align-items:center; gap:10px; }
          #td-risk .title  { font-weight:600; margin:0; }
          #td-risk .score  { font-size:12px; opacity:.85; margin-top:2px; }
          #td-risk .badge  { display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; font-weight:600; line-height:1; }
          #td-risk .badge.low    { background:#10331e; color:#61d092; border:1px solid #1a6c3f; }
          #td-risk .badge.medium { background:#2e2a12; color:#ffd166; border:1px solid #806b1a; }
          #td-risk .badge.high   { background:#3a1515; color:#ff7b7b; border:1px solid #8b1f1f; }
          #td-risk ul.reasons { margin:8px 0 0 0; padding-left:18px; }
          #td-risk ul.reasons li { margin:0 0 4px 0; }
          .chips span{ display:inline-block; padding:2px 8px; margin:2px 6px 2px 0; border:1px solid #2a2f37; background:#0f131a; border-radius:999px; font-size:12px; opacity:.9; }
          #td-scope [data-testid="stExpander"] details{ border-radius:10px; border:1px solid #2a2f37; background:#0c1016; }
          #td-scope [data-testid="stExpander"] summary{ padding:10px 12px; }
          #td-scope [data-testid="stExpander"] div[role="region"]{ padding:8px 12px 12px; }
        </style>
        """, unsafe_allow_html=True)

        summary_text = _extract_overall_summary(st.session_state.td_result) or "—"
        if summary_text and summary_text != "—":
            st.markdown('<div id="td-summary"><div class="ttl">Summary</div>', unsafe_allow_html=True)
            st.write(summary_text)
            st.markdown('</div>', unsafe_allow_html=True)

        lvl = _level_str(overall.get("risk_level") or overall.get("level") or "—")
        score = overall.get("risk_score", overall.get("score"))
        reasons = overall.get("risk_reasons") or overall.get("reasons") or []
        reasons_html = "<ul class='reasons'>" + "".join([f"<li>{r}</li>" for r in reasons[:6]]) + "</ul>" if reasons else ""
        st.markdown(f"""
        <div id="td-risk">
          <div class="header">
            <span class="badge {_badge_class(lvl)}">{lvl}</span>
            <div>
              <div class="title">Overall Risk</div>
              <div class="score">{'' if score in (None,'—') else f'Score {score}/100'}</div>
            </div>
          </div>
          {reasons_html}
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div id="td-scope">', unsafe_allow_html=True)
        for i, c in enumerate(clauses, start=1):
            idx = c.get("index", i) or i
            if c.get("labels"):
                try:
                    label0 = c["labels"][0][0] or "—"
                except Exception:
                    label0 = "—"
            else:
                label0 = c.get("label") or c.get("title") or "—"

            rlvl_raw = c.get("risk_level") or c.get("level") or "—"
            rlvl = _level_str(rlvl_raw)
            cscore = c.get("risk_score", c.get("score"))
            head = f"Clause {idx}: {label0} — {rlvl}{_fmt_score(cscore)}"
            expanded = isinstance(rlvl, str) and rlvl.lower().startswith("high")

            with st.expander(head, expanded=expanded):
                csum = (c.get("summary") or c.get("notes") or "").strip()
                if csum:
                    st.write("**In plain English:**", csum)

                creasons = (c.get("risk_reasons") or c.get("reasons") or [])
                if creasons:
                    st.caption("Why this matters:")
                    for rr in creasons[:3]:
                        st.write(f"• {rr}")
                    if len(creasons) > 3:
                        with st.expander(f"Show {len(creasons)-3} more reason(s)"):
                            for rr in creasons[3:6]:
                                st.write(f"• {rr}")

                ctext = (c.get("text") or c.get("raw_text") or "").strip()
                if ctext:
                    with st.expander("Show clause text"):
                        st.write(ctext)

                ents = c.get("entities") or []
                names = sorted({(e.get("text") or "").strip() for e in ents if isinstance(e, dict) and e.get("text")})
                if names:
                    with st.expander(f"Entities ({len(names)})"):
                        st.markdown(
                            '<div class="chips">' + " ".join(f"<span>{n}</span>" for n in names[:40]) + "</div>",
                            unsafe_allow_html=True
                        )
        st.markdown('</div>', unsafe_allow_html=True)

        col_dl, col_reset = st.columns([4, 1], vertical_alignment="center")
        with col_dl:
            if st.session_state.td_report_bytes:
                st.download_button(
                    label="⬇️ Download HTML report",
                    data=st.session_state.td_report_bytes,
                    file_name="contract-report_text_detailed.html",
                    mime="text/html",
                    key="td_download_btn",
                    use_container_width=True,
                )
        with col_reset:
            if st.button("Reset", key="td_reset_btn", help="Clear this result"):
                lock_tab("Text (Detailed)")
                for k in ("td_result", "td_report_bytes", "td_busy",
                          "prefill_text_detailed", "td_input"):
                    st.session_state.pop(k, None)
                reset_and_rerun("Text (Detailed)")

# =============================================================================
# Tab: Upload PDF (Quick)
# =============================================================================
elif TAB == "Upload PDF":
    st.markdown("### Upload PDF (Quick)")
    status_placeholder = st.empty()

    # --- tiny, local CSS ---
    st.markdown("""
    <style>
      .cc-card{ border:1px solid #2a2f37; border-radius:12px; background:#0f131a; padding:12px 14px; margin:10px 0 12px 0; }
      .cc-card .ttl{ font-weight:600; margin:0 0 8px 0; }
      .cc-card .score{ font-size:12px; opacity:.85; margin-top:2px; }
      .badge{ display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; font-weight:600; line-height:1; margin-right:8px; }
      .badge.low    { background:#10331e; color:#61d092; border:1px solid #1a6c3f; }
      .badge.medium { background:#2e2a12; color:#ffd166; border:1px solid #806b1a; }
      .badge.high   { background:#3a1515; color:#ff7b7b; border:1px solid #8b1f1f; }
      #pdfq-scope [data-testid="stExpander"] details{ border-radius:10px; border:1px solid #2a2f37; background:#0c1016; }
      #pdfq-scope [data-testid="stExpander"] summary{ padding:10px 12px; }
      #pdfq-scope [data-testid="stExpander"] div[role="region"]{ padding:8px 12px 12px; }
      .chips span{ display:inline-block; padding:2px 8px; margin:2px 6px 2px 0; border:1px solid #2a2f37; background:#0f131a; border-radius:999px; font-size:12px; opacity:.9; }
      .cc-summary p { margin:0 0 10px 0; line-height:1.55; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
<style>
/* Shared card look */
.cc-card {
  border:1px solid #2a2f37;
  border-radius:12px;
  background:#0f131a;
  padding:14px 16px;
  margin:12px 0 14px 0;
  line-height:1.55;
  font-size:14px;
  color:#e4e6eb;
}

/* Headings inside cards */
.cc-card .ttl,
.cc-card .hdr {
  font-weight:600;
  font-size:15px;
  margin:0 0 8px 0;
  line-height:1.3;
  display:flex;
  align-items:center;
  gap:8px;
  flex-wrap:wrap;
}

/* Risk badges unified across tabs */
.badge {
  display:inline-block;
  padding:3px 10px;
  border-radius:999px;
  font-size:12px;
  font-weight:700;
  line-height:1;
  letter-spacing:.2px;
}
.badge.low {
  background:#12351b;
  color:#7ceca3;
  border:1px solid #1f6137;
}
.badge.medium {
  background:#2e2a13;
  color:#ffd96c;
  border:1px solid #705e1c;
}
.badge.high {
  background:#3a1416;
  color:#ff8b8b;
  border:1px solid #6b1c21;
}

/* Score chip styling */
.chip {
  display:inline-block;
  padding:2px 8px;
  border-radius:8px;
  border:1px solid #2a2f37;
  font-size:12px;
  opacity:.9;
}

/* Unified paragraph rhythm */
.cc-card p {
  margin:0 0 10px 0;
  line-height:1.55;
  text-align:justify;
}

/* Light touch for lists */
.cc-card ul {
  margin:6px 0 6px 18px;
  padding:0;
  line-height:1.55;
}
.cc-card li {
  margin:2px 0;
}

/* Pills (tags / entities) */
.chips span,
.pill {
  display:inline-block;
  padding:4px 10px;
  margin:4px 6px 0 0;
  border:1px solid #2a2f37;
  background:#10151e;
  border-radius:999px;
  font-size:12px;
  opacity:.9;
}

/* Responsive grid for entity cards */
.grid {
  display:grid;
  grid-template-columns: repeat(auto-fill, minmax(160px,1fr));
  gap:8px;
}
.kv {
  border:1px solid #2a2f37;
  border-radius:10px;
  padding:8px 10px;
}
.kv .k { font-size:11px; opacity:.7; }
.kv .v { font-size:13px; font-weight:600; margin-top:2px; }

/* Sticky action area identical across tabs */
.sticky-actions {
  position:sticky;
  bottom:0;
  z-index:3;
  background:#0e1117;
  padding-top:8px;
  margin-top:8px;
  border-top:1px solid #2a2f37;
}
</style>
""", unsafe_allow_html=True)

    # --- local helper: paragraphize summary (NEW) ---
    def _qp_to_paragraphs(text: str | None) -> str:
        """
        Convert a flat summary string into <p> paragraphs.
        First split on blank lines; if none, split on single line breaks.
        """
        import html as _html
        t = (text or "").strip()
        if not t:
            return ""
        parts = [p.strip() for p in t.replace("\r","").split("\n\n") if p.strip()]
        if len(parts) <= 1:
            parts = [p.strip() for p in t.split("\n") if p.strip()]
        return "".join(f"<p>{_html.escape(p)}</p>" for p in parts)

    # --- state ---
    st.session_state.setdefault("pdfq_busy", False)
    st.session_state.setdefault("pdfq_result", None)
    st.session_state.setdefault("pdfq_report_bytes", None)
    st.session_state.setdefault("pdfq_uploader_key", 0)
    st.session_state.setdefault("pdfq_upload_bytes", b"")
    st.session_state.setdefault("_history_guard", set())  # guard for rerun duplicates
    # one-shot guard after forced rerun (prevents duplicate toasts)
    if st.session_state.pop("_pdfq_refresh_for_history", False):
        pass

    with st.form("form_pdf_quick", clear_on_submit=False):
        uploaded = st.file_uploader(
            "Choose a PDF",
            type=["pdf"],
            key=f"pdfq_uploader_{st.session_state.pdfq_uploader_key}",
            accept_multiple_files=False,
            help="Text-based PDFs work best (OCR not enabled in MVP)."
        )
        submit_pdfq = st.form_submit_button(
            "Analyze PDF",
            use_container_width=True,
            disabled=st.session_state.pdfq_busy,
        )

    if submit_pdfq:
        lock_tab("Upload PDF")   # keep view on this tab
        if not uploaded:
            st.warning("Please upload a PDF file.")
        else:
            st.session_state.pdfq_busy = True
            # ensure single status instance (fixes double grey box)
            status_placeholder.empty()
            status = status_placeholder.status("Analyzing PDF…", state="running", expanded=True)
            t0 = time.perf_counter()
            try:
                # cache the raw bytes for size checks / later reuse
                st.session_state.pdfq_upload_bytes = uploaded.getvalue()
                file_tuple = (uploaded.name, st.session_state.pdfq_upload_bytes, "application/pdf")

                status.write("Sending file to /analyze_pdf …")
                r = requests.post(f"{BACKEND}/analyze_pdf", files={"file": file_tuple}, timeout=120)
                r.raise_for_status()
                res = r.json() or {}

                # --- fallback for clauses/overall (skip for large files to keep Quick… quick) ---
                size_bytes = len(st.session_state.get("pdfq_upload_bytes") or b"")
                MAX_QUICK_FALLBACK_SIZE = 3_000_000  # ~3 MB

                if not res.get("clauses") and size_bytes <= MAX_QUICK_FALLBACK_SIZE:
                    try:
                        status.write("Fetching clause breakdown from /analyze_pdf_detailed …")
                        rd = requests.post(f"{BACKEND}/analyze_pdf_detailed", files={"file": file_tuple}, timeout=180)
                        rd.raise_for_status()
                        detailed = rd.json() or {}
                        if detailed.get("clauses"):
                            res["clauses"] = detailed["clauses"]
                        if not res.get("overall") and detailed.get("overall"):
                            res["overall"] = detailed["overall"]
                    except Exception:
                        pass

                st.session_state.pdfq_result = res

                status.write("Building printable report …")
                try:
                    r2 = requests.post(f"{BACKEND}/report_pdf_detailed", files={"file": file_tuple}, timeout=120)
                    r2.raise_for_status()
                    st.session_state.pdfq_report_bytes = r2.content
                except Exception:
                    st.session_state.pdfq_report_bytes = None

                # ----- RECENT: add item (success only, before rerun; de-duped) -----
                title = uploaded.name or "Upload PDF"
                hist_id = f"pdfq-{int(time.time())}"
                if hist_id not in st.session_state._history_guard:
                    # NEW: standard history entry (persists to history.json)
                    try:
                        if "add_history_entry" in globals():
                            add_history_entry(
                                kind="pdf_quick",
                                title=(title[:80] if title else "PDF (Quick)"),
                                payload={"result": st.session_state.get("pdfq_result")},
                                meta={
                                    "name": title,
                                    "size": len(st.session_state.get("pdfq_upload_bytes") or b""),
                                    "report": bool(st.session_state.get("pdfq_report_bytes")),
                                } if "meta" in add_history_entry.__code__.co_varnames else None  # safe if meta not supported
                            )
                    except Exception:
                        pass

                    # EXISTING: legacy/local history paths (kept intact)
                    try:
                        if "add_history_item" in globals():
                            add_history_item(
                                entry_type="pdf_quick",
                                title=title[:80],
                                payload=st.session_state.get("pdfq_result"),
                                meta={
                                    "name": title,
                                    "size": len(st.session_state.get("pdfq_upload_bytes") or b""),
                                    "report": bool(st.session_state.get("pdfq_report_bytes")),
                                },
                            )
                        elif "add_history" in globals():
                            add_history({
                                "id": hist_id,
                                "kind": "pdf_quick",
                                "title": title[:80],
                                "ts": datetime.utcnow().isoformat()
                            })
                    except Exception:
                        pass
                    st.session_state._history_guard.add(hist_id)

                # stay on tab + rerun so sidebar refreshes
                st.session_state["_select_tab"] = "Upload PDF"
                status.update(label=f"Done in {time.perf_counter()-t0:.1f}s ✓", state="complete")
                status_placeholder.empty()
                st.caption(f"Done in {time.perf_counter()-t0:.1f}s ✓")
                st.rerun()

            except requests.HTTPError as http_err:
                resp = http_err.response
                status.update(label="Analysis failed", state="error")
                st.error(f"Error contacting backend: {getattr(resp, 'status_code', '')} {getattr(resp, 'reason', '')}")
                try:
                    st.caption((resp.text or "")[:400])
                except Exception:
                    pass
                st.session_state.pdfq_result = None
                st.session_state.pdfq_report_bytes = None
            except Exception as e:
                status.update(label="Analysis failed", state="error")
                st.error(f"PDF analysis failed: {e}")
                st.session_state.pdfq_result = None
                st.session_state.pdfq_report_bytes = None
            finally:
                # clear status (prevents lingering second box)
                status_placeholder.empty()
                st.session_state.pdfq_busy = False

    if st.session_state.pdfq_result:
        res = st.session_state.pdfq_result or {}
        overall = (res.get("overall") or {})
        clauses = list(_get_clauses(res))

        s_txt = _qp_pick_summary(res)  # prefer concise summary if available
        if s_txt and s_txt != "—":
            # paragraphized summary (no blob)
            st.markdown('<div class="cc-card cc-summary"><div class="ttl">Summary</div>', unsafe_allow_html=True)
            st.markdown(_qp_to_paragraphs(s_txt), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Overall Risk (graceful empty reasons)
        lvl = _level_str(overall.get("risk_level") or overall.get("level") or "—")
        score = overall.get("risk_score", overall.get("score"))
        reasons = overall.get("risk_reasons") or overall.get("reasons") or res.get("risk_reasons") or []
        if reasons:
            reasons_html = "<ul style='margin:8px 0 0 16px'>" + "".join(f"<li>{r}</li>" for r in reasons[:6]) + "</ul>"
        else:
            reasons_html = "<div class='cc-subtle'>No key risks extracted in quick mode.</div>"

        st.markdown(f"""
        <div class="cc-card">
          <div class="ttl"><span class="badge {_badge_class(lvl)}">{lvl}</span> Overall Risk</div>
          <div class="score">{'' if score in (None,'—') else f'Score {score}/100'}</div>
          {reasons_html}
        </div>
        """, unsafe_allow_html=True)

        # --- Compact snapshot with Detailed-style controls ---
        def _one_liner(txt: str, n=240):
            if not txt: return "—"
            line = txt.strip().splitlines()[0]
            return (line[:n] + "…") if len(line) > n else line

        def _risk_score(c):
            v = c.get("risk_score", c.get("score"))
            try: return float(v)
            except: return -1.0

        clauses_all = list(clauses)

        from collections import Counter
        def _lev(c): return (c.get("risk_level") or c.get("level") or "low").lower()

        show_controls = len(clauses_all) >= 5
        if show_controls:
            st.divider()
            colA, colB, colC, colD = st.columns([2,1,1,1])
            with colA:
                q = st.text_input("Search clauses", key="pdfq_clause_query", placeholder="Label or text…")
            with colB:
                filt = st.selectbox("Show", ["All","High","Medium","Low"], key="pdfq_clause_filter")
            with colC:
                sort_mode = st.selectbox("Order", ["Document order","By risk (desc)"], key="pdfq_clause_sort")
            with colD:
                expand_all = st.checkbox("Expand all", key="pdfq_expand_all", value=False)

            st.markdown("<div style='opacity:.75;font-size:12px;margin:-6px 0 6px'>Filter or search (Quick shows a 5-clause preview).</div>", unsafe_allow_html=True)

            def keep_clause(c):
                rl = _lev(c)
                if filt != "All" and filt.lower() not in rl:
                    return False
                if q:
                    blob = " ".join([
                        str(c.get("label") or c.get("title") or ""),
                        str(c.get("summary") or c.get("notes") or ""),
                        str(c.get("text") or c.get("raw_text") or ""),
                    ]).lower()
                    return q.lower() in blob
                return True

            filtered = [c for c in clauses_all if keep_clause(c)]
            if sort_mode == "By risk (desc)":
                filtered = sorted(filtered, key=_risk_score, reverse=True)
        else:
            q, filt, sort_mode, expand_all = "", "All", "Document order", False
            filtered = clauses_all
            st.markdown("<div class='cc-subtle'>Quick view shows a 5-clause preview.</div>", unsafe_allow_html=True)

        topN = 5
        top_clauses = filtered[:topN]

        full_ct = Counter(_lev(c) for c in clauses_all)
        hi_all, md_all, lo_all = full_ct.get("high",0), full_ct.get("medium",0), full_ct.get("low",0)

        fct = Counter(_lev(c) for c in filtered)
        hi, md, lo = fct.get("high",0), fct.get("medium",0), fct.get("low",0)

        st.markdown(
            f"**Risk mix (filtered):** High {hi} · Medium {md} · Low {lo} — "
            f"**Clauses:** {len(filtered)}/{len(clauses_all)}  "
            f"<span style='opacity:.7'>(All: H {hi_all} · M {md_all} · L {lo_all})</span>",
            unsafe_allow_html=True
        )

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        st.markdown('<div id="pdfq-scope">', unsafe_allow_html=True)
        col_left, col_right = st.columns([2,1], vertical_alignment="top")

        with col_left:
            st.markdown('<div class="cc-card"><div class="ttl">Top clauses (preview)</div><div id="pdfq-top">', unsafe_allow_html=True)
            st.caption("Need the full clause-by-clause explorer? Use **Analyze in PDF (Detailed)**")
            if not top_clauses:
                st.markdown("<div class='cc-subtle'>No clause preview available in quick mode.</div>", unsafe_allow_html=True)
            else:
                for i, c in enumerate(top_clauses, 1):
                    idx   = c.get("index", i) or i
                    title = c.get("label") or c.get("title") or f"Clause {idx}"
                    rlvl  = _level_str(c.get("risk_level") or c.get("level") or "—")
                    sc    = c.get("risk_score", c.get("score"))

                    head  = f"Clause {idx}: {title} — {rlvl}{_fmt_score(sc)}"
                    expanded = expand_all or (isinstance(rlvl, str) and rlvl.lower().startswith("high"))

                    with st.expander(head, expanded=expanded):
                        one = _one_liner(c.get("summary") or c.get("notes") or c.get("text") or "")
                        if one and one != "—":
                            st.write("**In plain English:**", one)

                        creasons = (c.get("risk_reasons") or c.get("reasons") or [])
                        if creasons:
                            st.caption("Why this matters:")
                            for rr in creasons[:3]:
                                st.write(f"• {rr}")

                        ctext = (c.get("text") or c.get("raw_text") or "").strip()
                        if ctext:
                            with st.expander("Show clause text"):
                                st.write(ctext)

                        ents = c.get("entities") or []
                        names = sorted({(e.get("text") or "").strip() for e in ents if isinstance(e, dict) and e.get("text")})
                        if names:
                            with st.expander(f"Entities ({len(names)})"):
                                st.markdown('<div class="chips">' + " ".join(f"<span>{n}</span>" for n in names[:24]) + "</div>",
                                            unsafe_allow_html=True)
            st.markdown('</div></div>', unsafe_allow_html=True)

        with col_right:
            ents = res.get("entities") or []
            dates = res.get("dates") or []
            tags  = res.get("tags") or []

            if ents:
                names = sorted({(e.get("text") or "").strip() for e in ents if isinstance(e, dict) and (e.get("text") or "").strip()})
                st.markdown(f'<div class="cc-card"><div class="ttl">Entities ({len(names)})</div>', unsafe_allow_html=True)
                preview_names = names[:12]
                st.markdown('<div class="chips">' + " ".join(f"<span>{n}</span>" for n in preview_names) + "</div>", unsafe_allow_html=True)
                if len(names) > 12:
                    with st.expander(f"Show all entities ({len(names)})"):
                        st.markdown('<div class="chips">' + " ".join(f"<span>{n}</span>" for n in names) + "</div>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            if dates:
                st.markdown('<div class="cc-card"><div class="ttl">Key Dates</div>', unsafe_allow_html=True)
                for d in dates[:4]:
                    label = d.get("label") or "Date"
                    val   = d.get("value") or d.get("text") or ""
                    st.write(f"- **{label}**: {val}")
                st.markdown('</div>', unsafe_allow_html=True)

            if tags:
                st.markdown('<div class="cc-card"><div class="ttl">Tags</div>', unsafe_allow_html=True)
                st.write(", ".join(tags[:12]))
                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)  # close #pdfq-scope

        st.divider()
        col_dl, col_reset = st.columns([4,1], vertical_alignment="center")
        with col_dl:
            if st.session_state.get("pdfq_report_bytes"):
                st.download_button(
                    "⬇️ Download report (HTML)",
                    data=st.session_state.pdfq_report_bytes,
                    file_name="contract-report_pdf_quick.html",
                    mime="text/html",
                    key="pdfq_download_btn",
                    use_container_width=True,
                )
        with col_reset:
            if st.button("Reset", key="pdfq_reset_btn", help="Clear this result"):
                lock_tab("Upload PDF")
                for k in ("pdfq_result", "pdfq_summary", "pdfq_report_bytes", "pdfq_last_error", "pdfq_busy"):
                    st.session_state.pop(k, None)
                st.session_state["pdfq_uploader_key"] = st.session_state.get("pdfq_uploader_key", 0) + 1
                reset_and_rerun("Upload PDF")  # remounts uploader cleanly

# =============================================================================
# Tab: PDF (Detailed)
# =============================================================================
elif TAB == "PDF (Detailed)":
    st.markdown("### PDF (Detailed)")
    status_placeholder = st.empty()

    # --- local CSS atoms (match other tabs) -----------------------------------
    st.markdown("""
    <style>
      .cc-card{ border:1px solid #2a2f37; border-radius:12px; background:#0f131a; padding:12px 14px; margin:10px 0 12px; }
      .cc-card .ttl{ font-weight:600; margin:0 0 8px; }
      .cc-card .score{ font-size:12px; opacity:.85; margin-top:2px; }
      .badge{ display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; font-weight:600; line-height:1; margin-right:8px; }
      .badge.low    { background:#10331e; color:#61d092; border:1px solid #1a6c3f; }
      .badge.medium { background:#2e2a12; color:#ffd166; border:1px solid #806b1a; }
      .badge.high   { background:#3a1515; color:#ff7b7b; border:1px solid #8b1f1f; }
      #pdfd-scope [data-testid="stExpander"] details{ border-radius:10px; border:1px solid #2a2f37; background:#0c1016; }
      #pdfd-scope [data-testid="stExpander"] summary{ padding:10px 12px; }
      #pdfd-scope [data-testid="stExpander"] div[role="region"]{ padding:8px 12px 12px; }
      .chips span{ display:inline-block; padding:2px 8px; margin:2px 6px 2px 0; border:1px solid #2a2f37; background:#0f131a; border-radius:999px; font-size:12px; opacity:.9; }
      .sticky-actions{ position: sticky; bottom: 0; z-index: 3; background:#0e1117; padding-top:8px; margin-top:8px; border-top:1px solid #2a2f37; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
<style>
/* Shared card look */
.cc-card {
  border:1px solid #2a2f37;
  border-radius:12px;
  background:#0f131a;
  padding:14px 16px;
  margin:12px 0 14px 0;
  line-height:1.55;
  font-size:14px;
  color:#e4e6eb;
}

/* Headings inside cards */
.cc-card .ttl,
.cc-card .hdr {
  font-weight:600;
  font-size:15px;
  margin:0 0 8px 0;
  line-height:1.3;
  display:flex;
  align-items:center;
  gap:8px;
  flex-wrap:wrap;
}

/* Risk badges unified across tabs */
.badge {
  display:inline-block;
  padding:3px 10px;
  border-radius:999px;
  font-size:12px;
  font-weight:700;
  line-height:1;
  letter-spacing:.2px;
}
.badge.low { background:#12351b; color:#7ceca3; border:1px solid #1f6137; }
.badge.medium { background:#2e2a13; color:#ffd96c; border:1px solid #705e1c; }
.badge.high { background:#3a1416; color:#ff8b8b; border:1px solid #6b1c21; }

/* Score chip styling */
.chip { display:inline-block; padding:2px 8px; border-radius:8px; border:1px solid #2a2f37; font-size:12px; opacity:.9; }

/* Unified paragraph rhythm */
.cc-card p { margin:0 0 10px 0; line-height:1.55; text-align:justify; }

/* Light touch for lists */
.cc-card ul { margin:6px 0 6px 18px; padding:0; line-height:1.55; }
.cc-card li { margin:2px 0; }

/* Pills (tags / entities) */
.chips span, .pill {
  display:inline-block; padding:4px 10px; margin:4px 6px 0 0;
  border:1px solid #2a2f37; background:#10151e; border-radius:999px;
  font-size:12px; opacity:.9;
}

/* Responsive grid for entity cards */
.grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(160px,1fr)); gap:8px; }
.kv { border:1px solid #2a2f37; border-radius:10px; padding:8px 10px; }
.kv .k { font-size:11px; opacity:.7; }
.kv .v { font-size:13px; font-weight:600; margin-top:2px; }

/* Sticky action area identical across tabs */
.sticky-actions {
  position:sticky; bottom:0; z-index:3; background:#0e1117;
  padding-top:8px; margin-top:8px; border-top:1px solid #2a2f37;
}
</style>
""", unsafe_allow_html=True)

    # --- state -----------------------------------------------------------------
    st.session_state.setdefault("pdfd_busy", False)
    st.session_state.setdefault("pdfd_result", None)
    st.session_state.setdefault("pdfd_report_bytes", None)
    st.session_state.setdefault("pdfd_upload_bytes", b"")
    st.session_state.setdefault("_history_guard", set())  # de-dupe across reruns
    # one-shot refresh guard for RECENT cache-bust rerun
    if st.session_state.pop("_pdfd_refresh_for_history", False):
        pass

    # --- form ------------------------------------------------------------------
    with st.form("form_pdf_detailed", clear_on_submit=False):
        pdf_file_d = st.file_uploader(
            "Choose a PDF for clause-by-clause analysis:",
            type=["pdf"], accept_multiple_files=False, key="pdfd_uploader",
        )
        submit_pdfd = st.form_submit_button(
            "Analyze PDF (Detailed)",
            use_container_width=True,
            disabled=st.session_state.pdfd_busy,
        )

    # --- submit handling --------------------------------------------------------
    if submit_pdfd:
        lock_tab("PDF (Detailed)")
        if not pdf_file_d:
            st.warning("Please upload a PDF file.")
        else:
            st.session_state.pdfd_busy = True
            status_placeholder.empty()  # ensure single status instance
            status = status_placeholder.status("Analyzing…", state="running", expanded=True)
            t0 = time.perf_counter()
            try:
                # keep bytes in session (handy for size/debug or replays)
                st.session_state.pdfd_upload_bytes = pdf_file_d.getvalue()
                file_tuple = (pdf_file_d.name, st.session_state.pdfd_upload_bytes, "application/pdf")

                status.write("Sending file to /analyze_pdf_detailed …")
                r = requests.post(f"{BACKEND}/analyze_pdf_detailed", files={"file": file_tuple}, timeout=180)
                r.raise_for_status()
                st.session_state.pdfd_result = r.json() or {}

                status.write("Building printable report …")
                try:
                    r2 = requests.post(f"{BACKEND}/report_pdf_detailed", files={"file": file_tuple}, timeout=180)
                    r2.raise_for_status()
                    st.session_state.pdfd_report_bytes = r2.content
                except Exception:
                    st.session_state.pdfd_report_bytes = None

                # --------- RECENT entry (success only, before rerun; de-duped) ----------
                title = (pdf_file_d.name or "PDF (Detailed)")[:80]
                hist_id = f"pdfd-{int(time.time())}"
                if hist_id not in st.session_state._history_guard:
                    # NEW: standard history entry (persists to history.json)
                    try:
                        if "add_history_entry" in globals():
                            add_history_entry(
                                kind="pdf_detailed",
                                title=title,
                                payload={"result": st.session_state.get("pdfd_result")},
                                meta={
                                    "name": pdf_file_d.name,
                                    "size": len(st.session_state.get("pdfd_upload_bytes") or b""),
                                    "report": bool(st.session_state.get("pdfd_report_bytes")),
                                    "clauses": len(_get_clauses(st.session_state.pdfd_result or {})),
                                } if "meta" in add_history_entry.__code__.co_varnames else None  # safe if meta not supported
                            )
                    except Exception:
                        pass

                    # EXISTING: legacy/local history paths (kept intact)
                    try:
                        if "add_history_item" in globals():
                            add_history_item(
                                entry_type="pdf_detailed",
                                title=title,
                                payload=st.session_state.get("pdfd_result"),
                                meta={
                                    "name": pdf_file_d.name,
                                    "size": len(st.session_state.get("pdfd_upload_bytes") or b""),
                                    "report": bool(st.session_state.get("pdfd_report_bytes")),
                                    "clauses": len(_get_clauses(st.session_state.pdfd_result or {})),
                                },
                            )
                        elif "add_history" in globals():
                            add_history({
                                "id": hist_id,
                                "kind": "pdf_detailed",
                                "title": title,
                                "ts": datetime.utcnow().isoformat()
                            })
                    except Exception:
                        pass
                    st.session_state._history_guard.add(hist_id)

                # keep user on tab, refresh sidebar, then rerun
                st.session_state["_select_tab"] = "PDF (Detailed)"
                elapsed = time.perf_counter() - t0
                status.update(label=f"Done in {elapsed:.1f}s ✓", state="complete")
                status_placeholder.empty()
                st.caption(f"Done in {elapsed:.1f}s ✓")
                st.rerun()

            except requests.HTTPError as http_err:
                resp = http_err.response
                status.update(label="Analysis failed", state="error")
                st.error(f"Error contacting backend: {getattr(resp, 'status_code', '')} {getattr(resp, 'reason', '')}")
                try:
                    st.caption((resp.text or "")[:400])
                except Exception:
                    pass
                st.session_state.pdfd_result = None
                st.session_state.pdfd_report_bytes = None
            except Exception as e:
                status.update(label="Analysis failed", state="error")
                st.error(f"PDF detailed analyze failed: {e}")
                st.session_state.pdfd_result = None
                st.session_state.pdfd_report_bytes = None
            finally:
                status_placeholder.empty()
                st.session_state.pdfd_busy = False
                lock_tab("PDF (Detailed)")

    # --- anchor for gentle autofocus/scroll -----------------------------------
    st.markdown('<a id="pdfd-results"></a>', unsafe_allow_html=True)
    if st.session_state.get("pdfd_result"):
        st.markdown("""
        <script>document.getElementById('pdfd-results')?.scrollIntoView({behavior:'smooth', block:'start'});</script>
        """, unsafe_allow_html=True)

    # --- results ---------------------------------------------------------------
    pdres = st.session_state.get("pdfd_result")
    if pdres:
        res = pdres or {}
        overall = res.get("overall") or {}
        clauses = list(_get_clauses(res))

        # Summary
        s_txt = _qp_pick_summary(res)  # prefers detailed, rejects preamble
        if s_txt and s_txt != "—":
            st.markdown('<div class="cc-card cc-summary"><div class="ttl">Summary</div>', unsafe_allow_html=True)
            st.markdown(_qp_to_paragraphs(s_txt), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Overall Risk
        lvl = _level_str(overall.get("risk_level") or overall.get("level") or "—")
        score = overall.get("risk_score", overall.get("score"))
        reasons = overall.get("risk_reasons") or overall.get("reasons") or []
        reasons_html = (
            "<ul style='margin:8px 0 0 16px'>" + "".join(f"<li>{r}</li>" for r in reasons[:6]) + "</ul>"
        ) if reasons else ""
        st.markdown(f"""
        <div class="cc-card">
          <div class="ttl"><span class="badge {_badge_class(lvl)}">{lvl}</span> Overall Risk</div>
          <div class="score">{'' if score in (None,'—') else f'Score {score}/100'}</div>
          {reasons_html}
        </div>
        """, unsafe_allow_html=True)

        # Controls row
        from collections import Counter
        def _lev(c): return (c.get("risk_level") or c.get("level") or "low").lower()

        original = list(clauses)
        full_ct = Counter(_lev(c) for c in original)
        hi_all, md_all, lo_all = full_ct.get("high",0), full_ct.get("medium",0), full_ct.get("low",0)

        st.divider()
        colA, colB, colC, colD = st.columns([2,1,1,1])
        with colA:
            q = st.text_input("Search clauses", key="pdfd_clause_query", placeholder="Label or text…")
        with colB:
            filt = st.selectbox("Show", ["All","High","Medium","Low"], key="pdfd_clause_filter")
        with colC:
            order = st.selectbox("Order", ["Document order","By risk (desc)"], key="pdfd_clause_sort")
        with colD:
            expand_all = st.checkbox("Expand all", key="pdfd_expand_all", value=False)

        def keep(c):
            if filt != "All" and filt.lower() not in _lev(c): return False
            if q:
                blob = " ".join([
                    str(c.get("label") or c.get("title") or ""),
                    str(c.get("summary") or c.get("notes") or ""),
                    str(c.get("text") or c.get("raw_text") or ""),
                ]).lower()
                return q.lower() in blob
            return True

        clauses = [c for c in original if keep(c)]

        def rscore(c):
            v = c.get("risk_score", c.get("score"))
            try: return float(v)
            except: return -1.0

        if order == "By risk (desc)":
            clauses = sorted(clauses, key=rscore, reverse=True)

        fct = Counter(_lev(c) for c in clauses)
        hi, md, lo = fct.get("high",0), fct.get("medium",0), fct.get("low",0)
        st.markdown(
            f"**Risk mix (filtered):** High {hi} · Medium {md} · Low {lo} — "
            f"**Clauses:** {len(clauses)}/{len(original)}  "
            f"<span style='opacity:.7'>(All: H {hi_all} · M {md_all} · L {lo_all})</span>",
            unsafe_allow_html=True,
        )

        # Clause expanders
        st.markdown('<div id="pdfd-scope">', unsafe_allow_html=True)
        for i, c in enumerate(clauses, start=1):
            idx = c.get("index", i) or i
            label0 = c.get("label") or c.get("title") or "—"
            if c.get("labels"):
                try: label0 = c["labels"][0][0] or label0
                except Exception: pass

            rlvl_raw = c.get("risk_level") or c.get("level") or "—"
            rlvl = _level_str(rlvl_raw)
            cscore = c.get("risk_score", c.get("score"))
            head = f"Clause {idx}: {label0} — {rlvl}{_fmt_score(cscore)}"
            expanded = expand_all or (isinstance(rlvl, str) and rlvl.lower().startsWith("high"))

            with st.expander(head, expanded=bool(expanded)):
                csum = (c.get("summary") or c.get("notes") or "").strip()
                if csum: st.write("**In plain English:**", csum)

                creasons = (c.get("risk_reasons") or c.get("reasons") or [])
                if creasons:
                    st.caption("Why this matters:")
                    for rr in creasons[:3]: st.write(f"• {rr}")
                    if len(creasons) > 3:
                        with st.expander(f"Show {len(creasons)-3} more reason(s)"):
                            for rr in creasons[3:6]: st.write(f"• {rr}")

                ctext = (c.get("text") or c.get("raw_text") or "").strip()
                if ctext:
                    with st.expander("Show clause text"):
                        st.write(ctext)

                ents = c.get("entities") or []
                names = sorted({(e.get("text") or "").strip() for e in ents if isinstance(e, dict) and e.get("text")})
                if names:
                    with st.expander(f"Entities ({len(names)})"):
                        st.markdown('<div class="chips">' + " ".join(f"<span>{n}</span>" for n in names[:40]) + "</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Sticky bottom actions
        with st.container():
            st.markdown('<div class="sticky-actions">', unsafe_allow_html=True)
            col_dl, col_reset = st.columns([4,1], vertical_alignment="center")
            with col_dl:
                has_report = st.session_state.get("pdfd_report_bytes") is not None
                if has_report:
                    st.download_button(
                        "⬇️ Download report (HTML)",
                        data=st.session_state.pdfd_report_bytes,
                        file_name="contract-report_pdf_detailed.html",
                        mime="text/html",
                        key="pdfd_download_btn",
                        use_container_width=True,
                    )
                else:
                    st.button("⬇️ Download report (HTML)", key="pdfd_download_btn_disabled",
                              disabled=True, use_container_width=True)
            with col_reset:
                if st.button("Reset", key="pdfd_reset_btn", help="Clear this result"):
                    lock_tab("PDF (Detailed)")
                    for k in ("pdfd_result", "pdfd_report_bytes", "pdfd_busy"):
                        st.session_state.pop(k, None)
                    reset_and_rerun("PDF (Detailed)")
            st.markdown('</div>', unsafe_allow_html=True)


            
# =============================================================================
# Tab: Profile
# =============================================================================
elif TAB == "Profile":
    st.markdown("### Profile")
    prof = st.session_state.get("_profile") or load_profile()
    col1, col2 = st.columns([1,2])
    with col1:
        avatar_key = st.selectbox("avatar", list(AVATAR_CHOICES.keys()),
                                  index=list(AVATAR_CHOICES.keys()).index(prof["avatar_key"]))
        st.image(AVATAR_CHOICES[avatar_key], width=96)
    with col2:
        display_name = st.text_input("Display name", prof["display_name"])
        email = st.text_input("Email", prof["email"], disabled=True)

    st.markdown("---")
    st.subheader("Preferences")
    mode = st.radio("Default analysis mode", ["Quick", "Detailed"],
                    index=(0 if prof["default_mode"]=="quick" else 1), horizontal=True)
    show_reasons = st.toggle("Show risk badges and reasons", value=prof["show_reasons"])

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        if st.button("Save preferences", use_container_width=True):
            newp = {
                "display_name": display_name.strip() or "Pilot",
                "email": email,
                "avatar_key": avatar_key,
                "show_reasons": bool(show_reasons),
                "default_mode": "quick" if mode == "Quick" else "detailed",
            }
            save_profile(newp)
            st.session_state["_profile"] = newp
            st.success("Saved ✓")
    with c2:
        if st.button("Reset to defaults", use_container_width=True):
            save_profile({})
            st.session_state.pop("_profile", None)
            st.rerun()
    with c3:
        if st.button("Sign out", use_container_width=True):
            # whatever your sign-out action is
            st.session_state.clear()
            st.rerun()

# =============================================================================
# Footer
# =============================================================================
st.divider()
st.markdown(
    "<div style='text-align:center; color:#888; margin-top:4px;'>"
    "This is an AI assistant. Not legal advice."
    "</div>",
    unsafe_allow_html=True,
)