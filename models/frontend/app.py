# frontend/app.py
# ---- MUST be the first Streamlit call ----
import streamlit as st
st.set_page_config(page_title="Contract Co-Pilot", page_icon="🧾", layout="centered")

# ---- Std imports ----
import os
import requests
from datetime import datetime

# ---- Auth imports (matches your auth.py) ----
from supa import get_supa
from auth import render_auth_modal, get_user_id, bootstrap_session, sign_out

# ---- Helpers ----
def _clear_auth_overlay():
    # Removes any dim/overlay CSS your modal might add
    st.markdown("""
    <style>
      [data-testid="stAppViewContainer"] .main { filter:none !important; pointer-events:auto !important; }
      [data-testid="stSidebar"] .block-container { filter:none !important; pointer-events:auto !important; }
      .auth-dim::before { display:none !important; }
    </style>
    """, unsafe_allow_html=True)

# ---- Settings (constants only) ----
BACKEND = os.getenv("BACKEND", "http://127.0.0.1:8787")
LOGO_URL = f"{BACKEND}/static/ccp-logo.png"

# ---- Styles (safe to run before app content) ----
st.markdown(f"""
<style>
/* Make the sidebar a positioning context */
[data-testid="stSidebar"] {{
  position: relative;
}}
/* Small logo in the sidebar chrome */
[data-testid="stSidebar"]::before {{
  content: "";
  position: absolute;
  top: 10px;
  left: 12px;
  width: 62px;
  height: 62px;
  background: url('{LOGO_URL}') no-repeat center center / contain;
  border-radius: 6px;
  opacity: 0.98;
}}
[data-testid="stSidebar"] .block-container {{
  padding-top: 48px;
}}
@media (max-width: 900px) {{
  [data-testid="stSidebar"]::before {{ top: 8px; left: 10px; width: 24px; height: 24px; }}
  [data-testid="stSidebar"] .block-container {{ padding-top: 44px; }}
}}
/* Centered header + centered tabs */
.ccp-hero {{ text-align: center; margin: 6px 0 8px 0; }}
.ccp-hero h1 {{ margin: 0; padding: 0; font-size: 2.2rem; line-height: 1.1; }}
.ccp-hero .sub {{ color: #888; margin: 4px 0 0 0; line-height: 1.3; }}
div[role="tablist"] {{ justify-content: center !important; margin-top: 6px !important; margin-bottom: 8px !important; }}
</style>
""", unsafe_allow_html=True)

# ---- Auth bootstrap ----
_S = get_supa()
bootstrap_session(_S)  # try silent restore (memory or disk)

uid = get_user_id()
st.sidebar.caption(f"Streamlit {st.__version__}")
st.sidebar.caption(f"debug uid: {uid or '—'}")

# If not logged in, show your otp modal and stop the app render
if not uid:
    render_auth_modal(_S)   # this is your existing modal flow (email + 6-digit code)
    st.stop()

# Logged in → ensure any previous overlay is removed
_clear_auth_overlay()

# Sidebar sign-out
with st.sidebar:
    if st.button("Sign out", use_container_width=True):
        sign_out(_S)  # clears tokens + reruns inside your auth.py
        st.stop()

# === Text-only header (centered) ===
st.markdown(
    """
    <div class="ccp-hero">
      <h1>Contract Co-Pilot</h1>
      <div class="sub">Upload a contract to get plain-English summaries, risk alerts, tags &amp; entities.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- Session state (history & prefill) ----
if "history" not in st.session_state:
    st.session_state["history"] = []  # list of dicts: {type, title, payload, ts}

def add_history(entry: dict):
    entry = dict(entry)
    entry["ts"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    # keep newest first, max 20
    st.session_state["history"] = [entry] + st.session_state["history"][:19]

# ---- Sidebar: Recent entries ----
with st.sidebar:
    st.caption("Created by Contract Co-Pilot")
    st.markdown("### Recent entries")

    entries = st.session_state.get("history", [])
    if not entries:
        st.caption("No entries yet.")
    else:
        labels = [
            f"{e['ts']} — {'PDF' if e['type']=='pdf' else 'Text'} — {e['title']}"
            for e in entries
        ]
        sel = st.radio("Recent", labels, index=0, label_visibility="collapsed")
        idx = labels.index(sel)
        chosen = entries[idx]

        if chosen["type"] == "text":
            if st.button("Load into Text (Detailed)", use_container_width=True):
                st.session_state["prefill_text_detailed"] = chosen["payload"][:15000]
                st.success("Loaded into the Text (Detailed) tab ↑")

        if st.button("Clear history", use_container_width=True):
            st.session_state["history"] = []
            st.experimental_rerun()

# ------------------ Helper to render quick results ------------------
def show_simple(data: dict):
    st.subheader("Summary")
    st.write(data.get("summary", ""))

    st.subheader("Risk")
    rl = (data.get("risk_level", "low") or "low").lower()
    score = data.get("risk_score")
    badge = {"high": "🔴 High", "medium": "🟠 Medium", "low": "🟢 Low"}.get(rl, rl)
    st.write(badge)
    if isinstance(score, (int, float)):
        st.progress(min(100, max(0, int(score))), text=f"Risk score: {int(score)}/100")

    reasons = data.get("risk_reasons") or []
    if reasons:
        st.write("**Reasons:** " + ", ".join(reasons))

    st.subheader("Top Tags")
    for label, sc in data.get("labels", []):
        try:
            st.write(f"- **{label}** ({float(sc):.2f})")
        except Exception:
            st.write(f"- **{label}** ({sc})")

    ents = data.get("entities", [])
    if ents:
        st.subheader("Entities")
        for e in ents[:20]:
            try:
                st.write(f"- `{e['text']}` — {e['label']} ({float(e['score']):.2f})")
            except Exception:
                st.write(f"- `{e.get('text','?')}` — {e.get('label','?')}")

# ------------------ Tabs ------------------
tab_text, tab_text_detail, tab_pdf, tab_pdf_detail = st.tabs(
    ["Paste Text", "Text (Detailed)", "Upload PDF", "PDF (Detailed)"]
)

# ------------------ Paste Text (quick) ------------------
with tab_text:
    with st.form("text_form"):
        user_text = st.text_area(
            "Paste a contract clause or short section:",
            height=180,
            placeholder="Paste text here…",
        )
        go = st.form_submit_button("Analyze Text")
    if go:
        if not user_text.strip():
            st.warning("Please paste some text.")
        else:
            with st.spinner("Analyzing…"):
                try:
                    resp = requests.post(f"{BACKEND}/analyze_text", json={"text": user_text}, timeout=60)
                    resp.raise_for_status()
                    data = resp.json()
                    show_simple(data)
                    title = user_text.strip().splitlines()[0][:60] + ("…" if len(user_text.strip()) > 60 else "")
                    add_history({"type": "text", "title": title, "payload": user_text})
                except Exception as e:
                    st.error(f"Error contacting backend: {e}")

# ------------------ Text (Detailed) ------------------
with tab_text_detail:
    with st.form("text_detail_form"):
        user_text2 = st.text_area(
            "Paste longer contract text (multiple clauses):",
            height=240,
            placeholder="Paste full section or multiple clauses…",
            value=st.session_state.get("prefill_text_detailed", ""),
        )
        go2 = st.form_submit_button("Analyze Text (Detailed)")
    if go2:
        if not user_text2.strip():
            st.warning("Please paste some text.")
        else:
            with st.spinner("Analyzing (detailed)…"):
                try:
                    resp = requests.post(f"{BACKEND}/analyze_text_detailed", json={"text": user_text2}, timeout=120)
                    resp.raise_for_status()
                    data = resp.json()

                    overall = data.get("overall", {})
                    score = int(overall.get("risk_score", 0))
                    level = (overall.get("risk_level", "low") or "low").lower()
                    st.subheader("Overall Risk")
                    st.progress(min(100, max(0, score)), text=f"Overall risk: {score}/100 ({level})")
                    reasons = overall.get("risk_reasons") or []
                    if reasons:
                        st.write("**Reasons:** " + ", ".join(reasons))

                    st.subheader("Clauses")
                    for c in data.get("clauses", []):
                        with st.expander(f"Clause {c['index']} — {c['risk_level'].upper()} ({c['risk_score']}/100)"):
                            st.write(c["text"])
                            st.markdown("**Summary**")
                            st.write(c["summary"])
                            if c.get("risk_reasons"):
                                st.markdown("**Risk reasons:** " + ", ".join(c["risk_reasons"]))
                            if c.get("labels"):
                                st.markdown("**Tags:** " + ", ".join([f"{l} ({s:.2f})" for l, s in c["labels"]]))
                            ents = c.get("entities") or []
                            if ents:
                                st.markdown("**Entities:**")
                                for e in ents[:10]:
                                    try:
                                        st.write(f"- `{e['text']}` — {e['label']} ({float(e['score']):.2f})")
                                    except Exception:
                                        st.write(f"- `{e.get('text','?')}` — {e.get('label','?')}")

                    rep = requests.post(f"{BACKEND}/report_text_detailed", json={"text": user_text2}, timeout=120)
                    if rep.ok and rep.content:
                        st.download_button(
                            "⬇️ Download HTML report",
                            data=rep.content,
                            file_name="contract-report.html",
                            mime="text/html",
                        )
                    else:
                        st.info("Report generation available, but no file returned.")

                    title = user_text2.strip().splitlines()[0][:60] + ("…" if len(user_text2.strip()) > 60 else "")
                    add_history({"type": "text", "title": title, "payload": user_text2})
                except Exception as e:
                    st.error(f"Error contacting backend: {e}")

# ------------------ Upload PDF (quick) ------------------
with tab_pdf:
    uploaded = st.file_uploader("Upload a contract PDF", type=["pdf"], key="pdf_simple")
    if st.button("Analyze PDF", disabled=uploaded is None):
        if not uploaded:
            st.warning("Please upload a PDF.")
        else:
            pdf_bytes = uploaded.getvalue()
            with st.spinner("Analyzing PDF…"):
                try:
                    files = {"file": (uploaded.name, pdf_bytes, "application/pdf")}
                    resp = requests.post(f"{BACKEND}/analyze_pdf", files=files, timeout=120)
                    resp.raise_for_status()
                    show_simple(resp.json())
                    add_history({"type": "pdf", "title": uploaded.name, "payload": ""})
                except Exception as e:
                    st.error(f"Error contacting backend: {e}")

# ------------------ PDF (Detailed) ------------------
with tab_pdf_detail:
    uploaded2 = st.file_uploader("Upload a contract PDF for detailed analysis", type=["pdf"], key="pdf_detail")
    if st.button("Analyze PDF (Detailed)", disabled=uploaded2 is None):
        if not uploaded2:
            st.warning("Please upload a PDF.")
        else:
            pdf_bytes2 = uploaded2.getvalue()
            with st.spinner("Analyzing PDF (Detailed)…"):
                try:
                    files = {"file": (uploaded2.name, pdf_bytes2, "application/pdf")}
                    resp = requests.post(f"{BACKEND}/analyze_pdf_detailed", files=files, timeout=180)
                    resp.raise_for_status()
                    data = resp.json()

                    overall = data.get("overall", {})
                    score = int(overall.get("risk_score", 0))
                    level = (overall.get("risk_level", "low") or "low").lower()
                    st.subheader("Overall Risk")
                    st.progress(min(100, max(0, score)), text=f"Overall risk: {score}/100 ({level})")
                    reasons = overall.get("risk_reasons") or []
                    if reasons:
                        st.write("**Reasons:** " + ", ".join(reasons))

                    st.subheader("Clauses")
                    for c in data.get("clauses", []):
                        with st.expander(f"Clause {c['index']} — {c['risk_level'].upper()} ({c['risk_score']}/100)"):
                            st.write(c["text"])
                            st.markdown("**Summary**")
                            st.write(c["summary"])
                            if c.get("risk_reasons"):
                                st.markdown("**Risk reasons:** " + ", ".join(c["risk_reasons"]))
                            if c.get("labels"):
                                st.markdown("**Tags:** " + ", ".join([f"{l} ({s:.2f})" for l, s in c["labels"]]))
                            ents = c.get("entities") or []
                            if ents:
                                st.markdown("**Entities:**")
                                for e in ents[:10]:
                                    try:
                                        st.write(f"- `{e['text']}` — {e['label']} ({float(e['score']):.2f})")
                                    except Exception:
                                        st.write(f"- `{e.get('text','?')}` — {e.get('label','?')}")

                    rep = requests.post(
                        f"{BACKEND}/report_pdf_detailed",
                        files={"file": (uploaded2.name, pdf_bytes2, "application/pdf")},
                        timeout=180,
                    )
                    if rep.ok and rep.content:
                        st.download_button(
                            "⬇️ Download HTML report",
                            data=rep.content,
                            file_name="contract-report.html",
                            mime="text/html",
                        )
                    else:
                        st.info("Report generation available, but no file returned.")

                    add_history({"type": "pdf", "title": uploaded2.name, "payload": ""})
                except Exception as e:
                    st.error(f"Error contacting backend: {e}")

# ------------------ Footer ------------------
st.divider()
st.markdown(
    "<div style='text-align:center; color:#888; margin-top:4px;'>"
    "This is an AI assistant. Not legal advice."
    "</div>",
    unsafe_allow_html=True,
)