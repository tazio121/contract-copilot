# backend/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Any, Dict, List, Tuple
from datetime import datetime
from pathlib import Path
import base64
import io, re
from fastapi.responses import JSONResponse
import textwrap
import warnings
import logging
from transformers import pipeline, AutoTokenizer

# — Make transformers quieter —
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings(
    "ignore",
    message=r"Your max_length is set to .* Since this is a summarization task",
    category=UserWarning,
)

# — Init once (CPU) —
SUMM_MODEL_ID = "facebook/bart-large-cnn"  # or your existing model
_sum_tokenizer = AutoTokenizer.from_pretrained(SUMM_MODEL_ID)
SUMMARIZER = pipeline(
    "summarization",
    model=SUMM_MODEL_ID,
    tokenizer=SUMM_MODEL_ID,
    device=-1,  # CPU
)

def summarize_smart(text: str,
                    ratio: float = 0.45,
                    max_cap: int = 180,
                    min_cap: int = 20,
                    short_floor_tokens: int = 28) -> str:
    """
    Choose max/min_length based on input token length so we don't over-ask.
    - ratio: target fraction of input tokens for the summary
    - max_cap: absolute ceiling for max_length
    - min_cap: absolute floor for min_length
    """

    # Token length (cheap + accurate)
    toks = len(_sum_tokenizer.encode(text, add_special_tokens=False))

    # Very short inputs? Don't summarize; return as-is.
    if toks <= short_floor_tokens:
        return text

    # Target lengths
    max_len = max(min_cap, min(int(toks * ratio), max_cap))
    # min_length is typically ~40–60% of max_length to avoid ultra-short outputs
    min_len = max(min_cap, min(int(max_len * 0.6), max_len - 1))

    out = SUMMARIZER(
        text,
        max_length=max_len,
        min_length=min_len,
        do_sample=False,
        truncation=True,         # keep inputs within model limits
        no_repeat_ngram_size=3,  # cleaner summaries
    )[0]["summary_text"]
    return out

# -------- PDF parsing backend preference (PyMuPDF first) --------
try:
    import fitz  # PyMuPDF
    USE_PYMUPDF = True
    _FitzEmptyFileError = fitz.EmptyFileError
    _FitzFileDataError = fitz.FileDataError
except Exception:
    # If PyMuPDF isn't available, set a safe fallback
    USE_PYMUPDF = False
    fitz = None

    class _FitzEmptyFileError(Exception):
        pass

    class _FitzFileDataError(Exception):
        pass

# -------- App + static --------
app = FastAPI(title="Contract Co-Pilot API")

# Serve files from ./static at /static/...
app.mount("/static", StaticFiles(directory="static"), name="static")

# Favicon: serve static/favicon-32.png if present; else 204 (no icon)
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    path = Path("static/favicon-32.png")
    if path.exists():
        return FileResponse(path, media_type="image/png")
    return Response(status_code=204)

# Health + root
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"ok": True, "message": "Contract Co-Pilot API is running. See /docs and /health."}

# -------- Load/Embed brand logo once (offline-safe in reports) --------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../contract-copilot
LOGO_FILE = PROJECT_ROOT / "static" / "ccp-logo.png"
try:
    LOGO_DATA_URI = "data:image/png;base64," + base64.b64encode(LOGO_FILE.read_bytes()).decode("ascii")
except Exception:
    LOGO_DATA_URI = None  # if missing, report will omit logo unless meta overrides

# -------- Hugging Face pipelines (CPU-friendly models) --------
zsc = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")
ner = pipeline("token-classification", model="dslim/bert-base-NER", aggregation_strategy="simple")

CANDIDATE_LABELS = [
    "Termination","Liability","Indemnification","Confidentiality","Payment Terms",
    "Intellectual Property","Governing Law","Warranty","Non-compete","Arbitration",
]

# ---------- Risk Engine v2 (tuned) ----------
RULES: List[Tuple[str, int, str]] = [
    # Termination
    (r"terminate (this|the)?\s*(agreement|contract).{0,60}without cause", 70, "Termination without cause"),
    (r"terminate at any time", 70, "Termination at any time"),
    (r"termination.*upon\s*(\d+)\s*days?\s*notice", 0, "Notice period (for info)"),
    (r"\b(\d{1,2})\s*days?\s*notice\b", 15, "Short notice period (<30 days)"),

    # Liability / Indemnity
    (r"unlimited liability|liability[^.]{0,80}unlimited|liabilit(y|ies).{0,80}(no\s*cap|without\s*(a\s*)?cap|not\s*(be\s*)?subject\s*to\s*any\s*(cap|limit|limitation)|without\s*(limit|limitation)s?)",
     40, "Unlimited liability / no cap"),
    (r"hold harmless|defend and indemnify|indemnif(y|ication).*(including|for).{0,20}(negligence|acts)", 25, "Broad indemnity (incl. negligence)"),

    # Auto-renew / Perpetual / Exclusive / IP
    (r"automatic renewal|auto\-renew|renews automatically", 15, "Automatic renewal"),
    (r"\bperpetual\b|\bin perpetuity\b", 15, "Perpetual obligation / rights"),
    (r"\bexclusive\b (license|right|territory)", 15, "Exclusive rights"),
    (r"assignment of.*intellectual property|hereby assigns.*intellectual property", 25, "IP assignment"),

    # Venue / Jury
    (r"waive(s)? the right to jury trial|jury trial.*waiv", 10, "Jury trial waiver"),
    (r"venue shall be|exclusive jurisdiction|governing law.*exclusive", 10, "Exclusive venue/jurisdiction"),

    # Money / Confidentiality duration
    (r"liquidated damages|late fee(s)?|interest at (the )?rate of", 10, "Fees, interest, or liquidated damages"),
    (r"confidential(ity)?.{0,50}(indefinite|forever|perpetual)", 10, "Indefinite confidentiality"),
]

def score_risk(snippet: str) -> Dict:
    s = (snippet or "").lower()
    score = 0
    reasons: List[str] = []
    for pattern, pts, reason in RULES:
        if re.search(pattern, s, flags=re.DOTALL):
            score += pts
            reasons.append(reason)
    score = max(0, min(100, score))
    if score >= 60:
        level = "high"
    elif score >= 30:
        level = "medium"
    else:
        level = "low"
    return {"risk_score": score, "risk_level": level, "risk_reasons": sorted(set(reasons))}

def dynamic_lengths(text: str) -> Tuple[int, int]:
    wc = max(1, len(text.split()))
    max_len = min(180, max(40, wc // 2))
    min_len = min(80,  max(20, wc // 4))
    return max_len, min_len

def make_summary(text: str, max_len: int = 180, min_len: int = 60) -> str:
    """
    Wrapper that delegates to summarize_smart(), which auto-sets lengths
    based on input size and avoids Transformers 'max_length' warnings.
    """
    try:
        return summarize_smart(text)
    except Exception:
        # Robust fallback so PDFs never 500
        return textwrap.shorten((text or "").strip().replace("\n", " "), width=600, placeholder="…")

def tag_labels(text: str):
    z = zsc(text, CANDIDATE_LABELS, multi_label=True)
    pairs = list(zip(z["labels"], [float(x) for x in z["scores"]]))
    pairs = [(l, s) for (l, s) in pairs if s >= 0.55]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:5]

DATE_RE = re.compile(r"\b(?:\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s*\d{4})\b", re.I)
AMOUNT_RE = re.compile(r"\b(?:£|\$|€)\s?\d{1,3}(?:[,\s]\d{3})*(?:\.\d{2})?\b|\b\d+(?:\.\d{2})?\s?(?:USD|GBP|EUR)\b", re.I)

def _clean_ent_text(txt: str) -> str:
    return (txt or "").replace("##", "").replace(" .", ".").strip()

def extract_dates_amounts(text: str) -> Dict[str, list]:
    dates = DATE_RE.findall(text or "")
    amounts = AMOUNT_RE.findall(text or "")
    return {"dates": sorted(set(dates)), "amounts": sorted(set(amounts))}

def extract_entities(text: str):
    ents_raw = ner((text or "")[:1500])
    return [{
        "text": _clean_ent_text(e.get("word")),
        "label": e.get("entity_group"),
        "score": float(e.get("score", 0)),
        "start": int(e.get("start", 0)),
        "end": int(e.get("end", 0)),
    } for e in ents_raw]

# ---------- Clause splitting ----------
HEADLINE = re.compile(r"^\s*([A-Z][A-Z \-\d]{3,})\s*$")
def split_into_clauses(text: str, max_clauses: int = 18) -> List[str]:
    raw = re.sub(r"\r\n?", "\n", text or "")
    parts = [p.strip() for p in re.split(r"\n{2,}", raw) if p.strip()]
    clauses: List[str] = []
    for p in parts:
        lines = p.split("\n")
        chunk = []
        for ln in lines:
            if HEADLINE.match(ln) and chunk:
                clauses.append(" ".join(chunk).strip())
                chunk = [ln]
            else:
                chunk.append(ln)
        if chunk:
            p = " ".join(chunk).strip()
        if len(p) > 600:
            subparts = re.split(r"(?<=[.;])\s+(?=[A-Z])", p)
            for sp in subparts:
                if len(sp.strip()) >= 60:
                    clauses.append(sp.strip())
        else:
            if len(p) >= 40:
                clauses.append(p)
        if len(clauses) >= max_clauses:
            break
    if not clauses and (text or "").strip():
        clauses = [text.strip()]
    return clauses[:max_clauses]

# ---------- Core analyzers ----------
class AnalyzeTextIn(BaseModel):
    text: str

def analyze_core(snippet: str) -> Dict:
    snippet = (snippet or "").strip()
    if not snippet:
        raise HTTPException(status_code=400, detail="Empty text.")

    labels_scores = tag_labels(snippet)
    plain = make_summary(snippet)
    risk = score_risk(snippet) or {}
    entities = extract_entities(snippet)
    extras = extract_dates_amounts(snippet)

    # Ensure risk dict always has these keys
    risk_level = risk.get("risk_level")
    risk_score = risk.get("risk_score")
    risk_reasons = risk.get("risk_reasons") or []

    # fallback for older score_risk that doesn’t include reasons
    if not risk_reasons and isinstance(risk, dict):
        # simple heuristic-based extraction
        text_lower = snippet.lower()
        if "terminate" in text_lower:
            risk_reasons.append("Termination clause")
        if "liability" in text_lower:
            risk_reasons.append("Liability limitation")
        if "indemnif" in text_lower:
            risk_reasons.append("Indemnity clause")
        if "auto-renew" in text_lower or "renewal" in text_lower:
            risk_reasons.append("Automatic renewal")
        if "confidential" in text_lower:
            risk_reasons.append("Confidentiality / NDA")

    return {
        "labels": labels_scores[:5],
        "summary": plain,
        "risk_level": risk_level,
        "risk_score": risk_score,
        "risk_reasons": sorted(set(risk_reasons)),
        "entities": entities,
        "extracted": extras,
    }

def analyze_clauses(text: str) -> Dict:
    clauses = split_into_clauses(text)
    analyzed = []
    top_score = 0
    total = 0
    reasons_all: List[str] = []
    for idx, c in enumerate(clauses, 1):
        res = analyze_core(c)
        analyzed.append({
            "index": idx,
            "text": c,
            "summary": res["summary"],
            "labels": res["labels"],
            "risk_score": res["risk_score"],
            "risk_level": res["risk_level"],
            "risk_reasons": res["risk_reasons"],
            "entities": res["entities"],
        })
        total += res["risk_score"]
        top_score = max(top_score, res["risk_score"])
        reasons_all.extend(res["risk_reasons"])
    avg = int(round(total / max(1, len(analyzed))))
    overall_score = max(top_score, avg)
    overall_level = "low"
    if overall_score >= 70:
        overall_level = "high"
    elif overall_score >= 40:
        overall_level = "medium"
    return {
        "overall": {
            "risk_score": overall_score,
            "risk_level": overall_level,
            "risk_reasons": sorted(set(reasons_all)),
            "total_clauses": len(analyzed),
        },
        "clauses": analyzed
    }

# ---------- HTML report ----------
def build_html_report(doc_title: str, overall: Dict, clauses: List[Dict], meta: Dict = None) -> str:
    meta = meta or {}
    logo_url = meta.get("logo_url") or LOGO_DATA_URI  # default to embedded logo if available
    source_name = meta.get("source_name", "Text input")
    generated_at = meta.get("generated_at", "")
    css = """
    <style>
      body{font-family:Arial,Helvetica,sans-serif;max-width:940px;margin:40px auto;padding:20px;line-height:1.55;color:#111}
      h1{margin:0 0 8px 0}
      h2,h3{margin:10px 0 6px 0}
      .muted{color:#666;font-size:.95rem}
      .card{border:1px solid #eee; border-radius:12px; padding:16px; margin:14px 0; background:#fff}
      .badge{display:inline-block;padding:4px 10px;border-radius:999px;color:#fff;font-weight:700;font-size:.9rem}
      .high{background:#e11d48}.medium{background:#f59e0b}.low{background:#10b981}
      .small{font-size:.9rem}
      code{background:#f6f6f6;padding:2px 6px;border-radius:6px}
      p{margin:6px 0}
      .header{display:flex;align-items:center;gap:12px;margin-bottom:8px}
      .header img{height:64px; width:auto; max-width:280px}
@media print{ .header img{height:72px} }
      .meta{display:flex;gap:16px;flex-wrap:wrap}
      .meta div{color:#444;font-size:.92rem}
      @page{ size:A4; margin:12mm }
      @media print{
        @page{ margin:0 }
        body{ max-width:none; margin:12mm; padding:0 }
        .card{ page-break-inside:avoid; break-inside:avoid }
        h1,h2{ page-break-after:avoid }
        a[href]:after{ content:"" }
        .muted{ color:#333 }
      }
    </style>
    """
    level = (overall.get("risk_level") or "low").lower()
    score = int(overall.get("risk_score", 0))
    reasons = ", ".join(overall.get("risk_reasons", [])) or "—"
    logo_html = f"<img src='{logo_url}' alt='Logo'/>" if logo_url else ""
    header = f"""
      <div class="header">{logo_html}<h1>{doc_title}</h1></div>
      <div class="meta">
        <div><strong>Source:</strong> {source_name}</div>
        <div><strong>Generated:</strong> {generated_at}</div>
      </div>
      <div class="card">
        <h2>Overall Risk: <span class="badge {level}">{level.title()} — {score}/100</span></h2>
        <p class="muted">Reasons: {reasons}</p>
        <p class="muted">Total clauses: {overall.get('total_clauses', len(clauses))}</p>
      </div>
    """
    body = ""
    for c in clauses:
        tags = ", ".join([f"{l} ({s:.2f})" for l, s in c.get("labels", [])]) or "—"
        rr = ", ".join(c.get("risk_reasons", [])) or "—"
        ents = ", ".join([f"{e['text']}·{e['label']}" for e in c.get("entities", [])[:10]]) or "—"
        body += f"""
          <div class="card">
            <h3>Clause {c['index']} — <span class="badge {c['risk_level']}">{c['risk_level'].title()} {c['risk_score']}/100</span></h3>
            <p><strong>Text:</strong> {c['text']}</p>
            <p><strong>Summary:</strong> {c['summary']}</p>
            <p class="small"><strong>Risk reasons:</strong> {rr}</p>
            <p class="small"><strong>Tags:</strong> {tags}</p>
            <p class="small"><strong>Entities:</strong> {ents}</p>
          </div>
        """
    html = f"<!doctype html><html><head><meta charset='utf-8'><title>{doc_title}</title>{css}</head><body>{header}{body}</body></html>"
    return html

# ---------- Helpers for safer text & PDF parsing ----------
def _safe_trim(s: str, n: int) -> str:
    if not s:
        return s
    s = s.strip()
    return s[:n]

def _extract_text_pymupdf(pdf_bytes: bytes) -> str:
    # Only call this if PyMuPDF is available
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            if getattr(doc, "needs_pass", False):
                # Client issue: encrypted PDF
                raise HTTPException(status_code=400, detail="PDF is password-protected.")
            parts = []
            for p in doc:
                parts.append(p.get_text("text"))
            return ("\n".join(parts)).strip()
    except _FitzEmptyFileError:
        raise HTTPException(status_code=400, detail="Empty or invalid PDF file.")
    except _FitzFileDataError as e:
        # Let caller try fallback
        raise e
    except Exception as e:
        # Unexpected primary parser issue → let caller try fallback
        raise RuntimeError(f"PyMuPDF parse error: {e}")

def _extract_text_pdfminer(pdf_bytes: bytes) -> str:
    try:
        from pdfminer.high_level import extract_text
        return (extract_text(io.BytesIO(pdf_bytes)) or "").strip()
    except Exception as e:
        raise RuntimeError(f"PDFMiner parse error: {e}")

def _extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Primary: PyMuPDF (if available), fallback: pdfminer.six.
    Raises HTTPException for client errors (empty, password, no text).
    """
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    text = ""

    if USE_PYMUPDF and fitz is not None:
        try:
            text = _extract_text_pymupdf(pdf_bytes)
        except _FitzFileDataError:
            text = _extract_text_pdfminer(pdf_bytes)
        except RuntimeError:
            text = _extract_text_pdfminer(pdf_bytes)
    else:
        try:
            text = _extract_text_pdfminer(pdf_bytes)
        except RuntimeError:
            # Last resort: try PyMuPDF if present
            if fitz is not None:
                try:
                    text = _extract_text_pymupdf(pdf_bytes)
                except _FitzFileDataError:
                    raise HTTPException(status_code=400, detail="Invalid or corrupted PDF file.")
                except RuntimeError as e2:
                    raise HTTPException(status_code=400, detail=str(e2))
            else:
                raise HTTPException(status_code=400, detail="Could not parse PDF (no parser available).")

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Could not extract any text from the PDF. If it's a scan, run OCR first."
        )
    return text

# ---------- Public endpoints ----------
@app.post("/analyze_text")
def analyze_text(payload: AnalyzeTextIn):
    # same behavior, now with safe-trim and clear 4xx on empty
    text = _safe_trim(payload.text or "", 4000)
    if not text:
        raise HTTPException(status_code=400, detail="No text provided.")
    return analyze_core(text)

@app.post("/analyze_text_detailed")
def analyze_text_detailed(payload: AnalyzeTextIn):
    # same behavior, now with safe-trim and clear 4xx on empty
    text = _safe_trim(payload.text or "", 10000)
    if not text:
        raise HTTPException(status_code=400, detail="No text provided.")
    return analyze_clauses(text)

@app.post("/analyze_pdf")
async def analyze_pdf(file: UploadFile = File(...)):
    if not file or not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF.")
    content = await file.read()
    try:
        text = _extract_text_from_pdf(content)
    except HTTPException:
        # pass through known client-side issues (password/empty/no text)
        raise
    except Exception as e:
        # unexpected server-side issue
        raise HTTPException(status_code=500, detail=f"PDF parse error: {e}")
    snippet = _safe_trim(text, 6000)
    if not snippet:
        raise HTTPException(status_code=400, detail="Could not extract text from PDF.")
    return analyze_core(snippet)

@app.post("/analyze_pdf_detailed")
async def analyze_pdf_detailed(file: UploadFile = File(...)):
    if not file or not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF.")
    content = await file.read()
    try:
        text = _extract_text_from_pdf(content)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF parse error: {e}")
    return analyze_clauses(_safe_trim(text, 20000))

@app.post("/report_text_detailed")
def report_text_detailed(payload: AnalyzeTextIn):
    text = _safe_trim(payload.text or "", 20000)
    if not text:
        raise HTTPException(status_code=400, detail="No text provided.")
    res = analyze_clauses(text)
    meta = {
        # No need to pass logo_url; build_html_report defaults to embedded LOGO_DATA_URI
        "source_name": "Text input",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    html = build_html_report("Contract Report (Text)", res["overall"], res["clauses"], meta)
    fname = f"contract-report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    return Response(content=html, media_type="text/html",
                    headers={"Content-Disposition": f'attachment; filename="{fname}"'})

@app.post("/report_pdf_detailed")
async def report_pdf_detailed(file: UploadFile = File(...)):
    if not file or not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF.")
    content = await file.read()
    try:
        text = _extract_text_from_pdf(content)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF parse error: {e}")
    res = analyze_clauses(_safe_trim(text, 20000))
    meta = {
        "source_name": file.filename,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    html = build_html_report("Contract Report (PDF)", res["overall"], res["clauses"], meta)
    fname = f"contract-report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    return Response(content=html, media_type="text/html",
                    headers={"Content-Disposition": f'attachment; filename="{fname}"'})