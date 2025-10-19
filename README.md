# 🧾 Contract Co-Pilot
Upload contracts → Get clear, plain-English summaries + risk alerts — fast, accurate, and transparent.

---

## 🧭 Overview
**Contract Co-Pilot** analyzes contracts clause-by-clause and flags potential risks.  
Built with **FastAPI (backend)** and **Streamlit (frontend)**, it combines clause extraction, risk scoring, and plain-English explanations into one unified interface.

---

## 🧠 Features
- ⚡ **Quick Analysis** — Paste text or upload a PDF for an instant summary  
- 📑 **Detailed Mode** — Clause-by-clause risk cards with color-coded severity  
- 🧾 **Downloadable Reports** — Export HTML summaries for record-keeping or sharing  
- 🧰 **Streamlit UI** — Clean, tabbed layout with persistent state and dark-mode styling  
- 🧩 **FastAPI Backend** — Health checks, analysis endpoints, and AI-powered processing  
- 🔒 **Local-first Privacy** — All processing runs locally unless you deploy your own instance  

---

## 🚀 Quick Start

### 1️⃣ Clone the repo
```bash
git clone https://github.com/tazio121/contract-copilot.git
cd contract-copilot

2️⃣ Set up the environment

python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

3️⃣ Run the backend (FastAPI)

uvicorn backend.main:app --host 127.0.0.1 --port 8787 --reload

Open Swagger UI → http://127.0.0.1:8787/docs

4️⃣ Run the frontend (Streamlit)

streamlit run frontend/app.py --server.port 8501

App will open at → http://localhost:8501

```

🧾 Example Output

Each clause is identified (e.g., Termination, Indemnity, Confidentiality) and summarized in plain English with color-coded risk levels.



## 🧰 Tech Stack

    Frontend: Streamlit + custom CSS

    Backend: FastAPI + Pydantic + Uvicorn

    Parsing: PyMuPDF (fitz) for PDF extraction

    AI Engine: Transformers / OpenAI API compatible

    Language: Python 3.9+

## 🔐 Environment Variables

Create a .env file in the project root:

API_BASE=http://127.0.0.1:8787

## 🤝 Contributing

Pull requests are welcome!
1️⃣ Fork the repo
2️⃣ Create a feature branch (feature/your-feature)
3️⃣ Commit your changes
4️⃣ Push and open a Pull Request
📄 License

MIT License © 2025 Tazio Hussain
## 🏁 Project Origins

Started as a local prototype in VS Code to explore AI-driven contract analysis and risk scoring — now evolving into a full-stack tool for smarter, faster contract review.
