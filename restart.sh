#!/bin/bash
cd ~/contract-copilot
source venv/bin/activate

echo "🔴 Stopping old services..."
pkill -f uvicorn 2>/dev/null || true
pkill -f streamlit 2>/dev/null || true
sleep 1

echo "🚀 Starting backend (FastAPI)..."
nohup uvicorn backend.main:app --host 0.0.0.0 --port 8787 --reload > backend.log 2>&1 &

echo "🚀 Starting frontend (Streamlit)..."
nohup streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0 > frontend.log 2>&1 &

echo "✅ Restart complete — backend:8787  |  frontend:8501"
