# backend/run.py
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8787,
        reload=True,
        reload_excludes=["frontend/*", ".venv/*", "venv/*"],
    )