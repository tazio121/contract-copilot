# frontend/supa.py
import os
from dotenv import load_dotenv
from supabase import create_client, Client
from pathlib import Path


# Load .env that sits next to app.py/supa.py (local dev)
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

try:
    from supabase import create_client
except Exception:
    create_client = None  # allows repo to run even if package missing

def get_supa():
    """
    Returns a Supabase client if SUPABASE_URL and SUPABASE_ANON_KEY exist,
    otherwise returns None (Guest Mode).
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")
    if not url or not key or not create_client:
        return None
    return create_client(url, key)

