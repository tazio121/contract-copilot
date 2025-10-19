# frontend/supa.py
import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()
_SUPA_URL = os.getenv("SUPABASE_URL")
_SUPA_KEY = os.getenv("SUPABASE_ANON_KEY")

if not _SUPA_URL or not _SUPA_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_ANON_KEY in environment.")

def get_supa() -> Client:
    return create_client(_SUPA_URL, _SUPA_KEY)