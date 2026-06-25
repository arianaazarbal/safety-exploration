"""Make the package importable when running scripts directly from repo root."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:  # load .env if python-dotenv is present
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass
