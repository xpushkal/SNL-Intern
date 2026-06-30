"""Tests validate the guaranteed deterministic core, so the LLM is forced OFF here
(stable, fast, no network/token usage). The LLM path is exercised by eval/ and by the
dedicated monkeypatched fallback tests.
"""
import os

# Must run before app.config is imported. load_dotenv(override=False) will then keep
# this empty value instead of reading GROQ_API_KEY from .env.
os.environ["GROQ_API_KEY"] = ""
