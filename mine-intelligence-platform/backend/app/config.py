from dataclasses import dataclass, field
import logging
import os
import secrets

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load configuration next to the backend code even when uvicorn is started elsewhere.
# load_dotenv() never overrides a variable that is already set in the real process
# environment, so this is a no-op (and harmless) on Render, where there is no .env
# file in the deployed repo and GROQ_API_KEY is injected directly into os.environ.
load_dotenv(os.path.join(BASE_DIR, ".env"))
load_dotenv()


def _clean_env(name: str, default: str = "") -> str:
    """Read an env var and strip stray whitespace/quotes.

    Platform dashboards (Render, etc.) store the value verbatim with no shell/.env
    parsing, so a value pasted as GROQ_API_KEY="gsk_xxx" (quotes included) or with a
    trailing newline/space becomes part of the literal secret and silently breaks
    Groq auth even though the same key works fine when typed directly into source.
    """
    value = os.getenv(name, default)
    if value is None:
        return default
    return value.strip().strip('"').strip("'")


@dataclass
class Settings:
    database_path: str = os.getenv("DATABASE_PATH", os.path.join(BASE_DIR, "data", "app.db"))
    data_dir: str = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
    upload_dir: str = os.getenv("UPLOAD_DIR", os.path.join(BASE_DIR, "data", "uploads"))
    documents_dir: str = os.getenv("DOCUMENTS_DIR", os.path.join(BASE_DIR, "data", "documents"))
    reports_dir: str = os.getenv("REPORTS_DIR", os.path.join(BASE_DIR, "data", "reports"))
    models_dir: str = os.getenv("MODELS_DIR", os.path.join(BASE_DIR, "models"))

    groq_api_key: str = field(default_factory=lambda: _clean_env("GROQ_API_KEY"))
    # llama3-8b-8192 was decommissioned by Groq; openai/gpt-oss-120b is the current default.
    # Override via GROQ_MODEL in .env if Groq's catalog changes again.
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    assistant_history_limit: int = int(os.getenv("ASSISTANT_HISTORY_LIMIT", "12"))

    cors_origins: str = os.getenv("CORS_ORIGINS", "http://localhost:5173")
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "25"))

    jwt_secret_key: str = field(default_factory=lambda: os.getenv("JWT_SECRET_KEY") or secrets.token_urlsafe(32))


settings = Settings()

if not os.getenv("JWT_SECRET_KEY"):
    logger.warning(
        "JWT_SECRET_KEY is not set in the environment. A random secret was generated for this "
        "process only, so all existing login tokens will be invalidated on every restart. "
        "Set JWT_SECRET_KEY in .env for stable sessions and production deployments."
    )

if not settings.groq_api_key:
    # Never crash the whole app over this: the assistant has a deterministic, non-LLM
    # fallback (see rag.py) and stays usable without Groq. But log loudly, since a
    # silent fallback is exactly what makes "works hardcoded, fails on Render" hard
    # to diagnose - this line should be the first thing you check in Render logs.
    logger.warning(
        "GROQ_API_KEY environment variable is not configured. The AI assistant will fall back "
        "to local, non-LLM phrasing for every answer. Set GROQ_API_KEY in the environment "
        "(locally: backend/.env; on Render: Service -> Environment -> Environment Variables) "
        "with no surrounding quotes and no trailing whitespace."
    )
else:
    logger.info(
        "GROQ_API_KEY is configured (length=%d). Groq model: %s",
        len(settings.groq_api_key),
        settings.groq_model,
    )
