"""Central configuration. Everything is overridable with environment variables (or a .env file)."""
import os
from pathlib import Path

try:  # optional: load a local .env if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = Path(os.getenv("DATA_RAW", ROOT / "data" / "raw"))
NOTES_DIR = DATA_RAW / "notes"  # optional extra docs (OA guides, interview notes: .md/.txt/.pdf)
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data" / "processed"))
DB_PATH = DATA_DIR / "placements.db"
INDEX_DIR = DATA_DIR / "index"
CHECKPOINT_DB = DATA_DIR / "checkpoints.sqlite"

PORTAL_PDF = DATA_RAW / "Placements'26.pdf"
STATS_PDF = DATA_RAW / "Placement Statistics Till Day 5.pdf"

# LLM: "openai", "google_genai", "ollama" ... or "none" for the offline rule-based mode.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "none").strip().lower()
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "45"))  # seconds per LLM call
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "1"))

# Embeddings in langchain init_embeddings format, e.g.
#   "openai:text-embedding-3-small", "ollama:nomic-embed-text",
#   "huggingface:sentence-transformers/all-MiniLM-L6-v2"
# Empty -> BM25 keyword retrieval only.
EMBEDDINGS = os.getenv("EMBEDDINGS", "").strip()

# Placement drive calendar: Day 1 of the 2025-26 drive was 1 Dec 2025.
DRIVE_DAY1 = os.getenv("DRIVE_DAY1", "2025-12-01")

# Rough conversion used ONLY to rank non-INR offers next to INR ones (raw currency is always kept).
FX_TO_INR = {"INR": 1.0, "AED": float(os.getenv("AED_TO_INR", "24.0")), "USD": float(os.getenv("USD_TO_INR", "88.0"))}
