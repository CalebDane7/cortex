"""Centralized configuration for Cortex memory system.

All paths, thresholds, and tuning constants live here. Override via
environment variables or by importing and reassigning before use.
"""

import os
from pathlib import Path

# WHY: Single source of truth for all paths. Every module imports from here
# instead of hardcoding ~/.ai-controller/memory/ or ~/.cortex/.
MEMORY_DIR = Path(os.environ.get("CORTEX_MEMORY_DIR", str(Path.home() / ".cortex")))

# Core data files
CORE_MEMORY_FILE = MEMORY_DIR / "MEMORY.md"
MEMORY_LOG_FILE = MEMORY_DIR / "memory_log.jsonl"
CORE_TAGGED_FILE = MEMORY_DIR / "core_tagged.jsonl"
ARCHIVE_DIR = MEMORY_DIR / "archive"
ALIASES_FILE = MEMORY_DIR / "aliases.json"
CACHE_DB = MEMORY_DIR / "memory_cache.db"
DIRTY_MARKER = MEMORY_DIR / ".cache-dirty"
EXPANSION_CACHE_FILE = MEMORY_DIR / ".query-expansion-cache.json"

# Logging
LOG_DIR = MEMORY_DIR / "logs"
PIPELINE_LOG = LOG_DIR / "memory-pipeline.log"

# Scoring constants
# WHY 0.03: At this rate, a 30-day-old entry retains 41% of its score,
# 90-day retains 7%. Fast enough to prefer recent knowledge, slow enough
# that important corrections don't vanish in a week.
DECAY_RATE = float(os.environ.get("CORTEX_DECAY_RATE", "0.03"))

# Injection thresholds
# WHY 0.3 for HOT: Below this, too many false positives get full injection.
# WHY 0.15 for WARM: Raised from 0.1 — at 2000+ entries, 0.1 produces noise.
HOT_THRESHOLD = float(os.environ.get("CORTEX_HOT_THRESHOLD", "0.3"))
WARM_THRESHOLD = float(os.environ.get("CORTEX_WARM_THRESHOLD", "0.15"))

# Context budget: hard cap on total injected characters (~1000 tokens)
# WHY 4000: Claude's context is 200K tokens. 4000 chars = ~1000 tokens = 0.5%.
# Enough for 3 HOT + 2 WARM results with headers. Any more is diminishing returns.
MAX_INJECTION_CHARS = int(os.environ.get("CORTEX_MAX_INJECTION_CHARS", "4000"))
TOTAL_BUDGET_CHARS = MAX_INJECTION_CHARS


def ensure_dirs():
    """Create all required directories if they don't exist."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
