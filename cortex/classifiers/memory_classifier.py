#!/usr/bin/env python3
"""Shared LLM Memory Classifier — subprocess calls LLM to evaluate memory-worthiness.

Three functions for use by different hooks:
- classify_activities(activities): Batch-evaluate tool activities (auto_learning.py, 8s timeout)
- extract_from_transcript(text, activity_summary): Extract learnings from transcript (stop hook, 12s timeout)
- is_correction(prompt): Check if user prompt is a correction (correction_detector.py, 4s timeout)

Uses cortex.llm_adapter.complete() for LLM calls.
Fail-open: always returns [] or None on failure.
Recursion guard: CORTEX_CLASSIFYING=1 prevents infinite loops.
Cache: content-hash disk cache at ~/.cortex/cache/memory-classifier-cache.json (max 50 entries).
"""

import hashlib
import json
import os
import re
from datetime import date
from pathlib import Path

# WHY: Import from centralized config for all path references.
from cortex.config import MEMORY_DIR

_CACHE_DIR = MEMORY_DIR / "cache"
_CACHE_FILE = _CACHE_DIR / "memory-classifier-cache.json"
_MAX_CACHE = 50

# Valid memory types
_VALID_TYPES = {
    "ARCHITECTURAL_DECISION", "WORKING_SOLUTION", "FAILED_APPROACH",
    "CODEBASE_PATTERN", "USER_PREFERENCE", "USER_CORRECTION",
    "CONFIG_INSIGHT", "DEBUGGING_INSIGHT", "error_resolution",
}


def _is_recursion() -> bool:
    """Check recursion guard.

    WHY: Check both env var names for backward compatibility during migration
    from AI_CONTROLLER_CLASSIFYING to CORTEX_CLASSIFYING.
    """
    if os.environ.get("CORTEX_CLASSIFYING") == "1":
        return True
    if os.environ.get("AI_CONTROLLER_CLASSIFYING") == "1":
        return True
    return False


def _content_hash(text: str) -> str:
    """MD5 hash for cache key."""
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()[:12]


def _load_cache() -> dict:
    """Load disk cache."""
    try:
        if _CACHE_FILE.exists():
            data = json.loads(_CACHE_FILE.read_text())
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _save_cache(cache: dict) -> None:
    """Save disk cache, evicting oldest if over limit."""
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # Evict oldest entries if over limit
        if len(cache) > _MAX_CACHE:
            # Keep only the most recent _MAX_CACHE entries
            items = sorted(cache.items(), key=lambda x: x[1].get("_ts", 0))
            cache = dict(items[-_MAX_CACHE:])
        _CACHE_FILE.write_text(json.dumps(cache, indent=2))
    except Exception:
        pass


def _call_llm(prompt: str, timeout: int) -> str | None:
    """Call LLM via shared adapter with recursion guard. Returns raw text or None.

    WHY: Shared adapter centralizes multi-backend fallback (Gemini/Anthropic/subprocess)
    with caching. Replaces direct subprocess.run(['claude', '-p', ...]) which spawned
    independent processes (5-10s each) with no cross-caller caching.
    """
    if _is_recursion():
        return None

    try:
        from cortex.llm_adapter import complete as _gw
        return _gw(prompt, max_tokens=500, timeout=timeout)
    except Exception:
        return None


def _parse_json_response(raw: str) -> list | dict | None:
    """Parse LLM response as JSON, handling markdown code blocks."""
    if not raw:
        return None

    text = raw.strip()

    # Strip markdown code blocks if present
    if "```" in text:
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _validate_learnings(data: list | dict | None) -> list[dict]:
    """Validate and clean LLM-returned learnings."""
    if data is None:
        return []

    items = data if isinstance(data, list) else [data]
    valid = []

    for item in items:
        if not isinstance(item, dict):
            continue

        content = str(item.get("content", "")).strip()
        entry_type = str(item.get("type", "")).strip()

        # Must have meaningful content
        if len(content) < 20:
            continue

        # Validate type, default to WORKING_SOLUTION
        if entry_type not in _VALID_TYPES:
            entry_type = "WORKING_SOLUTION"

        # Clean tags
        tags = item.get("tags", [])
        if not isinstance(tags, list):
            tags = []
        tags = [str(t).strip().lower() for t in tags if isinstance(t, str) and len(str(t).strip()) >= 2]
        tags = tags[:8]  # Cap tags

        valid.append({
            "type": entry_type,
            "content": content[:500],
            "tags": tags,
            "date": date.today().isoformat(),
            "session_id": "",
            "priority": "normal",
        })

    return valid[:5]  # Max 5 learnings per call


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_activities(activities: list[dict], timeout: int = 8) -> list[dict]:
    """Evaluate a sequence of tool activities for memory-worthy insights.

    Called by auto_learning.py when a signal pattern (RETRY_SUCCESS, CONFIG_BREAKTHROUGH)
    is detected. Only the flagged activity subset is passed, not all 20.

    Returns list of {type, content, tags} dicts, or [] on failure.
    """
    if _is_recursion() or not activities:
        return []

    # Build compact activity summary
    summary_lines = []
    for a in activities[-10:]:  # Last 10 max
        cat = a.get("category", "?")
        desc = a.get("desc", "")[:120]
        summary_lines.append(f"- [{cat}] {desc}")
    summary = "\n".join(summary_lines)

    # Check cache
    cache_key = _content_hash(summary)
    cache = _load_cache()
    if cache_key in cache:
        cached = cache[cache_key]
        return cached.get("result", [])

    prompt = f"""Analyze this sequence of tool activities from a coding session.
Extract ONLY genuinely useful insights worth remembering for future sessions.

Activities:
{summary}

If there are memory-worthy insights (debugging breakthroughs, config fixes,
architectural decisions, better approaches discovered), return a JSON array:
[{{"type": "WORKING_SOLUTION|DEBUGGING_INSIGHT|CONFIG_INSIGHT|CODEBASE_PATTERN",
   "content": "Concise description of the insight (what was learned)",
   "tags": ["relevant", "keywords"]}}]

If nothing is memory-worthy, return: []

Return ONLY the JSON array, no explanation."""

    raw = _call_llm(prompt, timeout)
    result = _validate_learnings(_parse_json_response(raw))

    # Cache result
    import time
    cache[cache_key] = {"result": result, "_ts": time.time()}
    _save_cache(cache)

    return result


def extract_from_transcript(text: str, activity_summary: str = "", timeout: int = 12) -> list[dict]:
    """Extract learnings from session transcript text.

    Called by stop_learning_extractor.py at session end.
    Includes both assistant and user messages for full context.
    Optional activity_summary provides tool-level context.

    Returns list of {type, content, tags} dicts, or [] on failure.
    """
    if _is_recursion() or not text:
        return []

    # WHY: No truncation — periodic extraction produces bounded segments (~2000-10000 chars).
    # Both Gemini (1M tokens) and Sonnet (200K tokens) handle full segments easily.
    # Previously truncated to 6000 chars, silently discarding most of the transcript.

    # Check cache
    cache_key = _content_hash(text[:2000])
    cache = _load_cache()
    if cache_key in cache:
        cached = cache[cache_key]
        return cached.get("result", [])

    activity_block = ""
    if activity_summary:
        activity_block = f"""
Tool activity context (files edited, commands run):
{activity_summary[:1500]}
"""

    # WHY: Prompt restructured to prioritize correction chains, silent failures, and
    # assumption violations over generic categories. Old prompt extracted atomic facts that
    # lost the problem->wrong_fix->correct_fix narrative (e.g. FAILED_APPROACH and
    # WORKING_SOLUTION stored as unlinked entries). New prompt produces self-contained
    # entries that prevent future sessions from repeating wrong approaches.
    prompt = f"""Analyze this coding session transcript and extract the most valuable learnings
worth remembering for future sessions.
{activity_block}
Session transcript:
{text}

Extract up to 5 learnings. Prioritize these (highest value first):
- Correction chains: when an approach was tried then abandoned, capture the FULL arc
  in ONE entry: what was tried -> why it failed -> what actually fixed it. These prevent
  repeating mistakes and are the single most valuable type of learning.
- Silent failure patterns: things that APPEARED to work (no error at call site) but
  actually broke downstream. Include WHERE it silently fails and HOW to detect it.
- Assumption violations: when something behaved differently than expected. State the
  assumption, the reality, and the evidence that proved it wrong.
- Architectural decisions (and WHY — what was chosen, what was rejected, what breaks if reverted)
- User corrections or stated preferences
- Working solutions and configuration discoveries

Return a JSON array:
[{{"type": "ARCHITECTURAL_DECISION|WORKING_SOLUTION|FAILED_APPROACH|DEBUGGING_INSIGHT|CONFIG_INSIGHT|USER_PREFERENCE|CODEBASE_PATTERN",
   "content": "Self-contained learning. For corrections: what was tried + why it failed + actual fix, all in one entry. Include the WHY.",
   "tags": ["relevant", "keywords"]}}]

If nothing is memory-worthy, return: []
Return ONLY the JSON array, no explanation."""
    # PURPOSE: Extraction prompt above prioritizes correction chains, silent failures, and
    # assumption violations so memory captures full problem->fix narratives, not fragments.
    raw = _call_llm(prompt, timeout)
    result = _validate_learnings(_parse_json_response(raw))

    import time
    cache[cache_key] = {"result": result, "_ts": time.time()}
    _save_cache(cache)

    return result


def is_correction(prompt: str, timeout: int = 4) -> dict | None:
    """Check if a user prompt is a correction or preference statement.

    Called by correction_detector.py when regex doesn't match but prompt
    has weak correction signals.

    Returns {type, content, full_prompt} or None.
    """
    if _is_recursion() or not prompt or len(prompt) < 50:
        return None

    # Cheap heuristic gate: only call LLM if prompt has weak correction signals
    prompt_lower = prompt.lower()
    weak_signals = ["actually", "but", "wrong", "should be", "rather", "better",
                    "correct", "meant", "not what", "supposed to", "change",
                    "different", "fix", "mistake"]
    if not any(s in prompt_lower for s in weak_signals):
        return None

    # Check cache
    cache_key = _content_hash(prompt[:200])
    cache = _load_cache()
    if cache_key in cache:
        cached = cache[cache_key]
        return cached.get("result")

    prompt_text = prompt[:800]

    llm_prompt = f"""Is this user message correcting the AI's behavior or stating a preference?

Message: "{prompt_text}"

If YES (correction or preference), respond with JSON:
{{"is_correction": true, "type": "USER_CORRECTION|USER_PREFERENCE", "content": "the instruction/preference extracted"}}

If NO (just a regular request/question), respond with:
{{"is_correction": false}}

Return ONLY the JSON, no explanation."""

    raw = _call_llm(llm_prompt, timeout)
    data = _parse_json_response(raw)

    import time

    if not isinstance(data, dict) or not data.get("is_correction"):
        cache[cache_key] = {"result": None, "_ts": time.time()}
        _save_cache(cache)
        return None

    content = str(data.get("content", "")).strip()
    correction_type = str(data.get("type", "USER_CORRECTION")).strip()

    if len(content) < 10:
        cache[cache_key] = {"result": None, "_ts": time.time()}
        _save_cache(cache)
        return None

    if correction_type not in ("USER_CORRECTION", "USER_PREFERENCE"):
        correction_type = "USER_CORRECTION"

    result = {
        "type": correction_type,
        "content": content[:300],
        "full_prompt": prompt[:500],
    }

    cache[cache_key] = {"result": result, "_ts": time.time()}
    _save_cache(cache)

    return result


def main():
    """CLI entry point for testing. Reads a prompt from stdin and classifies it."""
    import sys
    prompt = sys.stdin.read().strip()
    if not prompt:
        print("Usage: echo 'your prompt' | python3 -m cortex.classifiers.memory_classifier")
        return
    result = is_correction(prompt)
    if result:
        print(json.dumps(result, indent=2))
    else:
        print("No correction detected.")


if __name__ == "__main__":
    main()
