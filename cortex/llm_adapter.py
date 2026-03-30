"""Standalone LLM adapter for Cortex query expansion and classification.

Supports three backends in priority order:
1. Gemini Flash API (GEMINI_API_KEY env var, ~1s, free tier)
2. Anthropic API (ANTHROPIC_API_KEY env var)
3. claude -p subprocess fallback (no key needed, ~12s)

WHY: The original system depended on a centralized llm_gateway.py that
managed rate limiting, caching, and model selection. For the open-source
release, this standalone adapter provides the same functionality with zero
external Python dependencies (uses urllib for HTTP).

All callers wrap this in try/except — if no LLM is available, Cortex
still works via keyword + stem + substring matching. LLM enables
query expansion (semantic recall) and memory classification.
"""

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from cortex.config import MEMORY_DIR

# WHY: Disk cache prevents redundant LLM calls for identical prompts.
# Query expansion prompts repeat across sessions (same user query patterns).
# 48h TTL balances freshness with cost savings.
_LLM_CACHE_DIR = MEMORY_DIR / ".llm-cache"
_CACHE_TTL = 48 * 3600  # 48 hours


def complete(prompt: str, max_tokens: int = 50, timeout: int = 4) -> str | None:
    """Complete a prompt using the best available LLM backend.

    Returns response text or None on failure. Never raises.

    WHY: Callers (expand_query, classifiers) expect None on failure and
    fall back to non-LLM paths. Raising would break the fail-open pattern
    that keeps Cortex functional without any API keys.
    """
    # WHY: Recursion guard prevents infinite loops when the LLM subprocess
    # itself triggers memory hooks that call back into the LLM adapter.
    # Check both names for backwards compatibility during migration.
    if os.environ.get("CORTEX_CLASSIFYING") == "1":
        return None
    if os.environ.get("AI_CONTROLLER_CLASSIFYING") == "1":
        return None

    # Check disk cache first
    cache_key = hashlib.md5(prompt.encode()).hexdigest()[:16]
    cached = _read_cache(cache_key)
    if cached is not None:
        return cached

    result = None

    # WHY: Gemini Flash first — fastest (~1s), free tier, no subprocess overhead.
    result = _try_gemini(prompt, max_tokens, timeout)

    # WHY: Anthropic API second — user likely has this key if using Claude Code.
    if result is None:
        result = _try_anthropic(prompt, max_tokens, timeout)

    # WHY: claude subprocess last — always available but slowest (~12s) due to
    # Node.js + Claude CLI cold start overhead.
    if result is None:
        result = _try_claude_subprocess(prompt, max_tokens, timeout + 8)

    if result:
        _write_cache(cache_key, result)

    return result


def _try_gemini(prompt: str, max_tokens: int, timeout: int) -> str | None:
    """Query expansion via Gemini Flash API. ~1s latency, no subprocess.

    WHY: Tries cheapest model first (flash-lite), then smarter (flash).
    Both are free tier. Timeout per model is half the total to allow fallthrough.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None

    import urllib.request

    # WHY: Cheapest first, then smarter. Both free tier.
    # Timeout=2s per model (max 4s total) to fit within caller's timeout window.
    for model in ("gemini-2.5-flash-lite", "gemini-2.0-flash"):
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={api_key}"
        )
        payload = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.1}
        }).encode()

        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=min(timeout, 2)) as resp:
                result = json.loads(resp.read())
                text = result["candidates"][0]["content"]["parts"][0]["text"].strip()
                if text and len(text) < 500:
                    return text
        except Exception:
            continue  # Try next model or fall through

    return None


def _try_anthropic(prompt: str, max_tokens: int, timeout: int) -> str | None:
    """Query expansion via Anthropic Messages API. Uses urllib, no SDK needed.

    WHY: Many Cortex users already have ANTHROPIC_API_KEY for Claude Code.
    This provides a second backend without requiring any pip install.
    Uses haiku for speed and cost — expansion doesn't need a large model.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    import urllib.request

    url = "https://api.anthropic.com/v1/messages"
    payload = json.dumps({
        "model": "claude-haiku-4-20250414",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()

    req = urllib.request.Request(url, data=payload, headers={
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    })

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read())
            text = result.get("content", [{}])[0].get("text", "").strip()
            if text and len(text) < 500:
                return text
    except Exception:
        pass

    return None


def _try_claude_subprocess(prompt: str, max_tokens: int, timeout: int) -> str | None:
    """Query expansion via Claude CLI subprocess. ~12s latency, always available.

    WHY: No API key needed — uses the user's authenticated Claude Code session.
    Slowest option due to Node.js + CLI cold start, but guaranteed to work if
    Claude Code is installed.
    """
    env = os.environ.copy()
    # WHY: Recursion guard — prevents the subprocess from triggering memory
    # hooks that call back into this adapter.
    env["CORTEX_CLASSIFYING"] = "1"
    env["AI_CONTROLLER_CLASSIFYING"] = "1"

    try:
        result = subprocess.run(
            ["claude", "-p", "--model", "haiku", prompt],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(Path.home()), env=env,
        )
        if result.returncode != 0:
            return None
        text = result.stdout.strip()
        if not text or len(text) > 500:
            return None
        return text
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None
    except Exception:
        return None


def _read_cache(cache_key: str) -> str | None:
    """Read a cached LLM response from disk. Returns None if missing or expired.

    WHY: Disk cache prevents redundant LLM calls. Query expansion prompts
    repeat across sessions (users ask similar things). 48h TTL keeps
    responses fresh while saving API calls and latency.
    """
    try:
        cache_file = _LLM_CACHE_DIR / f"{cache_key}.json"
        if not cache_file.exists():
            return None
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        if time.time() - data.get("ts", 0) > _CACHE_TTL:
            # WHY: Expired entries are deleted to prevent unbounded disk growth.
            cache_file.unlink(missing_ok=True)
            return None
        return data.get("text")
    except Exception:
        return None


def _write_cache(cache_key: str, text: str) -> None:
    """Write an LLM response to disk cache. Fire-and-forget.

    WHY: Fire-and-forget — cache write failure should never block the caller.
    The LLM response is already in hand; caching is a best-effort optimization.
    """
    try:
        _LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = _LLM_CACHE_DIR / f"{cache_key}.json"
        cache_file.write_text(
            json.dumps({"text": text, "ts": time.time()}),
            encoding="utf-8"
        )
    except Exception:
        pass
