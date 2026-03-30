#!/usr/bin/env python3
"""Shared memory utility module — file-based memory system.

All hooks import from here. Provides:
- load_core_memory(): Read MEMORY.md (full, legacy)
- search_core_tagged(): Score core_tagged.jsonl entries against query (IDF-weighted)
- append_learning(): Append JSON line to memory_log.jsonl
- search_memory_log(): Keyword match + recency decay scoring
- get_memory_stats(): Entry count, last date, MEMORY.md line count

Zero external dependencies (stdlib only).

ARCHITECTURE NOTES (DO NOT REMOVE — prevents future regression):
────────────────────────────────────────────────────────────────
1. SEARCH ALGORITHM (_score_entry): Uses a multi-signal scoring pipeline:
   exact token match + stem match + substring match, weighted by IDF,
   with tag/type/priority boosts and recency decay. Alias expansion handles
   named entities (e.g., "k8s" → "kubernetes").
   REMOVED (March 2026): Soundex phonetic matching — at 4400+ entries, the
   ~7000 unique Soundex codes caused cross-domain false positives (e.g.,
   Rust/iced entries matching memory system queries). Aliases + stems +
   substrings cover all legitimate use cases Soundex previously handled.
   A/B tested: FTS5 was tested as a replacement and REJECTED — only 4/10
   queries returned correct results. DO NOT replace _score_entry() with FTS5
   or any other search algorithm without A/B testing first.

2. SEARCH CACHE (memory_cache.py): Pre-computes tokens/stems/soundex per entry
   in SQLite and uses cheap set intersections to skip ~90% of entries before
   calling _score_entry(). This gives 12-56x speedup with IDENTICAL results
   (verified 10/10 queries). The cache is an ACCELERATION LAYER, not a search
   replacement. If the cache is missing/corrupt, search falls back to the
   original JSONL scan (same results, just slower). DO NOT make the cache the
   primary search — the JSONL scan is the authoritative fallback.

3. QUERY EXPANSION (expand_query): Uses Gemini Flash API (~1s) with Haiku
   subprocess fallback (~12s). The Haiku subprocess spawns Node.js + Claude CLI
   which causes 5-8s cold start overhead — that's why Gemini is primary.
   Expansion adds genuine semantic recall (e.g., "delete" -> "remove, erase")
   that stemming cannot replicate. DO NOT remove expansion — it finds entries
   that keyword search alone would miss. The rescore_results() two-pass pattern
   ensures expansion affects RECALL (what candidates are found) but not
   PRECISION (scoring always uses the original query).

4. DEDUP (append_learning): Three-way decision inspired by Mem0:
   - Jaccard > 0.5 -> SKIP (near-duplicate, _is_duplicate)
   - Jaccard 0.3-0.5 AND new is longer -> UPDATE (_supersede_entry)
   - Otherwise -> APPEND (new topic)
   After supersede, memory_cache.mark_dirty() is called to invalidate the
   search cache. This is CRITICAL — without it, the cache serves stale data.

5. IDF WEIGHTS (_compute_idf_stats): Persisted to memory_cache.db. Refreshed
   when >50 new entries or >24h elapsed. IDF does NOT detect supersedes (the
   weight change from a single supersede is <0.01% across 4000+ entries —
   negligible). The 24h/50-entry refresh catches any drift.
────────────────────────────────────────────────────────────────
"""

import json
import math
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

# WHY: All path constants imported from cortex.config — single source of truth.
# The original hardcoded ~/.ai-controller/memory/ paths are replaced.
from cortex.config import (
    MEMORY_DIR, CORE_MEMORY_FILE, MEMORY_LOG_FILE, CORE_TAGGED_FILE,
    ARCHIVE_DIR, ALIASES_FILE, EXPANSION_CACHE_FILE, PIPELINE_LOG,
    DECAY_RATE, ensure_dirs as _ensure_dirs_from_config,
)
# WHY: Cross-platform file locking replaces direct fcntl usage.
# The original used fcntl.flock() which only works on Unix.
from cortex.filelock_compat import lock_file, unlock_file

# Scoring: score = keyword_match_ratio * e^(-DECAY_RATE * age_in_days)
# DECAY_RATE imported from config (default 0.03)


def _load_aliases() -> dict:
    """Load alias map from aliases.json. Returns empty dict on failure."""
    try:
        if ALIASES_FILE.exists():
            return json.loads(ALIASES_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


_ALIASES = _load_aliases()

_EXPANSION_CACHE_FILE = EXPANSION_CACHE_FILE


def _expand_via_gemini(query: str, prompt: str) -> str | None:
    """Query expansion via Gemini Flash API. ~1s latency, no subprocess."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None

    import urllib.request

    # WHY: Tries cheapest model first (flash-lite), then smarter (flash).
    # Timeout=2s per model (max 4s total) to fit within ThreadPoolExecutor's 4s window
    for model in ("gemini-2.5-flash-lite", "gemini-2.0-flash"):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        payload = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": 50, "temperature": 0.1}
        }).encode()

        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=2) as resp:
                result = json.loads(resp.read())
                text = result["candidates"][0]["content"]["parts"][0]["text"].strip()
                if text and len(text) < 500:
                    _pipeline_log("EXPAND_GEMINI_OK", model=model, query=query[:40], expanded=text[:60])
                    return text
        except Exception:
            continue  # Try next model or fall through to Haiku

    return None


def _expand_via_haiku(query: str, prompt: str, timeout: int) -> str | None:
    """Query expansion via Claude Haiku subprocess. ~12s latency, fallback."""
    env = os.environ.copy()
    # WHY: Recursion guard — prevents the subprocess from triggering memory
    # hooks that call back into expand_query. Check both names for backwards compat.
    env["CORTEX_CLASSIFYING"] = "1"
    env["AI_CONTROLLER_CLASSIFYING"] = "1"

    try:
        result = subprocess.run(
            ["claude", "-p", "--model", "haiku", prompt],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(Path.home()), env=env,
        )
        if result.returncode != 0:
            _pipeline_log("EXPAND_HAIKU_FAIL", query=query[:40], rc=str(result.returncode))
            return None
        expanded = result.stdout.strip()
        if not expanded or len(expanded) > 500:
            _pipeline_log("EXPAND_HAIKU_EMPTY", query=query[:40])
            return None
        _pipeline_log("EXPAND_HAIKU_OK", query=query[:40], expanded=expanded[:60])
        return expanded
    except subprocess.TimeoutExpired:
        _pipeline_log("EXPAND_HAIKU_TIMEOUT", query=query[:40], timeout=str(timeout))
        return None
    except Exception as e:
        _pipeline_log("EXPAND_HAIKU_ERROR", query=query[:40], error=str(e)[:60])
        return None


def _cache_expansion(cache: dict, query_key: str, full_query: str):
    """Cache expansion result atomically. Max 100 entries."""
    try:
        cache[query_key] = full_query
        if len(cache) > 100:
            cache = dict(list(cache.items())[-100:])
        import tempfile
        tmp_fd, tmp_path = tempfile.mkstemp(dir=str(MEMORY_DIR), suffix=".tmp")
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(cache, f)
        os.replace(tmp_path, str(_EXPANSION_CACHE_FILE))
    except Exception:
        pass


def expand_query(query: str, timeout: int = 10) -> str | None:
    """Expand query with semantically related terms. LLM adapter primary, direct fallbacks.

    Returns expanded query string or None on failure. Fail-open.
    """
    # WHY: Recursion guard — check both names for backwards compatibility
    # during migration from ai-controller to cortex.
    if os.environ.get("CORTEX_CLASSIFYING") == "1":
        return None
    if os.environ.get("AI_CONTROLLER_CLASSIFYING") == "1":
        return None
    if os.environ.get("MEMORY_EXPANSION_DISABLED") == "1":
        return None

    query_key = query.lower().strip()

    # Check cache first
    cache = {}
    try:
        if _EXPANSION_CACHE_FILE.exists():
            cache = json.loads(_EXPANSION_CACHE_FILE.read_text())
            if query_key in cache:
                return cache[query_key]
    except Exception:
        pass

    prompt = (
        f'Query: "{query}"\n'
        'Generate 3-5 direct synonyms or alternate spellings for the key concepts. '
        'Focus on what the user literally means, not tangentially related topics. '
        'Return ONLY comma-separated keywords, nothing else.'
    )

    # WHY: Try the centralized llm_adapter first — it handles Gemini, Anthropic API,
    # and claude subprocess with caching. Falls back to direct Gemini/Haiku calls
    # if the adapter import fails (defensive — shouldn't happen in normal use).
    expanded = None
    try:
        from cortex.llm_adapter import complete as _gw
        expanded = _gw(prompt, max_tokens=50, timeout=4)
    except Exception:
        pass

    # WHY: Direct fallbacks if llm_adapter is unavailable. These are the same
    # functions the original code used before the gateway was introduced.
    if not expanded:
        expanded = _expand_via_gemini(query, prompt)
    if not expanded:
        expanded = _expand_via_haiku(query, prompt, timeout)

    if not expanded:
        return None

    full_query = f"{query} {expanded}"
    _cache_expansion(cache, query_key, full_query)
    return full_query


def _compute_idf_stats() -> tuple[dict, int]:
    """Compute IDF weights from all entries (memory_log + core_tagged).

    Returns (idf_dict, total_entries).
    idf_dict maps token -> log(1 + N/df) where df = entries containing that token.
    Uses smoothed IDF (log(1 + N/df)) to ensure tokens appearing in all entries
    still have positive weight (~0.69) rather than zero.
    """
    # Try cached IDF first
    try:
        from cortex.store import memory_cache
        cached = memory_cache.load_idf()
        if cached:
            return cached
    except Exception:
        pass

    doc_freq = {}  # token -> count of entries containing it
    total = 0

    for filepath in (MEMORY_LOG_FILE, CORE_TAGGED_FILE):
        if not filepath.exists():
            continue
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    total += 1
                    content = entry.get('content', '').lower()
                    tags_str = ' '.join(t.lower() for t in entry.get('tags', []))
                    section = entry.get('section', '').lower()
                    entry_type = entry.get('type', '').lower()
                    searchable = f"{content} {tags_str} {entry_type} {section}"
                    unique_tokens = set(re.findall(r'\w{3,}', searchable))
                    for token in unique_tokens:
                        doc_freq[token] = doc_freq.get(token, 0) + 1
        except Exception:
            continue

    if total == 0:
        return {}, 0

    idf = {}
    for token, df in doc_freq.items():
        idf[token] = math.log(1 + total / df)  # smoothed IDF, always > 0

    # Persist for next time
    try:
        from cortex.store import memory_cache
        memory_cache.persist_idf(idf, total)
    except Exception:
        pass

    return idf, total


_IDF_CACHE = None


def _get_idf() -> tuple[dict, int]:
    """Get IDF weights, computing lazily and caching at module level."""
    global _IDF_CACHE
    if _IDF_CACHE is None:
        _IDF_CACHE = _compute_idf_stats()
    return _IDF_CACHE


# WHY: Pipeline log path imported from config — single source of truth.
# Shared log file with memory_awareness_tiered.py (or equivalent Cortex hook).


def _pipeline_log(event: str, **kw):
    """Append one line to memory-pipeline.log. Never raises."""
    try:
        PIPELINE_LOG.parent.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%H:%M:%S")
        detail = " ".join(f"{k}={v}" for k, v in kw.items()) if kw else ""
        with open(PIPELINE_LOG, "a") as f:
            f.write(f"[{ts}] STORE:{event} {detail}\n")
    except Exception:
        pass


def _ensure_dirs():
    """Create memory directories if they don't exist."""
    # WHY: Delegates to config.ensure_dirs() which knows all required directories.
    _ensure_dirs_from_config()


def load_core_memory() -> str:
    """Read MEMORY.md and return its content as a string."""
    try:
        if CORE_MEMORY_FILE.exists():
            return CORE_MEMORY_FILE.read_text(encoding="utf-8")
    except Exception:
        pass
    return ""


def _jaccard_similarity(a: str, b: str) -> float:
    """Token-set Jaccard similarity with punctuation stripping."""
    tokens_a = set(re.findall(r'\b\w+\b', a.lower()))
    tokens_b = set(re.findall(r'\b\w+\b', b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _read_tail_lines(max_bytes: int = 8192) -> list[str]:
    """Read last N bytes of memory_log.jsonl, return non-empty lines."""
    try:
        if not MEMORY_LOG_FILE.exists():
            return []
        file_size = MEMORY_LOG_FILE.stat().st_size
        read_start = max(0, file_size - max_bytes)
        with open(MEMORY_LOG_FILE, "r", encoding="utf-8") as f:
            if read_start > 0:
                f.seek(read_start)
                f.readline()  # Skip partial line
            return [ln for ln in f.readlines() if ln.strip()]
    except Exception:
        return []


def _is_duplicate(content: str) -> bool:
    """Check if content is a near-duplicate of recent entries.

    Tries O(1) content hash check first, falls back to Jaccard scan.
    Returns True if prefix matches or Jaccard similarity > 0.5.
    Fail-open: returns False on any error (never lose data).
    """
    # Fast path: O(1) hash check against cache
    try:
        from cortex.store import memory_cache
        if memory_cache.is_duplicate_hash(content):
            return True
    except Exception:
        pass

    try:
        lines = _read_tail_lines()
        prefix = content[:80].lower()
        recent_entries = []
        for line in reversed(lines):
            try:
                entry = json.loads(line.strip())
                recent_entries.append(entry.get("content", ""))
            except json.JSONDecodeError:
                continue
            if len(recent_entries) >= 50:
                break
        for existing in recent_entries:
            if existing[:80].lower() == prefix:
                return True
            if _jaccard_similarity(content, existing) > 0.5:
                return True
        return False
    except Exception:
        return False


def _find_supersedable(content: str, tail_lines: list[str]) -> int | None:
    """Find an older entry that should be REPLACED by this newer, longer version.

    Inspired by Mem0 ADD/UPDATE/DELETE/NOOP — using Jaccard instead of LLM:
    - Jaccard > 0.5: handled by _is_duplicate (SKIP)
    - Jaccard 0.3-0.5 AND new is longer: UPDATE (return line index)
    - Jaccard 0.3-0.5 AND new is shorter: SKIP (old is better)
    - Jaccard < 0.3: APPEND (different topic)

    Returns: index into tail_lines to replace, or None.
    """
    new_tokens = set(re.findall(r'\b\w+\b', content.lower()))
    if not new_tokens:
        return None

    for i, line in enumerate(reversed(tail_lines)):
        try:
            entry = json.loads(line.strip())
            old_content = entry.get("content", "")
            old_tokens = set(re.findall(r'\b\w+\b', old_content.lower()))
            if not old_tokens:
                continue
            union = new_tokens | old_tokens
            jaccard = len(new_tokens & old_tokens) / len(union)
            if 0.3 <= jaccard <= 0.5 and len(content) >= len(old_content):
                # Found supersedable: new is similar but longer (more info)
                return len(tail_lines) - 1 - i
        except json.JSONDecodeError:
            continue
        if i >= 50:
            break
    return None


def _supersede_entry(line_idx: int, tail_lines: list[str], new_entry: dict) -> bool:
    """Replace an existing entry in memory_log.jsonl with a newer version.

    Reads full file, replaces the target line, writes back under lock.
    """
    try:
        with open(MEMORY_LOG_FILE, "r", encoding="utf-8") as f:
            all_lines = f.readlines()

        # Find the actual line in the full file that matches tail_lines[line_idx]
        target_line = tail_lines[line_idx].strip()
        replaced = False
        new_line = json.dumps(new_entry, ensure_ascii=False) + "\n"

        # Search from end (tail_lines are from the end of file)
        for j in range(len(all_lines) - 1, -1, -1):
            if all_lines[j].strip() == target_line:
                all_lines[j] = new_line
                replaced = True
                break

        if not replaced:
            return False

        # WHY: Cross-platform file locking via filelock_compat replaces
        # direct fcntl.flock() calls for Windows compatibility.
        with open(MEMORY_LOG_FILE, "w", encoding="utf-8") as f:
            lock_file(f)
            try:
                f.writelines(all_lines)
            finally:
                unlock_file(f)
        return True
    except Exception:
        return False


def append_learning(entry: dict) -> bool:
    """Append a learning entry as a JSON line to memory_log.jsonl.

    Entry should have: type, content, tags (list).
    Timestamp and session_id are added automatically if missing.

    Three-way decision (inspired by Mem0 ADD/UPDATE/NOOP):
    - Jaccard > 0.5 with existing → SKIP (near-duplicate)
    - Jaccard 0.3-0.5 AND new is longer → UPDATE (supersede old)
    - Otherwise → APPEND (new topic)

    Uses file locking for concurrent access safety.
    """
    _ensure_dirs()

    if "timestamp" not in entry:
        entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    if "session_id" not in entry:
        entry["session_id"] = "unknown"

    content = entry.get("content", "")

    # Write-time dedup: skip near-duplicates (Jaccard > 0.5)
    if content and _is_duplicate(content):
        _pipeline_log("DEDUP_SKIP", content=content[:80])
        return False

    # Check for supersedable entries (Jaccard 0.3-0.5, new is longer)
    if content:
        tail_lines = _read_tail_lines()
        supersede_idx = _find_supersedable(content, tail_lines)
        if supersede_idx is not None:
            if _supersede_entry(supersede_idx, tail_lines, entry):
                _pipeline_log("SUPERSEDE", type=entry.get("type", "?"), content=content[:80])
                # Mark cache dirty — in-place replacement invalidates line-based cache
                try:
                    from cortex.store import memory_cache
                    memory_cache.mark_dirty()
                except Exception:
                    pass
                return True
            # Fall through to append if supersede failed

    try:
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        # WHY: Cross-platform file locking via filelock_compat replaces
        # direct fcntl.flock() calls for Windows compatibility.
        with open(MEMORY_LOG_FILE, "a", encoding="utf-8") as f:
            lock_file(f)
            try:
                f.write(line)
            finally:
                unlock_file(f)
        _pipeline_log("APPEND", type=entry.get("type", "?"), content=content[:80])
        # Append to cache (fire-and-forget)
        try:
            from cortex.store import memory_cache
            memory_cache.append_entry(entry, line)
        except Exception:
            pass
        return True
    except Exception:
        return False


def _stem(word: str) -> str:
    """Basic suffix stripping for better matching. No dependencies."""
    if len(word) <= 4:
        return word
    for suffix in ('tion', 'sion', 'ment', 'ness', 'able', 'ible', 'ting',
                    'ling', 'ning', 'ring', 'sing', 'ying', 'ied', 'ies',
                    'ing', 'ted', 'sed', 'red', 'ned', 'led', 'ked',
                    'ers', 'ors', 'ess', 'ful', 'ous', 'ive', 'ize',
                    'ed', 'ly', 'er', 'or', 'es', 'al'):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    if word.endswith('s') and not word.endswith('ss') and len(word) > 4:
        return word[:-1]
    return word


def _soundex(word: str) -> str:
    """American Soundex encoding for phonetic matching. No dependencies.

    Maps words to a 4-char code based on how they sound.
    e.g., "k8s" -> "K200", "kubernetes" -> "K163"
    """
    if not word or len(word) < 2:
        return ""
    word = word.upper()
    # Keep first letter
    first = word[0]
    if not first.isalpha():
        return ""
    # Soundex coding table
    codes = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6',
    }
    result = [first]
    prev_code = codes.get(first, '0')
    for ch in word[1:]:
        code = codes.get(ch, '0')
        if code != '0' and code != prev_code:
            result.append(code)
            if len(result) == 4:
                break
        prev_code = code if code != '0' else prev_code
    return ''.join(result).ljust(4, '0')


def _substring_match(query_tokens: set, entry_tokens: set, idf: dict | None = None) -> float:
    """Count partial matches where one token is a substring of another.

    When idf is provided, each match is weighted by the query token's IDF
    (rarer tokens contribute more). Without idf, uniform 0.5 per match.
    """
    partial = 0.0
    for qt in query_tokens:
        for et in entry_tokens:
            if qt == et:
                break  # Already counted as exact match
            if len(qt) >= 3 and len(et) >= 3:
                shorter, longer = (qt, et) if len(qt) <= len(et) else (et, qt)
                if shorter in longer and len(shorter) / len(longer) >= 0.5:
                    weight = idf.get(qt, 1.0) if idf else 1.0
                    partial += 0.5 * weight
                    break
    return partial


# High-priority types get a score boost
_PRIORITY_TYPES = {"user_correction", "repeated_correction", "user_preference",
                   "frustrated_correction", "universal_directive"}


def _is_worth_surfacing(content: str) -> bool:
    """Structural quality gate for search results — filter garbage before scoring.

    Same structural signals as correction_detector._is_worth_storing():
    - Minimum 5 words, 3 unique meaningful words
    - Uppercase start (complete sentence, not mid-sentence fragment)
    """
    if not content or len(content) < 20:
        return False
    words = content.split()
    if len(words) < 5:
        return False
    meaningful = {w.lower() for w in words if len(w) > 3}
    if len(meaningful) < 3:
        return False
    if content[0].islower() and not content.startswith(("npm", "git", "pip", "apt", "ssh", "curl", "docker")):
        return False
    return True


def _score_entry(entry: dict, query_tokens: set, query_stems: set, now: float,
                 use_decay: bool = True, aliases: dict | None = None,
                 idf: dict | None = None) -> float:
    """Score a single entry against query tokens. Shared by search_memory_log and search_core_tagged.

    Scoring:
    - Base: IDF-weighted match ratio when idf provided, else uniform
    - Tag boost: 2x multiplier if query tokens overlap with entry tags
    - Type boost: 1.5x for priority types (corrections, preferences, universals)
    - Recency (if use_decay): * e^(-0.03 * age_in_days)
    - Alias expansion: if aliases provided, query tokens are expanded for matching
      but the denominator stays as original query_tokens count (no score dilution)
    - IDF weighting: when idf dict provided, rare tokens contribute more to score
    """
    # Structural quality gate — garbage entries never surface
    raw_content = entry.get("content", "")
    if not _is_worth_surfacing(raw_content):
        return 0.0

    content = raw_content.lower()
    tags_list = [t.lower() for t in entry.get("tags", [])]
    tags_str = " ".join(tags_list)
    entry_type = entry.get("type", "").lower()
    # For core_tagged entries, also search section name
    section = entry.get("section", "").lower()
    searchable = f"{content} {tags_str} {entry_type} {section}"

    entry_tokens = set(re.findall(r'\w{3,}', searchable))
    if not entry_tokens:
        return 0.0

    # 1. Exact matches (including alias expansion)
    exact_matches = query_tokens & entry_tokens
    # Expand via aliases: if "k8s" is in query and "kubernetes" is in entry,
    # count "k8s" as matched (denominator stays len(query_tokens))
    if aliases:
        for qt in query_tokens:
            if qt not in exact_matches:
                for alias in aliases.get(qt, []):
                    if alias in entry_tokens:
                        exact_matches = exact_matches | {qt}
                        break
    # IDF-weighted denominator (total query importance) — rare tokens raise the bar
    total_w = sum(idf.get(t, 1.0) for t in query_tokens) if idf else len(query_tokens)
    if total_w <= 0:
        return 0.0

    if not exact_matches:
        # 2. Try stem matching
        entry_stems = {_stem(t) for t in entry_tokens}
        stem_matches = query_stems & entry_stems
        if not stem_matches:
            # 3. Substring matching (already IDF-weighted when idf provided)
            sub_score = _substring_match(query_tokens, entry_tokens, idf)
            if sub_score == 0:
                # WHY: Soundex phonetic matching removed — too coarse at 4400+ entries.
                # Only ~7000 unique Soundex codes, causing cross-domain false positives
                # (e.g., Rust/iced entries matching memory system queries).
                # Aliases handle named entities, stems handle morphology, substrings
                # handle partial matches — Soundex is redundant.
                return 0.0
            else:
                match_score = sub_score / total_w
        else:
            if idf:
                stem_w = sum(idf.get(t, 1.0) for t in stem_matches) * 0.7
                match_score = stem_w / total_w
            else:
                match_score = len(stem_matches) * 0.7 / total_w
    else:
        entry_stems = {_stem(t) for t in entry_tokens}
        stem_only = (query_stems & entry_stems) - {_stem(t) for t in exact_matches}
        sub_score = _substring_match(query_tokens - exact_matches, entry_tokens - exact_matches, idf)
        if idf:
            exact_w = sum(idf.get(t, 1.0) for t in exact_matches)
            stem_w = sum(idf.get(t, 1.0) for t in stem_only) * 0.7
            match_score = (exact_w + stem_w + sub_score * 0.5) / total_w
        else:
            match_score = (len(exact_matches) + len(stem_only) * 0.7 + sub_score * 0.5) / total_w

    # Multiplier eligibility: weak base matches shouldn't be promoted to HOT
    # - base >= 0.3: always eligible (strong match on its own)
    # - base 0.15-0.3 with exact matches: eligible (genuine token overlap)
    # - base 0.15-0.3 without exact matches: NOT eligible (stem/phonetic only = too weak)
    # - base < 0.15: never eligible
    multiplier_eligible = (match_score >= 0.3 or (match_score >= 0.15 and len(exact_matches) > 0)) and len(query_tokens) >= 3

    # Tag boost (exact only)
    tag_tokens = set(re.findall(r'\w{3,}', tags_str))
    tag_overlap = query_tokens & tag_tokens
    # WHY: Only exact tag overlap gets boost. Phonetic tag matching removed (same
    # Soundex false positive issue as content scoring — aliases handle named entities).
    tag_boost = (2.0 if tag_overlap else 1.0) if multiplier_eligible else 1.0

    # Type boost
    type_boost = (1.5 if entry_type in _PRIORITY_TYPES else 1.0) if multiplier_eligible else 1.0

    # Priority boost for core_tagged entries
    priority = entry.get("priority", 0)
    priority_boost = 1.0 + (priority * 0.1) if priority else 1.0

    # Age decay (optional — core_tagged entries are timeless distilled knowledge)
    decay = 1.0
    if use_decay:
        try:
            ts = entry.get("timestamp", "")
            if isinstance(ts, str) and ts:
                entry_time = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
            else:
                entry_time = now
        except (ValueError, TypeError):
            entry_time = now
        age_days = max(0, (now - entry_time) / 86400)
        decay = math.exp(-DECAY_RATE * age_days)

    # Bidirectional relevance: penalize superficial cross-topic matches
    if exact_matches and len(entry_tokens) > 3:
        entry_coverage = len(exact_matches & entry_tokens) / math.sqrt(len(entry_tokens))
        coverage_factor = max(0.15, min(1.0, entry_coverage * 3))
    else:
        coverage_factor = 1.0

    return match_score * tag_boost * type_boost * priority_boost * decay * coverage_factor


def _search_with_cache(query_tokens, query_stems, idf, cache, max_results):
    """Search using pre-computed cache. Same results as JSONL scan, much faster."""
    now = time.time()
    # Expand query tokens with aliases for pre-filter
    expanded_tokens = set(query_tokens)
    if _ALIASES:
        for qt in query_tokens:
            expanded_tokens.update(_ALIASES.get(qt, []))

    results = []
    for cached in cache:
        # WHY: Soundex removed from scoring — soundex-only matches score 0.0.
        has_overlap = (
            (expanded_tokens & cached['tokens']) or
            (query_stems & cached['stems'])
        )
        if not has_overlap:
            continue

        # Full scoring on candidates that pass pre-filter
        score = _score_entry(cached['entry'], query_tokens, query_stems, now,
                             use_decay=True, aliases=_ALIASES, idf=idf)
        if score > 0:
            results.append({"entry": cached['entry'], "score": score})

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:max_results]


def search_memory_log(query: str, max_results: int = 5) -> list[dict]:
    """Search memory_log.jsonl using keyword match + stemming + tag boost + recency decay.

    Returns list of dicts with 'entry' and 'score' keys, sorted by score desc.
    """
    if not query or not MEMORY_LOG_FILE.exists():
        return []

    query_lower = query.lower()
    query_tokens = set(re.findall(r'\w{3,}', query_lower))
    if not query_tokens:
        return []

    idf, _ = _get_idf()

    # IDF-based token selection for verbose queries
    # Keep the 10 most informative tokens (highest IDF = rarest = most specific)
    # Unknown tokens get max IDF (novel terms are likely project-specific and important)
    if len(query_tokens) > 10 and idf:
        max_idf = max(idf.values())
        ranked = sorted(query_tokens, key=lambda t: -idf.get(t, max_idf))
        query_tokens = set(ranked[:10])

    query_stems = {_stem(t) for t in query_tokens}

    # Try cached path first (12-56x faster, identical results)
    try:
        from cortex.store import memory_cache
        cache = memory_cache.get_loaded_cache()
        if cache:
            return _search_with_cache(query_tokens, query_stems, idf, cache, max_results)
    except Exception:
        pass  # Fall through to original JSONL scan

    # Original JSONL scan (fallback when cache unavailable)
    now = time.time()
    results = []

    try:
        with open(MEMORY_LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                score = _score_entry(entry, query_tokens, query_stems, now, use_decay=True, aliases=_ALIASES, idf=idf)
                if score > 0:
                    results.append({"entry": entry, "score": score})

    except Exception:
        return []

    # Trigger async cache build for next search
    try:
        from cortex.store import memory_cache
        memory_cache._build_async()
    except Exception:
        pass

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:max_results]


def search_core_tagged(query: str, max_results: int = 5) -> list[dict]:
    """Search core_tagged.jsonl using same scoring as memory_log but without recency decay.

    Core tagged entries are distilled knowledge — they don't age out.
    Returns list of dicts with 'entry' and 'score' keys, sorted by score desc.
    """
    if not query or not CORE_TAGGED_FILE.exists():
        return []

    query_lower = query.lower()
    query_tokens = set(re.findall(r'\w{3,}', query_lower))
    if not query_tokens:
        return []

    idf, _ = _get_idf()

    # IDF-based token selection for verbose queries (same as search_memory_log)
    if len(query_tokens) > 10 and idf:
        max_idf = max(idf.values())
        ranked = sorted(query_tokens, key=lambda t: -idf.get(t, max_idf))
        query_tokens = set(ranked[:10])

    query_stems = {_stem(t) for t in query_tokens}
    now = time.time()
    results = []

    try:
        with open(CORE_TAGGED_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                score = _score_entry(entry, query_tokens, query_stems, now, use_decay=False, aliases=_ALIASES, idf=idf)
                if score > 0:
                    results.append({"entry": entry, "score": score})

    except Exception:
        return []

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:max_results]


def rescore_results(results: list[dict], original_query: str) -> list[dict]:
    """Re-score results against original query, not expanded.

    Expansion provides RECALL (finding entries with different wording).
    Scoring is anchored to what the user actually asked.
    """
    if not results or not original_query:
        return results
    query_lower = original_query.lower()
    query_tokens = set(re.findall(r'\w{3,}', query_lower))
    if not query_tokens:
        return results
    query_stems = {_stem(t) for t in query_tokens}
    now = time.time()
    idf, _ = _get_idf()
    rescored = []
    for r in results:
        entry = r.get('entry', {})
        new_score = _score_entry(entry, query_tokens, query_stems, now,
                                 use_decay=True, aliases=_ALIASES, idf=idf)
        if new_score > 0:
            rescored.append({'entry': entry, 'score': new_score})
    return rescored


def get_memory_stats() -> dict:
    """Return stats about the memory system.

    Returns dict with:
    - log_entries: number of entries in memory_log.jsonl
    - last_entry_date: ISO date of most recent entry
    - core_memory_lines: line count of MEMORY.md
    """
    stats = {
        "log_entries": 0,
        "last_entry_date": None,
        "core_memory_lines": 0,
    }

    try:
        if CORE_MEMORY_FILE.exists():
            stats["core_memory_lines"] = len(CORE_MEMORY_FILE.read_text(encoding="utf-8").splitlines())
    except Exception:
        pass

    try:
        if MEMORY_LOG_FILE.exists():
            last_ts = None
            count = 0
            with open(MEMORY_LOG_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    count += 1
                    try:
                        entry = json.loads(line)
                        ts = entry.get("timestamp")
                        if ts:
                            last_ts = ts
                    except json.JSONDecodeError:
                        pass
            stats["log_entries"] = count
            stats["last_entry_date"] = last_ts
    except Exception:
        pass

    return stats
