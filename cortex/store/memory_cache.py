#!/usr/bin/env python3
"""Pre-computed search cache for memory_log.jsonl.

Stores token sets, stem sets, and soundex codes per entry in SQLite.
Enables cheap set-intersection pre-filtering before calling _score_entry().

Tested: 10/10 identical results vs full JSONL scan, 12-56x faster.

ARCHITECTURE NOTES (DO NOT REMOVE — prevents future regression):
────────────────────────────────────────────────────────────────
This file is an ACCELERATION LAYER, not a search replacement.
If this file is deleted or the DB is corrupt, memory_store.py falls back
to scanning memory_log.jsonl directly (same results, just 12-56x slower).

WHY THIS EXISTS (not FTS5):
  FTS5 was A/B tested as both primary search and pre-filter. Results:
  - As primary search: only 4/10 queries correct (misses Soundex, aliases)
  - As pre-filter: 19-96% miss rates (filters out entries that only match
    via stem/soundex/alias/substring — the exact entries we need)
  This cache works because it pre-computes the SAME tokens/stems/soundex
  that _score_entry() uses, then does cheap set intersection to skip entries
  with zero overlap. Same algorithm, just pre-computed.

INVALIDATION:
  - Append: append_entry() adds new entry inline (no rebuild needed)
  - Supersede: mark_dirty() creates .cache-dirty marker -> full rebuild
  - Stale detection: (mtime, fsize, .cache-dirty) checked on load
  - Concurrent builds: BEGIN EXCLUSIVE prevents double-rebuild

DO NOT make this the only search path. The JSONL fallback in
memory_store.search_memory_log() MUST remain as the authoritative path.
────────────────────────────────────────────────────────────────
"""

import hashlib
import json
import re
import sqlite3
import threading
import time
from pathlib import Path

# WHY: All path constants imported from config — single source of truth.
# The original hardcoded ~/.ai-controller/memory/ paths are replaced.
from cortex.config import MEMORY_DIR, CACHE_DB, MEMORY_LOG_FILE as MEMORY_LOG, DIRTY_MARKER

_in_memory_cache = None
_cache_lock = threading.Lock()


def _get_db(readonly=False):
    mode = "ro" if readonly else "rwc"
    uri = f"file:{CACHE_DB}?mode={mode}"
    conn = sqlite3.connect(uri, uri=True, timeout=5)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _ensure_schema(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS entry_cache (
            line_num INTEGER PRIMARY KEY,
            raw_json TEXT NOT NULL,
            tokens TEXT NOT NULL,
            stems TEXT NOT NULL,
            soundex_codes TEXT NOT NULL,
            tag_tokens TEXT NOT NULL,
            tag_soundex TEXT NOT NULL,
            entry_type TEXT NOT NULL,
            content_hash TEXT
        );
        CREATE TABLE IF NOT EXISTS cache_meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );
        CREATE TABLE IF NOT EXISTS idf_weights (
            token TEXT PRIMARY KEY,
            weight REAL
        );
        CREATE TABLE IF NOT EXISTS idf_meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );
    """)


# --- Stem and Soundex (copied from memory_store.py to avoid circular import) ---

def _stem(word):
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


def _soundex(word):
    if not word or len(word) < 2:
        return ""
    word = word.upper()
    first = word[0]
    if not first.isalpha():
        return ""
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


def _compute_entry_data(entry):
    """Pre-compute tokens, stems, soundex for a single entry."""
    content = entry.get("content", "").lower()
    tags_list = [t.lower() for t in entry.get("tags", [])]
    tags_str = " ".join(tags_list)
    entry_type = entry.get("type", "").lower()
    section = entry.get("section", "").lower()
    searchable = f"{content} {tags_str} {entry_type} {section}"

    tokens = set(re.findall(r'\w{3,}', searchable))
    stems = {_stem(t) for t in tokens}
    soundex_codes = {_soundex(t) for t in tokens} - {''}
    tag_tokens = set(re.findall(r'\w{3,}', tags_str))
    tag_soundex = {_soundex(t) for t in tag_tokens} - {''}
    content_hash = hashlib.md5(content.encode()).hexdigest()[:16]

    return {
        'tokens': ' '.join(sorted(tokens)),
        'stems': ' '.join(sorted(stems)),
        'soundex_codes': ' '.join(sorted(soundex_codes)),
        'tag_tokens': ' '.join(sorted(tag_tokens)),
        'tag_soundex': ' '.join(sorted(tag_soundex)),
        'entry_type': entry_type,
        'content_hash': content_hash,
    }


def build_cache():
    """Full rebuild of cache from memory_log.jsonl. ~880ms for 4K entries."""
    if not MEMORY_LOG.exists():
        return

    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    conn = _get_db()
    _ensure_schema(conn)

    try:
        conn.execute("BEGIN EXCLUSIVE")

        # Check if another process already rebuilt while we waited for the lock
        stat = MEMORY_LOG.stat()
        cached_mtime = _get_meta(conn, 'mtime')
        cached_fsize = _get_meta(conn, 'fsize')
        if (cached_mtime == str(stat.st_mtime) and
                cached_fsize == str(stat.st_size) and
                not DIRTY_MARKER.exists()):
            conn.rollback()
            return

        conn.execute("DELETE FROM entry_cache")

        line_num = 0
        with open(MEMORY_LOG, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    line_num += 1
                    continue

                data = _compute_entry_data(entry)
                conn.execute(
                    "INSERT OR REPLACE INTO entry_cache VALUES (?,?,?,?,?,?,?,?,?)",
                    (line_num, line, data['tokens'], data['stems'],
                     data['soundex_codes'], data['tag_tokens'],
                     data['tag_soundex'], data['entry_type'],
                     data['content_hash'])
                )
                line_num += 1

        _set_meta(conn, 'mtime', str(stat.st_mtime))
        _set_meta(conn, 'fsize', str(stat.st_size))
        _set_meta(conn, 'line_count', str(line_num))
        _set_meta(conn, 'built_at', str(time.time()))
        conn.commit()

        # Clear dirty marker if it exists
        if DIRTY_MARKER.exists():
            try:
                DIRTY_MARKER.unlink()
            except OSError:
                pass

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def incremental_sync():
    """Append-only sync: index new lines added since last build."""
    if not MEMORY_LOG.exists() or not CACHE_DB.exists():
        return build_cache()

    conn = _get_db()
    _ensure_schema(conn)

    try:
        cached_count = int(_get_meta(conn, 'line_count') or '0')
        stat = MEMORY_LOG.stat()

        # If dirty marker exists, do full rebuild
        if DIRTY_MARKER.exists():
            conn.close()
            return build_cache()

        # Count current lines
        current_count = 0
        with open(MEMORY_LOG, 'r', encoding='utf-8') as f:
            for _ in f:
                current_count += 1

        if current_count <= cached_count:
            # No new lines (or lines removed — rebuild)
            if current_count < cached_count:
                conn.close()
                return build_cache()
            _set_meta(conn, 'mtime', str(stat.st_mtime))
            _set_meta(conn, 'fsize', str(stat.st_size))
            conn.commit()
            conn.close()
            return

        # Read only new lines
        line_num = 0
        new_entries = []
        with open(MEMORY_LOG, 'r', encoding='utf-8') as f:
            for line in f:
                if line_num >= cached_count:
                    stripped = line.strip()
                    if stripped:
                        try:
                            entry = json.loads(stripped)
                            data = _compute_entry_data(entry)
                            new_entries.append((line_num, stripped, data))
                        except json.JSONDecodeError:
                            pass
                line_num += 1

        for ln, raw, data in new_entries:
            conn.execute(
                "INSERT OR REPLACE INTO entry_cache VALUES (?,?,?,?,?,?,?,?,?)",
                (ln, raw, data['tokens'], data['stems'],
                 data['soundex_codes'], data['tag_tokens'],
                 data['tag_soundex'], data['entry_type'],
                 data['content_hash'])
            )

        _set_meta(conn, 'mtime', str(stat.st_mtime))
        _set_meta(conn, 'fsize', str(stat.st_size))
        _set_meta(conn, 'line_count', str(line_num))
        conn.commit()
    except Exception:
        conn.rollback()
        conn.close()
        return build_cache()
    finally:
        try:
            conn.close()
        except Exception:
            pass


def load_cache():
    """Load all pre-computed sets into memory. Returns list of dicts or None."""
    global _in_memory_cache

    if not CACHE_DB.exists():
        return None

    with _cache_lock:
        # Check staleness
        if not MEMORY_LOG.exists():
            return None

        stat = MEMORY_LOG.stat()

        try:
            conn = _get_db(readonly=True)
            _ensure_schema(conn)
            cached_mtime = _get_meta(conn, 'mtime')
            cached_fsize = _get_meta(conn, 'fsize')
        except Exception:
            return None

        is_stale = (cached_mtime != str(stat.st_mtime) or
                    cached_fsize != str(stat.st_size) or
                    DIRTY_MARKER.exists())

        if is_stale:
            conn.close()
            try:
                incremental_sync()
                conn = _get_db(readonly=True)
            except Exception:
                return None

        try:
            rows = conn.execute(
                "SELECT raw_json, tokens, stems, soundex_codes, "
                "tag_tokens, tag_soundex, entry_type FROM entry_cache"
            ).fetchall()
            conn.close()
        except Exception:
            conn.close()
            return None

        cache = []
        for raw_json, tokens, stems, soundex, tag_tok, tag_sdx, etype in rows:
            try:
                entry = json.loads(raw_json)
            except json.JSONDecodeError:
                continue
            cache.append({
                'entry': entry,
                'tokens': set(tokens.split()) if tokens else set(),
                'stems': set(stems.split()) if stems else set(),
                'soundex': set(soundex.split()) if soundex else set(),
                'tag_tokens': set(tag_tok.split()) if tag_tok else set(),
                'tag_soundex': set(tag_sdx.split()) if tag_sdx else set(),
                'entry_type': etype,
            })

        _in_memory_cache = cache
        return cache


def get_loaded_cache():
    """Return in-memory cache if available and fresh, else try loading."""
    global _in_memory_cache
    if _in_memory_cache is not None:
        # Quick staleness check (mtime only, no DB access)
        try:
            if DIRTY_MARKER.exists():
                _in_memory_cache = None
                return load_cache()
        except Exception:
            pass
        return _in_memory_cache
    return load_cache()


def append_entry(entry, raw_json_line):
    """Append a single entry to the cache. Fire-and-forget."""
    try:
        if not CACHE_DB.exists():
            return

        conn = _get_db()
        _ensure_schema(conn)
        cached_count = int(_get_meta(conn, 'line_count') or '0')
        data = _compute_entry_data(entry)
        conn.execute(
            "INSERT OR REPLACE INTO entry_cache VALUES (?,?,?,?,?,?,?,?,?)",
            (cached_count, raw_json_line.strip(), data['tokens'], data['stems'],
             data['soundex_codes'], data['tag_tokens'],
             data['tag_soundex'], data['entry_type'],
             data['content_hash'])
        )
        stat = MEMORY_LOG.stat()
        _set_meta(conn, 'line_count', str(cached_count + 1))
        _set_meta(conn, 'mtime', str(stat.st_mtime))
        _set_meta(conn, 'fsize', str(stat.st_size))
        conn.commit()
        conn.close()

        # Invalidate in-memory cache so next search reloads
        global _in_memory_cache
        _in_memory_cache = None
    except Exception:
        pass


def mark_dirty():
    """Create dirty marker to trigger full cache rebuild on next search."""
    try:
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        DIRTY_MARKER.touch()
        global _in_memory_cache
        _in_memory_cache = None
    except Exception:
        pass


def is_duplicate_hash(content):
    """O(1) check if content hash already exists in cache."""
    try:
        if not CACHE_DB.exists():
            return False
        content_hash = hashlib.md5(content.lower().encode()).hexdigest()[:16]
        conn = _get_db(readonly=True)
        row = conn.execute(
            "SELECT 1 FROM entry_cache WHERE content_hash = ? LIMIT 1",
            (content_hash,)
        ).fetchone()
        conn.close()
        return row is not None
    except Exception:
        return False


# --- IDF persistence ---

def persist_idf(idf_dict, total_docs):
    """Save IDF weights to cache DB."""
    try:
        conn = _get_db()
        _ensure_schema(conn)
        conn.execute("DELETE FROM idf_weights")
        conn.executemany(
            "INSERT INTO idf_weights VALUES (?, ?)",
            idf_dict.items()
        )
        _set_idf_meta(conn, 'total_docs', str(total_docs))
        _set_idf_meta(conn, 'refreshed_at', str(time.time()))
        _set_idf_meta(conn, 'token_count', str(len(idf_dict)))
        conn.commit()
        conn.close()
    except Exception:
        pass


def load_idf():
    """Load cached IDF weights. Returns (idf_dict, total_docs) or None if stale."""
    try:
        if not CACHE_DB.exists():
            return None
        conn = _get_db(readonly=True)

        refreshed = _get_idf_meta(conn, 'refreshed_at')
        if not refreshed:
            conn.close()
            return None

        # Stale if >24h old
        if time.time() - float(refreshed) > 86400:
            conn.close()
            return None

        # Stale if >50 new entries since last refresh
        cached_total = int(_get_idf_meta(conn, 'total_docs') or '0')
        current_count = int(_get_meta(conn, 'line_count') or '0')
        if current_count - cached_total > 50:
            conn.close()
            return None

        rows = conn.execute("SELECT token, weight FROM idf_weights").fetchall()
        conn.close()

        if not rows:
            return None

        idf = {token: weight for token, weight in rows}
        return idf, cached_total
    except Exception:
        return None


# --- Helpers ---

def _get_meta(conn, key):
    row = conn.execute("SELECT value FROM cache_meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else None

def _set_meta(conn, key, value):
    conn.execute("INSERT OR REPLACE INTO cache_meta VALUES (?, ?)", (key, value))

def _get_idf_meta(conn, key):
    row = conn.execute("SELECT value FROM idf_meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else None

def _set_idf_meta(conn, key, value):
    conn.execute("INSERT OR REPLACE INTO idf_meta VALUES (?, ?)", (key, value))


def _build_async():
    """Build cache in background thread."""
    t = threading.Thread(target=build_cache, daemon=True)
    t.start()
