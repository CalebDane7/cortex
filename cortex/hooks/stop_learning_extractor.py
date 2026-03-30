#!/usr/bin/env python3
"""Stop Hook Learning Extractor — extracts observations from transcript at session end.

Fires on Stop event. Reads un-processed tail of the session transcript
(using offset tracking), uses LLM extraction via memory_classifier's Observer prompt.
Writes observations to observations.jsonl (new format) AND memory_log.jsonl (backward compat).

Performance: Extraction runs in a forked background process so the hook
returns in <100ms. The child process continues the LLM call independently.
"""

import json
import os
import re
import sys
import time
from datetime import date
from pathlib import Path

# Force UTF-8 on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# WHY: Package imports replace sys.path hacking. memory_store is the shared
# read/write engine, memory_classifier provides LLM-based extraction.
from cortex.store import memory_store

try:
    from cortex.classifiers import memory_classifier
except ImportError:
    memory_classifier = None

# WHY: Import paths from centralized config so all Cortex state is co-located.
from cortex.config import MEMORY_DIR, LOG_DIR, MEMORY_LOG_FILE

# WHY: Cross-platform file locking — fcntl on Linux/macOS, msvcrt on Windows.
from cortex.filelock_compat import lock_file, unlock_file

# Timeout for the entire extractor (seconds)
# Hook timeout in settings.json is 45s. Budget: 40s for extraction, 5s buffer.
MAX_RUNTIME = 40
MAX_LEARNINGS = 10

# Log path for background process
_LOG_DIR = LOG_DIR
_LOG_FILE = _LOG_DIR / "stop-learning-bg.log"


def _get_lock_file(session_id: str) -> Path:
    """Per-session lock file — eliminates cross-session contention."""
    return _LOG_DIR / f"stop-learning-{session_id[:12]}.lock"

# Expanded patterns that indicate extractable learnings (~15 patterns)
DECISION_PATTERNS = [
    # Explicit decisions
    (r"(?:decided|chose|chosen|picked|went with|opting for|going with)\s+(.{20,300})", "ARCHITECTURAL_DECISION"),
    (r"(?:the approach is|the design is|architecture:|the plan is)\s+(.{20,300})", "ARCHITECTURAL_DECISION"),
    # Problem-solution pairs
    (r"(?:fix(?:ed)?|resolved|solved|the (?:fix|solution) (?:is|was)|root cause)\s+(.{20,300})", "WORKING_SOLUTION"),
    (r"(?:resolved by|fixed by|the issue was|the problem was)\s+(.{20,300})", "WORKING_SOLUTION"),
    # Failures
    (r"(?:failed|didn't work|broken|bug|error was|doesn't work)\s+(.{20,300})", "FAILED_APPROACH"),
    # Patterns/techniques
    (r"(?:pattern|approach|technique|method|strategy):\s*(.{20,300})", "CODEBASE_PATTERN"),
    (r"(?:the trick is|the key is|important:)\s+(.{20,300})", "CODEBASE_PATTERN"),
    # Learnings/discoveries
    (r"(?:learned|discovered|found out|realized|turns out)\s+(.{20,300})", "WORKING_SOLUTION"),
    # User corrections
    (r"(?:actually,|no,\s+|instead,?|not that,?)\s+(.{20,300})", "USER_PREFERENCE"),
    # Preferences
    (r"(?:always|never|prefer|don't want|I want)\s+(.{20,300})", "USER_PREFERENCE"),
    # Gotchas/caveats
    (r"(?:watch out|gotcha|caveat|careful|beware|warning:)\s+(.{20,300})", "FAILED_APPROACH"),
    # Compaction summaries (Claude's own summaries)
    (r"\[compact\]\s*(.{30,500})", "ARCHITECTURAL_DECISION"),
    (r"(?:Summary of changes|Key changes made|What was done):\s*(.{30,500})", "WORKING_SOLUTION"),
]


def _has_verb_like_word(text: str) -> bool:
    """Check if text contains at least one verb-like word (past/present tense)."""
    return bool(re.search(r'\b\w+(?:ed|ing|es|s|tion|ment)\b', text, re.IGNORECASE))


def _is_duplicate_of_existing(content: str) -> bool:
    """Check if content is a near-duplicate of an existing memory entry."""
    if not memory_store:
        return False
    try:
        results = memory_store.search_memory_log(content[:80], max_results=1)
        if results and results[0].get('score', 0) > 0.6:
            return True
    except Exception:
        pass
    return False


def append_observation(obs: dict) -> bool:
    """Append observation to observations.jsonl with file locking."""
    obs_file = MEMORY_DIR / "observations.jsonl"
    obs_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        line = json.dumps(obs, ensure_ascii=False) + "\n"
        with open(obs_file, "a") as f:
            lock_file(f)
            try:
                f.write(line)
            finally:
                unlock_file(f)
        return True
    except Exception:
        return False


def _load_offset(session_id: str) -> int:
    """Load last extracted transcript offset for this session."""
    offset_file = Path(f"/tmp/obs-{session_id}.json")
    try:
        if offset_file.exists():
            data = json.loads(offset_file.read_text())
            return data.get("last_extracted_offset", 0)
    except Exception:
        pass
    return 0


def _save_offset(session_id: str, offset: int):
    """Save transcript offset after extraction."""
    offset_file = Path(f"/tmp/obs-{session_id}.json")
    try:
        offset_file.write_text(json.dumps({
            "last_extracted_offset": offset,
            "updated_at": time.time(),
        }))
    except Exception:
        pass


def extract_learnings_from_text(text: str, max_learnings: int = MAX_LEARNINGS) -> list[dict]:
    """Extract learnings from transcript text using pattern matching.

    Quality filters applied:
    - Reject content < 50 chars (too short to be useful)
    - Reject content without verb-like words (fragments, not insights)
    - Reject near-duplicates of existing memory entries (score > 0.6)
    """
    learnings = []
    seen_content = set()

    for pattern, learning_type in DECISION_PATTERNS:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            content = match.group(1).strip()
            # Clean up: remove trailing punctuation noise, truncate
            content = re.sub(r'[\n\r]+', ' ', content)
            content = content[:300]

            # Quality filter: reject too-short content
            if len(content) < 50:
                continue

            # Quality filter: must contain verb-like word
            if not _has_verb_like_word(content):
                continue

            # Deduplicate within this extraction
            content_key = content[:50].lower()
            if content_key in seen_content:
                continue
            seen_content.add(content_key)

            # Quality filter: reject near-duplicates of existing entries
            if _is_duplicate_of_existing(content):
                continue

            learnings.append({
                "type": learning_type,
                "content": content,
            })

            if len(learnings) >= max_learnings:
                return learnings

    return learnings


def get_transcript_tail(num_lines: int = 200, session_id: str = "") -> str:
    """Read the last N lines from the session transcript file.

    Args:
        num_lines: Max JSONL lines to scan from the tail of the file.
        session_id: If provided, look for {session_id}.jsonl directly (avoids subagent files).
    """
    transcript_dirs = [
        Path.home() / ".claude" / "projects",
    ]

    latest_file = None

    # Try session-ID-based lookup first (avoids subagent file pollution)
    if session_id:
        for tdir in transcript_dirs:
            if not tdir.exists():
                continue
            for jsonl in tdir.rglob(f"{session_id}.jsonl"):
                if jsonl.is_file():
                    latest_file = jsonl
                    break
            if latest_file:
                break

    # Fallback to mtime-based selection (original behavior)
    if not latest_file:
        latest_mtime = 0
        for tdir in transcript_dirs:
            if not tdir.exists():
                continue
            for jsonl in tdir.rglob("*.jsonl"):
                try:
                    mtime = jsonl.stat().st_mtime
                    if mtime > latest_mtime:
                        latest_mtime = mtime
                        latest_file = jsonl
                except OSError:
                    continue

    if not latest_file:
        return ""

    try:
        file_size = latest_file.stat().st_size
        with open(latest_file, 'r', encoding='utf-8', errors='replace') as f:
            # 500KB window captures ~17 assistant entries vs 4 at 50KB
            read_size = min(file_size, 500_000)
            f.seek(max(0, file_size - read_size))
            raw = f.read()

        lines = raw.strip().split('\n')
        text_parts = []
        for line in lines[-num_lines:]:
            try:
                entry = json.loads(line)
                if not isinstance(entry, dict):
                    continue

                # Handle both old format and new Claude Code JSONL format.
                # Old format: {"role": "assistant", "content": ...}
                # New format: {"type": "assistant", "message": {"role": "assistant", "content": ...}}
                role = entry.get("role", "")
                content = entry.get("content")

                if not role and entry.get("type") in ("assistant", "user"):
                    # New nested format — content is inside entry.message
                    role = entry["type"]
                    msg = entry.get("message", {})
                    content = msg.get("content") if isinstance(msg, dict) else None

                if role not in ("assistant", "user"):
                    continue

                prefix = "[USER] " if role == "user" else ""
                if isinstance(content, str):
                    text_parts.append(f"{prefix}{content}")
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text", "")
                            if text:
                                text_parts.append(f"{prefix}{text}")
            except (json.JSONDecodeError, KeyError):
                continue

        # Format drift detection — warn if file was non-empty but 0 text extracted
        if not text_parts and file_size > 1000:
            print(
                f"WARNING: 0 text entries extracted from {latest_file.name} "
                f"({file_size} bytes, {len(lines)} lines) — possible format change",
                file=sys.stderr,
            )

        # No artificial cap — return all extracted text.
        # Natural bound: 500KB window -> ~10-30KB text -> 2500-7500 tokens (trivial for Claude).
        return "\n".join(text_parts)
    except Exception:
        return ""


def _acquire_lock(lock_file_path: Path) -> bool:
    """Try to acquire a per-session lock file. Returns True if acquired."""
    try:
        lock_file_path.parent.mkdir(parents=True, exist_ok=True)
        # Use O_CREAT | O_EXCL for atomic creation (fails if exists)
        fd = os.open(str(lock_file_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        # Check if the lock is stale (holder process dead or lock too old)
        try:
            lock_age = time.time() - lock_file_path.stat().st_mtime
            if lock_age > MAX_RUNTIME + 10:
                # Stale lock — remove and retry
                lock_file_path.unlink(missing_ok=True)
                return _acquire_lock(lock_file_path)
            # Check if PID is alive
            pid_text = lock_file_path.read_text().strip()
            if pid_text.isdigit():
                try:
                    os.kill(int(pid_text), 0)  # signal 0 = check if alive
                except ProcessLookupError:
                    # Process is dead — stale lock
                    lock_file_path.unlink(missing_ok=True)
                    return _acquire_lock(lock_file_path)
        except Exception:
            pass
        return False
    except Exception:
        return False


def _release_lock(lock_file_path: Path):
    """Release a per-session lock file."""
    try:
        lock_file_path.unlink(missing_ok=True)
    except Exception:
        pass


def _sweep_stale_locks():
    """Remove lock files older than 120s (stale from crashed processes)."""
    try:
        for f in _LOG_DIR.glob("stop-learning-*.lock"):
            try:
                if time.time() - f.stat().st_mtime > 120:
                    f.unlink(missing_ok=True)
            except OSError:
                pass
    except Exception:
        pass


def _extract_plan_reasoning(session_id: str):
    """Extract decision reasoning from plan files to observations."""
    try:
        plan_dir = Path.home() / ".claude" / "plans"
        if not plan_dir.exists():
            return

        today = date.today().isoformat()

        import time as _time
        _cutoff = _time.time() - 86400  # Only plans modified in the last 24 hours
        for plan_file in plan_dir.glob("*.md"):
            if plan_file.stat().st_mtime < _cutoff:
                continue
            plan_text = plan_file.read_text(encoding='utf-8', errors='replace')
            if len(plan_text) < 100:
                continue

            # Extract sections about decisions/synthesis/why
            decision_patterns = [
                r'(?:#{2,3}\s*(?:Synthesis|Why|Decision|Chose|Rationale).*?\n)((?:.*?\n)*?)(?=#{2,3}\s|\Z)',
                r'(?:Chose\s+.+?\s+because\s+.{50,300})',
                r'(?:Taking\s+.+?\s+from\s+.{50,300})',
            ]

            for pattern in decision_patterns:
                matches = re.findall(pattern, plan_text, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    content = match.strip()
                    if len(content) > 50:
                        obs = {
                            "date": today,
                            "project_tags": [],
                            "topic_tags": ["plan-reasoning", plan_file.stem],
                            "content": content[:500],  # Cap length
                            "session_id": session_id,
                            "source": "plan",
                            "priority": "high",
                            "type": "ARCHITECTURAL_DECISION",
                        }
                        append_observation(obs)
    except Exception:
        pass  # Never block session end on extraction failure


def _do_extraction(session_id: str, is_session_end: bool = False):
    """The actual extraction work — runs in background child process.

    Uses offset tracking to only extract the un-processed tail of the transcript.
    Writes observations to observations.jsonl (new format) AND memory_log.jsonl (backward compat).

    Args:
        is_session_end: If True, writes a SESSION_CHECKPOINT summary after storing learnings.
                        Only the Stop hook passes True; precompact passes False (default).
    """
    start_time = time.time()

    try:
        transcript = get_transcript_tail(session_id=session_id)
        if not transcript:
            print(f"[{time.strftime('%H:%M:%S')}] No transcript found, skipping extraction")
            return

        # Offset tracking: only extract un-processed tail
        last_offset = _load_offset(session_id)
        if last_offset > 0 and last_offset < len(transcript):
            transcript = transcript[last_offset:]
        current_offset = last_offset + len(transcript)

        if len(transcript) < 100:
            print(f"[{time.strftime('%H:%M:%S')}] Transcript tail too short ({len(transcript)} chars), skipping")
            _save_offset(session_id, current_offset)
            return

        print(f"[{time.strftime('%H:%M:%S')}] Transcript: {len(transcript)} chars (offset {last_offset})")

        # Load activity context from auto_learning state file
        activity_summary = ""
        try:
            # WHY: Cache dir lives under MEMORY_DIR so all Cortex state is co-located
            state_file = MEMORY_DIR / "cache" / "auto-learning-state.json"
            if state_file.exists():
                state_data = json.loads(state_file.read_text())
                activities = state_data.get("activities", [])
                if activities:
                    lines = []
                    for a in activities[-15:]:
                        cat = a.get("category", "?")
                        desc = a.get("desc", "")[:100]
                        lines.append(f"[{cat}] {desc}")
                    activity_summary = "\n".join(lines)
        except Exception:
            pass

        # LLM extraction via Observer prompt (memory_classifier.extract_from_transcript)
        learnings = []
        if memory_classifier and time.time() - start_time < MAX_RUNTIME - 2:
            llm_learnings = memory_classifier.extract_from_transcript(
                transcript, activity_summary=activity_summary, timeout=38
            )
            if llm_learnings:
                learnings = llm_learnings[:3]  # Observer extracts 1-3 observations

        # WHY: Pattern-based fallback when LLM returns empty (rate limit, timeout, nothing found).
        # extract_learnings_from_text() processes the FULL transcript without truncation,
        # catching insights the LLM missed. Was the primary method before LLM extraction was added.
        if not learnings and time.time() - start_time < MAX_RUNTIME - 2:
            pattern_learnings = extract_learnings_from_text(transcript, max_learnings=3)
            if pattern_learnings:
                learnings = pattern_learnings

        stored_count = 0
        today = date.today().isoformat()

        for learning in learnings:
            if time.time() - start_time > MAX_RUNTIME - 2:
                break

            # Build observation in new format
            obs = {
                "date": today,
                "project_tags": learning.get("project_tags", []),
                "topic_tags": learning.get("topic_tags", []),
                "content": learning["content"],
                "session_id": session_id,
                "source": "stop",
                "priority": "normal",
            }

            # Write to observations.jsonl (new format)
            obs_ok = append_observation(obs)

            # Backward compat: also write to memory_log.jsonl during parallel rollout
            legacy_ok = False
            if memory_store:
                tags = learning.get("tags", []) + ["stop_hook", "llm_extracted"]
                legacy_ok = memory_store.append_learning({
                    "type": learning.get("type", "WORKING_SOLUTION"),
                    "content": learning["content"],
                    "tags": tags,
                    "session_id": session_id,
                })

            if obs_ok or legacy_ok:
                stored_count += 1

        # Save offset after successful extraction
        _save_offset(session_id, current_offset)

        print(
            f"[{time.strftime('%H:%M:%S')}] Stored {stored_count} observation(s) "
            f"in {time.time() - start_time:.1f}s"
        )

        # Extract decision reasoning from plan files
        _extract_plan_reasoning(session_id)

        # SESSION CHECKPOINT: Write a structured summary at session end.
        # WHY: get_recent_session_summary() in memory_awareness.py reads this
        # to bridge sessions. Without it, session bridging requires reconstructing
        # topics from raw entry tags (slower, less informative).
        # WHY is_session_end guard: precompact_learning_extractor.py also calls
        # _do_extraction() during mid-session compaction. Without this guard,
        # checkpoints would be written mid-session (wrong — session isn't over).
        # precompact passes is_session_end=False (the default), so no checkpoint.
        # WHY stored_count > 0: No point writing a checkpoint if nothing was learned.
        # WHY try/except: Checkpoint is nice-to-have. NEVER block session end for it.
        # The fallback tag-scanning in get_recent_session_summary() handles the case
        # where checkpoint writing fails.
        # FORMAT: "Session abc12345: learning1 | learning2 | learning3"
        # - Starts with "Session" (uppercase) so _is_worth_surfacing() passes
        # - session_id[:8] prefix makes _is_duplicate() skip it (unique per session)
        # - Content capped at 500 chars, tags at 5 (plus "session_checkpoint")
        if is_session_end and stored_count > 0 and memory_store:
            try:
                all_tags = []
                content_summaries = []
                for l in learnings:
                    for t in l.get("tags", []):
                        if t.lower() not in {'stop_hook', 'llm_extracted'} and t not in all_tags:
                            all_tags.append(t)
                    content_summaries.append(l.get("content", "")[:100])
                checkpoint_content = f"Session {session_id[:8]}: " + " | ".join(content_summaries[:3])
                checkpoint = {
                    "type": "SESSION_CHECKPOINT",
                    "content": checkpoint_content[:500],
                    "tags": ["session_checkpoint"] + all_tags[:5],
                    "session_id": session_id,
                }
                memory_store.append_learning(checkpoint)
                print(f"[{time.strftime('%H:%M:%S')}] Session checkpoint written")
            except Exception:
                pass  # Never block session end on checkpoint failure

    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] Extraction error: {e}")


def main():
    # Read stdin immediately (must happen before fork — stdin won't be available in child)
    try:
        raw = sys.stdin.read()
        data = json.loads(raw) if raw.strip() else {}
    except Exception:
        data = {}

    session_id = data.get("session_id", f"stop-{int(time.time())}")

    # Return immediately — never block session teardown
    result = {}
    print(json.dumps(result))
    sys.stdout.flush()

    # Nothing to do without memory_store
    if not memory_store:
        return

    # --- Background extraction via fork ---
    # Parent returns immediately (hook done in <100ms).
    # Child continues with LLM extraction in the background.

    try:
        pid = os.fork()
    except (OSError, AttributeError):
        # fork() unavailable (Windows without WSL) or failed — fall back to synchronous
        # This preserves the old behavior: extraction blocks, but hook already printed result
        _do_extraction(session_id, is_session_end=True)
        return

    if pid > 0:
        # Parent process — exit immediately, hook is done
        return

    # --- Child process (background) ---

    # Detach from parent's process group so we survive parent exit
    try:
        os.setsid()
    except Exception:
        pass

    # Redirect stdout/stderr to log file (parent's stdout is closed/gone)
    try:
        _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        log_fd = os.open(str(_LOG_FILE), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        os.dup2(log_fd, 1)  # stdout
        os.dup2(log_fd, 2)  # stderr
        os.close(log_fd)
        # Rebind Python's sys.stdout/stderr to the new fd
        sys.stdout = os.fdopen(1, 'w', buffering=1)
        sys.stderr = os.fdopen(2, 'w', buffering=1)
    except Exception:
        pass

    # Close stdin (no longer connected)
    try:
        os.close(0)
    except Exception:
        pass

    print(f"\n[{time.strftime('%H:%M:%S')}] Background extraction started for session {session_id}")

    # Sweep stale locks from crashed processes before acquiring
    _sweep_stale_locks()

    # Acquire per-session lock (different sessions never contend)
    lock_file_path = _get_lock_file(session_id)
    if not _acquire_lock(lock_file_path):
        print(f"[{time.strftime('%H:%M:%S')}] Lock held for this session, skipping")
        os._exit(0)

    try:
        _do_extraction(session_id, is_session_end=True)
    finally:
        _release_lock(lock_file_path)
        print(f"[{time.strftime('%H:%M:%S')}] Background extraction complete")

    # Use os._exit to avoid running any atexit handlers from the parent
    os._exit(0)


if __name__ == "__main__":
    main()
