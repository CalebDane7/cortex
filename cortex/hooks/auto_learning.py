#!/usr/bin/env python3
"""Auto-Learning PostToolUse Hook — captures error-resolution pairs, test results, and signal-based insights.

Fires on every PostToolUse. Tracks:
- Error->fix sequences (high-value error-resolution pairs)
- Test pass/fail with edit context
- Signal-based LLM extraction for RETRY_SUCCESS and CONFIG_BREAKTHROUGH patterns

Writes to file-based memory via memory_store.append_learning().
"""

import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Force UTF-8 on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# WHY: Import from centralized config instead of computing paths from __file__.
# All Cortex state lives under MEMORY_DIR.
from cortex.config import MEMORY_DIR, LOG_DIR

_CACHE_DIR = MEMORY_DIR / "cache"

# WHY: Package imports replace sys.path hacking. These are the shared memory
# read/write engine and the LLM classifier for signal-based extraction.
from cortex.store import memory_store
try:
    from cortex.classifiers import memory_classifier
except ImportError:
    memory_classifier = None

STATE_FILE = _CACHE_DIR / "auto-learning-state.json"
DEBUG_LOG = LOG_DIR / "auto-learning-debug.log"
DEBUG = False

# Max activities to keep in state (for edit tracking only now)
MAX_ACTIVITIES = 20


def debug_log(msg: str) -> None:
    if not DEBUG:
        return
    try:
        DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(DEBUG_LOG, "a") as f:
            f.write(f"[{datetime.now().isoformat()}] {msg}\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Intent bridge: read current user intent from shared state file
# ---------------------------------------------------------------------------

def _load_intent(session_id: str) -> dict:
    """Load current user intent written by memory_awareness.py."""
    try:
        intent_file = _CACHE_DIR / f"current-intent-{session_id[:8]}.json"
        if intent_file.exists():
            data = json.loads(intent_file.read_text())
            if time.time() - data.get("timestamp", 0) < 1800:
                return data
    except Exception:
        pass
    return {}


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def load_state(session_id: str = "") -> dict:
    try:
        if STATE_FILE.exists():
            data = json.loads(STATE_FILE.read_text())
            state = {
                "activities": data.get("activities", []),
                "edits": data.get("edits", []),
                "turn_count": data.get("turn_count", 0),
                "error_sequences": data.get("error_sequences", []),
                "current_intent": data.get("current_intent", {}),
                "signal_processed": data.get("signal_processed", False),
            }
            if session_id:
                intent = _load_intent(session_id)
                if intent:
                    state["current_intent"] = intent
            return state
    except Exception:
        pass
    return {
        "activities": [], "edits": [], "turn_count": 0,
        "error_sequences": [], "current_intent": {},
        "signal_processed": False,
    }


def save_state(state: dict) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        state["activities"] = state["activities"][-MAX_ACTIVITIES:]
        state["edits"] = state["edits"][-10:]
        state["error_sequences"] = state.get("error_sequences", [])[-5:]
        STATE_FILE.write_text(json.dumps(state))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_file_name(path_str: str) -> str:
    """Get just the filename from a path string."""
    if "/" in path_str:
        return path_str.rsplit("/", 1)[-1]
    if "\\" in path_str:
        return path_str.rsplit("\\", 1)[-1]
    return path_str


def _extract_error_line(output: str) -> str:
    """Extract the most informative error line from command output."""
    if not output:
        return "no output"
    lines = output.strip().split("\n")
    for line in lines:
        line_stripped = line.strip()
        if re.search(r'(Error|error|ERROR|FAIL|Failed|Exception|Traceback|fatal|FATAL)', line_stripped):
            return line_stripped[:120]
    for line in reversed(lines):
        if line.strip():
            return line.strip()[:120]
    return output[:120]


# ---------------------------------------------------------------------------
# Activity tracking (for edit context in test results + error sequences)
# ---------------------------------------------------------------------------

def _track_activity(state: dict, tool_name: str, tool_input: dict, tool_response: dict) -> dict | None:
    """Track a tool call as a semantic activity. Returns insight dict or None."""
    try:
        if tool_name in ("Edit", "Write", "MultiEdit"):
            file_path = str(tool_input.get("file_path", ""))
            file_name = _extract_file_name(file_path)
            old_str = str(tool_input.get("old_string", ""))[:40]
            new_str = str(tool_input.get("new_string", ""))[:40]
            desc = f"{old_str} -> {new_str}" if old_str else "new content"
            state["edits"].append({
                "file": file_name,
                "description": desc,
                "timestamp": time.time(),
            })
            return {
                "category": "CHANGED",
                "desc": f"Edited {file_name}",
                "file": file_path,
                "timestamp": time.time(),
            }

        if tool_name == "Bash":
            command = str(tool_input.get("command", ""))
            output = str(tool_response.get("output", ""))
            exit_code = tool_response.get("exitCode", tool_response.get("exit_code", 0))
            cmd_short = command[:80]

            if exit_code != 0:
                error_line = _extract_error_line(output)
                return {
                    "category": "TRIED-FAILED",
                    "desc": f"Command failed: {cmd_short} -> {error_line}",
                    "error": error_line,
                    "command": cmd_short,
                    "timestamp": time.time(),
                }
            return {
                "category": "INFRA",
                "desc": f"Ran: {cmd_short}",
                "timestamp": time.time(),
            }

    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Error-resolution tracking — writes to memory_store
# ---------------------------------------------------------------------------

def _track_error_resolution(state: dict, activity: dict, session_id: str) -> dict | None:
    """Track failure->fix sequences. Returns hook output if resolution stored."""
    if not activity or not memory_store:
        return None

    error_seqs = state.get("error_sequences", [])

    # Record new errors
    if activity.get("category") == "TRIED-FAILED" and activity.get("error"):
        error_seqs.append({
            "error": activity["error"],
            "command": activity.get("command", ""),
            "timestamp": activity["timestamp"],
            "resolved": False,
        })
        state["error_sequences"] = error_seqs[-5:]
        return None

    # Check if a successful action resolves a pending error
    if activity.get("category") in ("CHANGED", "INFRA") and error_seqs:
        for err in error_seqs:
            if err.get("resolved"):
                continue
            if activity["timestamp"] - err["timestamp"] > 300:
                continue

            resolution_desc = activity.get("desc", "")
            error_desc = err.get("error", "")

            # Skip when descriptions are too short to be useful
            if len(error_desc) < 15 or len(resolution_desc) < 15:
                continue

            content = f"Resolved '{error_desc}' by: {resolution_desc}"

            stored = memory_store.append_learning({
                "type": "error_resolution",
                "content": content,
                "tags": ["error_resolution", "auto_extracted"],
                "session_id": session_id,
            })
            if stored:
                err["resolved"] = True
                state["error_sequences"] = error_seqs
                debug_log(f"Error resolution stored: {content[:80]}")
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PostToolUse",
                        "additionalContext": f"AUTO-LEARNING: Stored error-resolution: {content[:100]}"
                    }
                }
    return None


# ---------------------------------------------------------------------------
# Test result handling — writes to memory_store
# ---------------------------------------------------------------------------

def _handle_test_result(state: dict, tool_input: dict, tool_response: dict, session_id: str) -> dict | None:
    """Check if a Bash result is a test pass/fail. Returns output dict or None."""
    if not memory_store:
        return None

    command = str(tool_input.get("command", ""))
    output = str(tool_response.get("output", ""))
    exit_code = tool_response.get("exitCode", tool_response.get("exit_code"))

    test_patterns = ["test", "pytest", "vitest", "jest", "npm run test", "cargo test", "go test"]
    is_test = any(p in command.lower() for p in test_patterns)
    if not is_test:
        return None

    five_min_ago = time.time() - 300
    recent_edits = [e for e in state["edits"] if e.get("timestamp", 0) > five_min_ago]
    if not recent_edits:
        return None

    edit_summary = "; ".join(f"{e['file']}: {e['description']}" for e in recent_edits[:5])
    files_changed = ", ".join(set(e["file"] for e in recent_edits))

    pass_patterns = ["passed", "ok (", "success", "PASS"]
    fail_patterns = ["failed", "FAIL", "error", "ERROR"]

    is_pass = exit_code == 0 and any(p.lower() in output.lower() for p in pass_patterns)
    is_fail = exit_code != 0 or any(p in output for p in fail_patterns)

    if is_pass:
        content = (
            f"Tests passed after editing: {files_changed}. "
            f"Changes: {edit_summary}. "
            f"Test output summary: {output[:150]}"
        )
        stored = memory_store.append_learning({
            "type": "test_result",
            "content": content,
            "tags": ["test_pass", "auto_extracted"],
            "session_id": session_id,
        })
        if stored:
            state["edits"] = []
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": f"AUTO-LEARNING: Stored test-pass learning about {files_changed}"
                }
            }

    elif is_fail and recent_edits:
        # Skip empty/whitespace-only failure output — major junk source
        failure_snippet = output[:200].strip()
        if not failure_snippet:
            return None

        content = (
            f"Tests FAILED after editing: {files_changed}. "
            f"Changes: {edit_summary}. "
            f"Failure output: {failure_snippet}"
        )

        # Dedup: skip if very similar entry already exists
        existing = memory_store.search_memory_log(content[:80], max_results=1)
        if existing and existing[0].get("score", 0) > 0.6:
            return None

        stored = memory_store.append_learning({
            "type": "FAILED_APPROACH",
            "content": content,
            "tags": ["test_fail", "auto_extracted"],
            "session_id": session_id,
        })
        if stored:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": f"AUTO-LEARNING: Stored test-failure learning about {files_changed}"
                }
            }

    return None


# ---------------------------------------------------------------------------
# Signal-based LLM extraction — only RETRY_SUCCESS and CONFIG_BREAKTHROUGH
# ---------------------------------------------------------------------------

def _detect_memory_signal(activities: list[dict]) -> tuple[str, list[dict]] | None:
    """Scan recent activities for high-value memory patterns.

    Only returns patterns worth blocking ~7s for LLM evaluation:
    - RETRY_SUCCESS: 2+ TRIED-FAILED then CHANGED/INFRA success
    - CONFIG_BREAKTHROUGH: TRIED-FAILED then INFRA success on config/setup commands

    Returns (signal_type, relevant_activities) or None.
    """
    if len(activities) < 3:
        return None

    recent = activities[-8:]  # Look at last 8 activities
    categories = [a.get("category", "") for a in recent]

    # RETRY_SUCCESS: 2+ failures followed by success
    fail_count = 0
    last_fail_idx = -1
    for i, cat in enumerate(categories):
        if cat == "TRIED-FAILED":
            fail_count += 1
            last_fail_idx = i

    if fail_count >= 2 and last_fail_idx < len(categories) - 1:
        # Check if something after the last failure succeeded
        for i in range(last_fail_idx + 1, len(categories)):
            if categories[i] in ("CHANGED", "INFRA"):
                return ("RETRY_SUCCESS", recent)

    # CONFIG_BREAKTHROUGH: failure on config-like command then INFRA success
    config_keywords = ["config", "setup", "install", "pip", "npm", "apt",
                        "brew", "cargo", "env", "export", "chmod", "systemctl",
                        "docker", "ssh"]
    for i, act in enumerate(recent):
        if act.get("category") != "TRIED-FAILED":
            continue
        cmd = act.get("command", "").lower()
        if not any(kw in cmd for kw in config_keywords):
            continue
        # Check if followed by INFRA success
        for j in range(i + 1, len(recent)):
            if recent[j].get("category") == "INFRA":
                return ("CONFIG_BREAKTHROUGH", recent[i:j + 1])

    return None


def _try_llm_extraction_sync(session_id: str, signal_type: str, activities: list[dict]) -> int:
    """Synchronous LLM extraction. Called in forked child. Returns count of stored learnings."""
    if not memory_classifier or not memory_store:
        return 0

    learnings = memory_classifier.classify_activities(activities, timeout=8)
    if not learnings:
        return 0

    stored_count = 0
    for learning in learnings:
        success = memory_store.append_learning({
            "type": learning["type"],
            "content": learning["content"],
            "tags": learning.get("tags", []) + ["llm_extracted", signal_type.lower()],
            "session_id": session_id,
            "source": "auto_learning_signal",
        })
        if success:
            stored_count += 1
    return stored_count


def _try_llm_extraction(state: dict, session_id: str, signal_type: str, activities: list[dict]) -> dict | None:
    """Fork to background for LLM extraction. Parent returns None immediately."""
    import os as _os

    if not memory_classifier or not memory_store:
        return None

    # Mark processed BEFORE fork to prevent re-triggering
    state["signal_processed"] = True

    # Fork: parent returns immediately, child does LLM work
    try:
        pid = _os.fork()
    except (OSError, AttributeError):
        # Windows or fork failure: fall back to sync
        count = _try_llm_extraction_sync(session_id, signal_type, activities)
        if count > 0:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": f"AUTO-LEARNING: LLM extracted {count} insight(s) from {signal_type} pattern"
                }
            }
        return None

    if pid > 0:
        # Parent: return immediately (no hookSpecificOutput — child handles storage)
        return None

    # Child process: redirect output and do LLM work
    try:
        # WHY: Log dir uses config-based path for all Cortex state co-location
        log_dir = LOG_DIR
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "auto-learning-bg.log"
        fd = _os.open(str(log_file), _os.O_WRONLY | _os.O_CREAT | _os.O_APPEND, 0o644)
        _os.dup2(fd, 1)
        _os.dup2(fd, 2)
        _os.close(fd)

        import datetime as _dt
        ts = _dt.datetime.now().isoformat()
        print(f"\n[{ts}] BG extraction start: signal={signal_type}, activities={len(activities)}", flush=True)

        count = _try_llm_extraction_sync(session_id, signal_type, activities)
        print(f"[{ts}] BG extraction done: stored {count} learning(s)", flush=True)
    except Exception as e:
        try:
            print(f"BG extraction error: {e}", flush=True)
        except Exception:
            pass
    finally:
        _os._exit(0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    try:
        raw = sys.stdin.read()
        data = json.loads(raw)
    except Exception:
        print("{}")
        return

    tool_name = data.get("tool_name", "")
    tool_input = data.get("tool_input", {})
    tool_response = data.get("tool_response", {})
    session_id = data.get("session_id", "unknown")

    state = load_state(session_id)
    state["turn_count"] += 1

    # Track activity (for edit context + error sequences)
    activity = _track_activity(state, tool_name, tool_input, tool_response)
    if activity:
        state["activities"].append(activity)

    # Check for test results (Bash only)
    if tool_name == "Bash":
        test_result = _handle_test_result(state, tool_input, tool_response, session_id)
        if test_result:
            save_state(state)
            print(json.dumps(test_result))
            return

    # Error-resolution tracking
    if activity:
        resolution_result = _track_error_resolution(state, activity, session_id)
        if resolution_result:
            save_state(state)
            print(json.dumps(resolution_result))
            return

    # Signal-based LLM extraction (RETRY_SUCCESS, CONFIG_BREAKTHROUGH only)
    if memory_classifier and not state.get("signal_processed"):
        signal = _detect_memory_signal(state["activities"])
        if signal:
            signal_type, signal_activities = signal
            llm_result = _try_llm_extraction(state, session_id, signal_type, signal_activities)
            if llm_result:
                state["signal_processed"] = True  # One LLM extraction per session max
                save_state(state)
                print(json.dumps(llm_result))
                return

    save_state(state)
    print("{}")


if __name__ == "__main__":
    main()
