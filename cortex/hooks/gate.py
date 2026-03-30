"""Cortex PreToolUse gate — forces Claude to acknowledge past session memories.

WHY: Without enforcement, Claude can ignore injected memories. This gate
blocks implementation tools until Claude mentions what it recalls. The marker
is self-consuming: one block per injection cycle, then all subsequent tools pass.

How it works:
  1. awareness.py (UserPromptSubmit) injects memories + writes pending-ack marker
  2. Claude tries an implementation tool → this gate fires
  3. Gate reads marker → self-consumes (deletes) → BLOCKS with memory info
  4. Claude mentions memories in text, retries the tool → marker gone → passes through
  5. 10-min TTL prevents stale markers from blocking forever

Usage in settings.json (registered by `cortex-install`):
  PreToolUse hook with matcher "Edit|Write|MultiEdit|NotebookEdit|Bash"
"""

import json
import sys
import time

from cortex.config import MEMORY_DIR


def main():
    # WHY: Read hook input from stdin — Claude Code passes tool call context as JSON
    try:
        input_data = json.loads(sys.stdin.read())
    except Exception:
        # FAIL-CLOSED: If we can't parse input, we can't check — pass through.
        # This is safe because the marker will be checked on the next tool call.
        return

    session_id = input_data.get("session_id", "unknown")
    marker_file = MEMORY_DIR / f"pending-memory-ack-{session_id}.txt"

    if not marker_file.exists():
        return  # No pending memories — pass through

    # WHY: 10-min TTL prevents stale markers (from crashed sessions, etc.)
    # from blocking tools indefinitely. 600s = generous window for Claude
    # to process and acknowledge, short enough to self-heal.
    try:
        marker_age = time.time() - marker_file.stat().st_mtime
    except OSError:
        return  # File vanished between exists() and stat() — race condition, pass through

    if marker_age > 600:
        try:
            marker_file.unlink()
        except OSError:
            pass
        return  # Stale marker cleaned up — pass through

    # Read marker content
    try:
        marker_data = json.loads(marker_file.read_text())
        match_count = marker_data.get("count", 0)
        summary = marker_data.get("summary", "")
    except Exception:
        match_count = 0
        summary = ""

    if match_count == 0:
        # WHY: Marker exists but no matches — shouldn't happen, clean up
        try:
            marker_file.unlink()
        except OSError:
            pass
        return

    # WHY: Self-consuming — delete marker BEFORE blocking. This ensures
    # the retry passes through. If delete fails, the marker persists and
    # blocks again (fail-closed), which is safe — the TTL will clean it up.
    try:
        marker_file.unlink()
    except OSError:
        pass

    # Block with memory info
    block_message = (
        f"STOP. {match_count} relevant learning(s) from past sessions found.\n"
        f"\n"
        f"{summary}\n"
        f"\n"
        f"Mention what you recall from past sessions that applies to this task "
        f"before proceeding. This is the normal workflow — not an error."
    )

    output = {
        "decision": "block",
        "reason": block_message,
    }
    print(json.dumps(output))


if __name__ == "__main__":
    main()
