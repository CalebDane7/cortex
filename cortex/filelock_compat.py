"""Cross-platform file locking for concurrent memory access.

WHY: memory_store.py uses file locking to prevent corruption when multiple
Claude Code sessions write to memory_log.jsonl simultaneously. Linux/macOS
use fcntl.flock; Windows uses msvcrt.locking. This module abstracts the
difference so Cortex works on all platforms.
"""

import os
import sys

# WHY: Platform detection at import time, not per-call. The platform doesn't
# change during a process lifetime.
_IS_WINDOWS = sys.platform == "win32"

if _IS_WINDOWS:
    import msvcrt

    def lock_file(f):
        """Acquire exclusive lock on file. Blocks until available."""
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)

    def unlock_file(f):
        """Release exclusive lock on file."""
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
else:
    import fcntl

    def lock_file(f):
        """Acquire exclusive lock on file. Blocks until available."""
        fcntl.flock(f, fcntl.LOCK_EX)

    def unlock_file(f):
        """Release exclusive lock on file."""
        fcntl.flock(f, fcntl.LOCK_UN)
