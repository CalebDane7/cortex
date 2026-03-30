"""Cortex installer — configures Claude Code hooks for the memory system.

Usage:
    python3 -m cortex.scripts.install           # Install hooks + create ~/.cortex/
    python3 -m cortex.scripts.install --dry-run  # Preview changes without writing
    python3 -m cortex.scripts.install --uninstall # Remove Cortex hooks from settings.json
"""

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

# WHY: These are the exact hook commands Cortex registers. Used both for
# installation (merging into settings.json) and uninstallation (identifying
# which entries to remove).
CORTEX_HOOK_COMMANDS = {
    "python3 -m cortex.hooks.memory_awareness",
    "python3 -m cortex.hooks.correction_detector",
    "python3 -m cortex.hooks.auto_learning",
    "python3 -m cortex.hooks.stop_learning_extractor",
    "python3 -m cortex.hooks.gate",
}

CORTEX_HOOKS = {
    "hooks": {
        "UserPromptSubmit": [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": "python3 -m cortex.hooks.memory_awareness",
                        "timeout": 10000,
                    }
                ]
            },
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": "python3 -m cortex.hooks.correction_detector",
                        "timeout": 5000,
                    }
                ]
            },
        ],
        "PostToolUse": [
            {
                "matcher": {
                    "type": "toolName",
                    "pattern": "Bash|Edit|Write|MultiEdit|Read|Grep|Glob|WebSearch|WebFetch",
                },
                "hooks": [
                    {
                        "type": "command",
                        "command": "python3 -m cortex.hooks.auto_learning",
                        "timeout": 10000,
                    }
                ],
            }
        ],
        "Stop": [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": "python3 -m cortex.hooks.stop_learning_extractor",
                        "timeout": 45000,
                    }
                ]
            }
        ],
        "PreToolUse": [
            {
                "matcher": "Edit|Write|MultiEdit|NotebookEdit|Bash",
                "hooks": [
                    {
                        "type": "command",
                        "command": "python3 -m cortex.hooks.gate",
                        "timeout": 5000,
                    }
                ],
            }
        ],
    }
}

SETTINGS_PATH = Path.home() / ".claude" / "settings.json"
CORTEX_DIR = Path.home() / ".cortex"

# WHY: Starter files are created empty so hooks don't need to handle the
# "first run" case — they can always assume the files exist.
STARTER_FILES = [
    "memory_log.jsonl",
    "core_tagged.jsonl",
    "aliases.json",
    "MEMORY.md",
]


def _is_cortex_hook_entry(entry: dict) -> bool:
    """Return True if a hook entry belongs to Cortex."""
    hooks_list = entry.get("hooks", [])
    for hook in hooks_list:
        if hook.get("command", "") in CORTEX_HOOK_COMMANDS:
            return True
    return False


def _deep_merge_hooks(existing: dict, cortex: dict) -> dict:
    """Merge Cortex hooks into existing settings without clobbering.

    For each hook event (UserPromptSubmit, PostToolUse, Stop):
    - Remove any existing Cortex entries (by command match) to avoid duplicates
    - Append the fresh Cortex entries at the end
    - Preserve all non-Cortex hook entries untouched
    """
    # WHY: deepcopy prevents mutation of the original dict, which would make
    # the merged == existing comparison always return True (shallow copy bug).
    result = copy.deepcopy(existing)
    if "hooks" not in result:
        result["hooks"] = {}

    cortex_hooks = cortex["hooks"]
    for event_name, cortex_entries in cortex_hooks.items():
        existing_entries = result["hooks"].get(event_name, [])

        # WHY: Filter out old Cortex entries first, then append new ones.
        # This handles upgrades cleanly — old hook definitions get replaced.
        cleaned = [e for e in existing_entries if not _is_cortex_hook_entry(e)]
        cleaned.extend(cortex_entries)
        result["hooks"][event_name] = cleaned

    return result


def _remove_cortex_hooks(settings: dict) -> dict:
    """Remove all Cortex hook entries from settings, preserving everything else."""
    result = dict(settings)
    if "hooks" not in result:
        return result

    hooks = result["hooks"]
    empty_events = []

    for event_name, entries in hooks.items():
        if isinstance(entries, list):
            hooks[event_name] = [e for e in entries if not _is_cortex_hook_entry(e)]
            if not hooks[event_name]:
                empty_events.append(event_name)

    # WHY: Clean up empty event arrays so settings.json stays tidy.
    for event_name in empty_events:
        del hooks[event_name]

    if not hooks:
        del result["hooks"]

    return result


def create_starter_files(dry_run: bool = False) -> list[str]:
    """Create ~/.cortex/ with empty starter files. Returns list of actions taken."""
    actions = []

    if not CORTEX_DIR.exists():
        actions.append(f"  Create directory: {CORTEX_DIR}")
        if not dry_run:
            CORTEX_DIR.mkdir(parents=True, exist_ok=True)

    logs_dir = CORTEX_DIR / "logs"
    if not logs_dir.exists():
        actions.append(f"  Create directory: {logs_dir}")
        if not dry_run:
            logs_dir.mkdir(parents=True, exist_ok=True)

    for filename in STARTER_FILES:
        filepath = CORTEX_DIR / filename
        if not filepath.exists():
            actions.append(f"  Create file: {filepath}")
            if not dry_run:
                if filename == "aliases.json":
                    filepath.write_text("{}\n")
                elif filename == "MEMORY.md":
                    filepath.write_text("# Cortex Memory\n\nThis file is auto-managed by Cortex.\n")
                else:
                    filepath.write_text("")

    if not actions:
        actions.append("  ~/.cortex/ already exists with all starter files")

    return actions


def install_hooks(dry_run: bool = False) -> list[str]:
    """Merge Cortex hooks into ~/.claude/settings.json. Returns list of actions."""
    actions = []

    settings_dir = SETTINGS_PATH.parent
    if not settings_dir.exists():
        actions.append(f"  Create directory: {settings_dir}")
        if not dry_run:
            settings_dir.mkdir(parents=True, exist_ok=True)

    if SETTINGS_PATH.exists():
        existing = json.loads(SETTINGS_PATH.read_text())
        actions.append(f"  Read existing: {SETTINGS_PATH}")
    else:
        existing = {}
        actions.append(f"  Create new: {SETTINGS_PATH}")

    merged = _deep_merge_hooks(existing, CORTEX_HOOKS)

    if merged == existing:
        actions.append("  Cortex hooks already present in settings.json")
    else:
        actions.append("  Merge Cortex hooks into settings.json:")
        for event_name in CORTEX_HOOKS["hooks"]:
            count = len(CORTEX_HOOKS["hooks"][event_name])
            actions.append(f"    {event_name}: +{count} hook(s)")

        if not dry_run:
            SETTINGS_PATH.write_text(json.dumps(merged, indent=2) + "\n")

    return actions


def uninstall_hooks(dry_run: bool = False) -> list[str]:
    """Remove Cortex hooks from ~/.claude/settings.json. Returns list of actions."""
    actions = []

    if not SETTINGS_PATH.exists():
        actions.append("  No settings.json found — nothing to uninstall")
        return actions

    existing = json.loads(SETTINGS_PATH.read_text())
    cleaned = _remove_cortex_hooks(existing)

    if cleaned == existing:
        actions.append("  No Cortex hooks found in settings.json")
    else:
        actions.append("  Remove Cortex hooks from settings.json:")
        for event_name in CORTEX_HOOKS["hooks"]:
            actions.append(f"    {event_name}: removed")

        if not dry_run:
            SETTINGS_PATH.write_text(json.dumps(cleaned, indent=2) + "\n")

    return actions


def main():
    parser = argparse.ArgumentParser(
        description="Install or uninstall Cortex memory hooks for Claude Code"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing anything",
    )
    parser.add_argument(
        "--uninstall",
        action="store_true",
        help="Remove Cortex hooks from settings.json",
    )
    args = parser.parse_args()

    # Preflight: check for Claude Code
    if not shutil.which("claude"):
        print("ERROR: Claude Code CLI ('claude') not found on PATH.")
        print("Install it first: https://docs.anthropic.com/claude-code/getting-started")
        sys.exit(1)

    if args.dry_run:
        print("=== DRY RUN (no files will be modified) ===\n")

    if args.uninstall:
        print("Uninstalling Cortex hooks...")
        actions = uninstall_hooks(dry_run=args.dry_run)
        for action in actions:
            print(action)
        print()
        print("Memory data preserved at ~/.cortex/")
        print("To remove data: rm -rf ~/.cortex/")
    else:
        print("Installing Cortex memory system...\n")

        print("Step 1: Create ~/.cortex/ directory")
        for action in create_starter_files(dry_run=args.dry_run):
            print(action)

        print("\nStep 2: Configure Claude Code hooks")
        for action in install_hooks(dry_run=args.dry_run):
            print(action)

        print("\n" + "=" * 50)
        print("Cortex installed successfully!")
        print("=" * 50)
        print()
        print("Next steps:")
        print("  1. Start a new Claude Code session")
        print("  2. Cortex hooks activate automatically")
        print("  3. Correct Claude when it gets things wrong — Cortex remembers")
        print()
        print("Memory data:    ~/.cortex/")
        print("Hook config:    ~/.claude/settings.json")
        print("View logs:      cat ~/.cortex/logs/memory-pipeline.log")


if __name__ == "__main__":
    main()
