#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Correction Detector — captures user corrections and repeated instructions.

UserPromptSubmit hook that detects when the user is correcting Claude or
repeating an instruction, and stores it as a high-priority memory entry.

Signals detected:
- Frustration patterns: "I already told you", "again", "stop doing", "how many times"
- Negation+instruction: "no, use X", "don't use Y", "never do Z"
- Explicit preferences: "always use", "never use", "prefer X over Y"
- Repetition: same correction stored twice = REPEATED_CORRECTION (highest priority)
"""

import json
import os
import re
import sys
from pathlib import Path

# WHY: Package imports replace sys.path hacking. memory_store is the shared
# read/write engine, memory_classifier provides LLM-based correction detection.
from cortex.store import memory_store

try:
    from cortex.classifiers import memory_classifier
except ImportError:
    memory_classifier = None

# Correction signal patterns — each returns the extracted instruction
CORRECTION_PATTERNS = [
    # Frustration + correction
    (r"(?:i (?:already|just) (?:told|said|mentioned|explained)|how many times|stop (?:doing|using)|quit (?:doing|using)|i keep (?:telling|saying|having to))\b[^.!?]*[.!?]?\s*(.{10,300})", "FRUSTRATED_CORRECTION"),
    # Direct negation + instruction
    (r"(?:^|\. )(?:no[,.]?\s+)(?:use|do|try|run|make|set|put|change|switch to)\s+(.{10,200})", "USER_CORRECTION"),
    (r"(?:^|\. )(?:don'?t|do not|never|stop)\s+(?:use|do|try|run|make|set|put|change|add|announce|mention|output|print|show|display|say)\s+(.{10,200})", "USER_CORRECTION"),
    # "instead" pattern — "use X instead of Y" or "instead of X, use Y"
    (r"(?:instead of .{5,80}?,?\s*(?:use|do|try|run)\s+.{5,100}|(?:use|do|try|run)\s+.{5,100}\s+instead\b.{0,80})", "USER_CORRECTION"),
    # Explicit preference declarations
    (r"(?:always|never|prefer|i want you to|you should always|you should never|the rule is|the pattern is)\s+(.{10,250})", "USER_PREFERENCE"),
    # "not X, Y" or "X not Y" correction
    (r"(?:use|do|run)\s+(\w+)[,.]?\s+not\s+(\w+)", "USER_CORRECTION"),
    # Repeat/again signals
    (r"(?:again|once more|like i said|as i said|like before|same as before|remember)[,:]?\s+(.{10,200})", "USER_CORRECTION"),
]

# Minimum prompt length to analyze (skip short messages)
MIN_PROMPT_LENGTH = 20

# Words that strongly signal a correction context
CORRECTION_SIGNAL_WORDS = {
    "don't", "dont", "never", "stop", "wrong", "incorrect", "no,", "no.",
    "instead", "not that", "already told", "again", "repeat", "keep telling",
    "how many times", "i said", "like i said", "prefer", "always use",
    "never use", "should always", "should never", "the rule is",
}


def _is_worth_storing(content: str) -> bool:
    """Structural quality gate — rejects mid-sentence fragments and filler.

    Uses structural signals (not regex on content) to determine quality:
    - Minimum 5 words
    - At least 3 unique meaningful words (>3 chars)
    - Uppercase start = complete sentence (lowercase = mid-sentence fragment)
    """
    if not content:
        return False
    words = content.split()
    if len(words) < 5:
        return False
    meaningful = {w.lower() for w in words if len(w) > 3}
    if len(meaningful) < 3:
        return False
    # Uppercase start = complete sentence. Lowercase = mid-sentence fragment.
    # Exception: CLI commands that naturally start lowercase
    if content[0].islower() and not content.startswith(("npm", "git", "pip", "apt", "ssh", "curl", "docker")):
        return False
    return True


def has_correction_signals(prompt: str) -> bool:
    """Quick check if prompt likely contains a correction."""
    prompt_lower = prompt.lower()
    return any(signal in prompt_lower for signal in CORRECTION_SIGNAL_WORDS)


def extract_correction(prompt: str) -> dict | None:
    """Extract correction content from user prompt."""
    # Match against original prompt (preserves casing for structural quality check)
    prompt_stripped = prompt.strip()

    for pattern, correction_type in CORRECTION_PATTERNS:
        match = re.search(pattern, prompt_stripped, re.IGNORECASE | re.DOTALL)
        if match:
            # Use full match if no group, else first group
            content = match.group(1) if match.lastindex else match.group(0)
            content = content.strip()
            # Clean up
            content = re.sub(r'\s+', ' ', content)
            if len(content) < 10:
                continue
            # Sentence-boundary truncation: if >120 chars, cut at first sentence end
            if len(content) > 120:
                for end_char in ('.', '!', '?'):
                    idx = content.find(end_char, 40)
                    if 40 < idx < 200:
                        content = content[:idx + 1]
                        break
            return {
                "type": correction_type,
                "content": content[:300],
                "full_prompt": prompt[:500],
            }
    return None


def extract_smart_tags(content: str, full_prompt: str) -> list[str]:
    """Extract actionable tags from correction content for better search matching."""
    combined = f"{content} {full_prompt}".lower()
    # Extract meaningful nouns/verbs (skip very common words)
    stop_words = {
        'a', 'an', 'the', 'is', 'are', 'was', 'be', 'been', 'have', 'has',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'can',
        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
        'i', 'me', 'my', 'you', 'your', 'we', 'it', 'this', 'that',
        'and', 'but', 'or', 'if', 'so', 'just', 'how', 'what', 'why',
        'not', 'don', 'use', 'do', 'make', 'get', 'set', 'put', 'try',
        'no', 'yes', 'never', 'always', 'stop', 'keep', 'told', 'said',
        'instead', 'already', 'again', 'like', 'want', 'need', 'think',
    }
    words = re.findall(r'\b[a-z][a-z0-9_-]{2,}\b', combined)
    tags = []
    seen = set()
    for word in words:
        if word not in stop_words and word not in seen:
            seen.add(word)
            tags.append(word)
        if len(tags) >= 8:
            break
    return tags


def check_repetition(content: str) -> tuple[bool, float]:
    """Check if a similar correction was already stored recently.
    Returns (is_repeat, max_similarity_score)."""
    if not memory_store:
        return False, 0.0
    results = memory_store.search_memory_log(content, max_results=3)
    max_score = 0.0
    for r in results:
        entry = r.get("entry", {})
        entry_type = entry.get("type", "")
        if entry_type in ("USER_CORRECTION", "FRUSTRATED_CORRECTION", "USER_PREFERENCE", "REPEATED_CORRECTION"):
            score = r.get("score", 0)
            max_score = max(max_score, score)
    return max_score > 0.4, max_score


def main():
    try:
        try:
            input_data = json.load(sys.stdin)
        except Exception:
            return

        # Skip subagents and classifier subprocess
        if os.environ.get('CLAUDE_AGENT_ID'):
            return
        # WHY: Check both env var names for backward compatibility during migration
        if os.environ.get('CORTEX_CLASSIFYING') == '1':
            return
        if os.environ.get('AI_CONTROLLER_CLASSIFYING') == '1':
            return

        prompt = input_data.get('prompt', '')
        if len(prompt) < MIN_PROMPT_LENGTH:
            return

        if not memory_store:
            return

        # Try regex first (fast path)
        correction = None
        if has_correction_signals(prompt):
            correction = extract_correction(prompt)

        # LLM fallback for subtle corrections missed by regex
        if not correction and memory_classifier and len(prompt) > 50:
            correction = memory_classifier.is_correction(prompt, timeout=4)

        if not correction:
            return

        # Structural quality gate — reject mid-sentence fragments and filler
        if not _is_worth_storing(correction["content"]):
            print("{}")
            return

        # Check for repetition — upgrade to REPEATED_CORRECTION or skip if duplicate
        is_repeat, similarity_score = check_repetition(correction["content"])

        # Dedup: skip storage entirely for near-identical corrections (score > 0.6)
        if similarity_score > 0.6:
            print("{}")
            return

        if is_repeat:
            correction["type"] = "REPEATED_CORRECTION"

        # Extract smart tags for better future retrieval
        tags = extract_smart_tags(correction["content"], correction["full_prompt"])
        tags.extend(["user_correction", "auto_detected"])
        if is_repeat:
            tags.append("repeated")

        session_id = input_data.get("session_id", "unknown")

        stored = memory_store.append_learning({
            "type": correction["type"],
            "content": correction["content"],
            "tags": tags,
            "session_id": session_id,
            "source": "correction_detector",
        })

        if stored:
            priority = "REPEATED" if is_repeat else "NEW"
            output = {
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": f"[MEMORY] {priority} correction captured: {correction['content'][:80]}..."
                }
            }
            print(json.dumps(output))
        else:
            print("{}")
    except Exception:
        # Fail-safe: never crash the hook process
        print("{}")


if __name__ == "__main__":
    main()
