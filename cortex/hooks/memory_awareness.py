#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Memory Awareness Hook with Tiered Injection (File-Based)

UserPromptSubmit hook that:
1. Always injects MEMORY.md core memory
2. Searches memory_log.jsonl for relevant entries (keyword match + recency decay)
3. Injects compact memory directive (1-line, not 5-line)

Tiered approach based on match score:
- HOT (score > 0.3): Full content injection + memory ack gate
- WARM (score 0.1-0.3): Preview + hint
- COLD (score < 0.1): Nothing from search (MEMORY.md still injected)

ARCHITECTURE NOTES (DO NOT REMOVE — prevents future regression):
────────────────────────────────────────────────────────────────
This file is the READ SIDE of the memory system. It injects context into
every user prompt. The WRITE SIDE is stop_learning_extractor.py + memory_store.py.

CONTEXT REDUCTION (Phase 4, March 2026):
  Three optimizations reduce injected context ~40-50% with ZERO information loss:
  1. CLUSTER-MERGE: Groups HOT items with >40% Jaccard overlap. Shows first item's
     full content + "(+N related)". Corrections NEVER merged. See format_tiered_context().
  2. DIRECTIVE COMPRESSION: 5-line <memory-directive> -> 1-line. Saves ~140 chars/prompt.
     DO NOT remove entirely — Claude starts trying to manually manage memory files.
  3. CROSS-SOURCE DEDUP: Filters memory_log results that duplicate core_tagged results.
     Tracked by content[:80] keys.
  User explicitly rejected LLM-generated summaries (risk of losing details like flag
  names and config values). All content is shown at FULL length — no truncation.

SESSION BRIDGING (Phase 5, March 2026):
  get_recent_session_summary() checks for SESSION_CHECKPOINT entries first (written
  by stop_learning_extractor.py at session end), falls back to tag scanning.
  The fallback MUST stay — old sessions lack checkpoints.
────────────────────────────────────────────────────────────────
"""

import json
import os
import re
import sys
import time as _time
from pathlib import Path

# WHY: Import paths and thresholds from centralized config so every module
# references the same source of truth. No hardcoded paths.
from cortex.config import (
    PIPELINE_LOG as _MEMORY_PIPELINE_LOG,
    MEMORY_DIR,
    MEMORY_LOG_FILE,
    HOT_THRESHOLD,
    WARM_THRESHOLD,
    MAX_INJECTION_CHARS,
    TOTAL_BUDGET_CHARS,
)

# WHY: Package import replaces sys.path hacking. memory_store is the shared
# read/write engine for memory_log.jsonl and core_tagged.jsonl.
from cortex.store import memory_store


def _log(event: str, **kw):
    """Append one line to memory-pipeline.log. Never raises."""
    try:
        _MEMORY_PIPELINE_LOG.parent.mkdir(parents=True, exist_ok=True)
        ts = _time.strftime("%H:%M:%S")
        detail = " ".join(f"{k}={v}" for k, v in kw.items()) if kw else ""
        with open(_MEMORY_PIPELINE_LOG, "a") as f:
            f.write(f"[{ts}] {event} {detail}\n")
    except Exception:
        pass


def extract_intent(prompt: str) -> str:
    """Extract core intent keywords from user prompt for memory search.

    Always extracts keywords (not just as fallback). Keeps top 10 non-stop-word
    tokens to prevent score dilution in the scoring formula which divides
    matches by query token count.
    """
    _STOP_WORDS = {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'can', 'may', 'might', 'shall', 'must',
        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
        'into', 'about', 'between', 'through', 'during', 'before', 'after',
        'i', 'me', 'my', 'mine', 'you', 'your', 'yours', 'we', 'our', 'us',
        'he', 'she', 'it', 'they', 'them', 'his', 'her', 'its', 'their',
        'this', 'that', 'these', 'those', 'which', 'what', 'who', 'whom',
        'and', 'but', 'or', 'nor', 'not', 'no', 'if', 'so', 'yet',
        'just', 'how', 'why', 'when', 'where', 'then', 'than', 'also',
        'very', 'too', 'quite', 'really', 'much', 'more', 'most', 'some',
        'any', 'all', 'each', 'every', 'both', 'few', 'many', 'such',
        'like', 'want', 'need', 'know', 'think', 'look', 'make', 'get',
        'got', 'let', 'say', 'said', 'tell', 'told', 'give', 'gave',
        'take', 'took', 'come', 'came', 'going', 'went', 'see', 'saw',
        'last', 'night', 'yesterday', 'today', 'now', 'here', 'there',
        'still', 'already', 'even', 'well', 'back', 'only', 'again',
        'new', 'old', 'used', 'using', 'use', 'way', 'thing', 'things',
        'right', 'sure', 'okay', 'yeah', 'yes', 'please', 'thanks',
        'help', 'show', 'find', 'remember', 'don', 'didn', 'doesn',
        'working', 'project', 'stuff', 'done', 'lot', 'trying', 'started', 'start',
    }

    # Strip quoted text (>20 chars) to prevent feedback loops where
    # previously-injected memory text dominates the intent extraction
    cleaned = re.sub(r'"[^"]{20,}"', ' ', prompt)
    cleaned = re.sub(r'```[\s\S]{20,}?```', ' ', cleaned)

    # Strip punctuation, lowercase, split into words
    text = re.sub(r'[^\w\s-]', ' ', cleaned.lower())
    words = text.split()

    # Extract meaningful keywords: 3+ chars, not stop words, deduplicated
    keywords = []
    seen = set()
    for w in words:
        if len(w) >= 3 and w not in _STOP_WORDS and w not in seen:
            seen.add(w)
            keywords.append(w)

    # Return ALL keywords — the search function handles IDF-based selection
    # (keeps top-10 by IDF when >10 tokens, so scoring denominators stay bounded)
    return ' '.join(keywords) if keywords else prompt.strip()[:50]


def get_recent_session_summary(current_session_id: str, max_chars: int = 250) -> str:
    """Get topic summary from the most recent different session.

    First checks for SESSION_CHECKPOINT entries (structured summaries written at session end
    by stop_learning_extractor.py). Falls back to tag scanning if no checkpoint found.
    Returns a compact summary like "Last session topics: docker, DNS, cloudflare"

    WHY TWO PATHS:
      - SESSION_CHECKPOINT (fast path): Written at session end by stop_learning_extractor.py
        with is_session_end=True. Contains pre-assembled summary — no reconstruction needed.
        Format: "Session abc12345: learning1 | learning2 | learning3"
      - Tag scanning (fallback): For entries written before checkpointing was added, or if
        no checkpoint exists. Reconstructs topics from raw entry tags.
      The fallback MUST stay — old sessions won't have checkpoints, and checkpoint writing
      can fail (try/except, never blocks session end).

    WHY 'session_checkpoint' in skip_tags: Without it, the checkpoint entry's own tags
      would be counted as topics, creating circular references in the summary.
    """
    try:
        # WHY: Uses config-based path instead of hardcoded ~/.ai-controller/memory/
        log_file = MEMORY_LOG_FILE
        if not log_file.exists():
            return ""

        # Read last 4KB (roughly 30-50 entries)
        file_size = log_file.stat().st_size
        read_start = max(0, file_size - 4096)

        with open(log_file, 'r', encoding='utf-8') as f:
            if read_start > 0:
                f.seek(read_start)
                f.readline()  # Skip partial line
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]

        current_prefix = current_session_id[:8] if current_session_id else ""

        # Fast path: look for SESSION_CHECKPOINT from a different session
        for line in reversed(lines):
            try:
                entry = json.loads(line)
                if entry.get('type') != 'SESSION_CHECKPOINT':
                    continue
                sid = entry.get('session_id', '')
                if isinstance(sid, str) and sid[:8] != current_prefix and sid != 'unknown':
                    return entry.get('content', '')[:max_chars]
            except json.JSONDecodeError:
                continue

        # Fallback: tag scanning (original approach)
        prev_session_entries = []
        prev_session_id = None

        for line in reversed(lines):
            try:
                entry = json.loads(line)
                sid = entry.get('session_id', '')
                if isinstance(sid, str) and sid[:8] != current_prefix and sid != 'unknown':
                    if prev_session_id is None:
                        prev_session_id = sid[:8]
                    if sid[:8] == prev_session_id:
                        prev_session_entries.append(entry)
                    elif prev_session_entries:
                        break  # Found a different older session, stop
            except json.JSONDecodeError:
                continue

        if not prev_session_entries:
            return ""

        # Extract unique tags (skip generic ones)
        skip_tags = {'stop_hook', 'llm_extracted', 'auto_detected', 'in_session', 'repeated',
                     'user_correction', 'user_preference', 'auto_extracted', 'session_checkpoint'}
        all_tags = []
        for entry in prev_session_entries:
            for tag in entry.get('tags', []):
                if tag.lower() not in skip_tags and tag not in all_tags:
                    all_tags.append(tag)

        if not all_tags:
            # Fall back to content snippets
            snippets = [e.get('content', '')[:40] for e in prev_session_entries[:3]]
            summary = "Last session: " + " | ".join(snippets)
        else:
            summary = "Last session topics: " + ", ".join(all_tags[:8])

        return summary[:max_chars]
    except Exception:
        return ""


def format_tiered_context(results: list, intent: str, char_budget: int = 0) -> tuple:
    """Format context based on score tiers, respecting character budget.

    Args:
        char_budget: remaining characters available. 0 = use MAX_INJECTION_CHARS.
    Returns: (formatted_text, hot_items_list)
    """
    budget = char_budget if char_budget > 0 else MAX_INJECTION_CHARS
    hot_items = []
    warm_items = []

    for r in results:
        score = r.get('score', 0)
        entry = r.get('entry', {})
        content = entry.get('content', '')
        learning_type = entry.get('type', 'UNKNOWN')

        if score >= HOT_THRESHOLD:
            # WHY: Require >=2 exact token matches for HOT injection (defense in depth).
            # Entries that score high purely via stem/substring matching without real
            # token overlap are often cross-domain false positives.
            if r.get('exact_matches', 0) < 2:
                continue
            hot_items.append({
                'type': learning_type,
                'content': content,
                'score': score
            })
        elif score >= WARM_THRESHOLD:
            if r.get('exact_matches', 0) < 2:
                continue  # Single-word coincidental match — skip
            preview = ' '.join(content.split('\n')[:2])[:150]
            if len(content) > 150:
                preview += '...'
            warm_items.append({
                'type': learning_type,
                'preview': preview,
                'score': score
            })

    if not hot_items and not warm_items:
        return "", []

    # CLUSTER-MERGE: group HOT items with >40% content overlap to reduce repetition.
    # WHY: Search often returns 2-3 entries about the same topic (e.g., two docker
    # deployment entries). Showing all of them wastes context budget. Merging shows
    # the first one's full content + "(+N related)" suffix — zero info loss.
    # WHY 40% threshold: Lower catches unrelated items. Higher misses genuine dupes.
    # Tested with real search results — 40% Jaccard correctly groups same-topic entries.
    # WHY expanding token set: As cluster grows, union grows faster than intersection,
    # so Jaccard DECREASES — natural brake against mega-clusters. Max 5 items anyway.
    # CRITICAL: Corrections are NEVER merged — they contain specific commands/rules
    # (e.g., "Never run docker prune -a") where every word matters.
    _correction_types = {'USER_CORRECTION', 'repeated_correction', 'frustrated_correction'}
    if len(hot_items) > 1:
        mergeable = []
        protected = []
        for item in hot_items:
            if item['type'] in _correction_types:
                protected.append(item)
            else:
                item['_tokens'] = set(re.findall(r'\w{3,}', item['content'].lower()))
                mergeable.append(item)

        # Greedy clustering: merge items with >40% Jaccard overlap
        clusters = []
        used_idx = set()
        for i, a in enumerate(mergeable):
            if i in used_idx:
                continue
            cluster = [a]
            cluster_tokens = set(a['_tokens'])
            for j, b in enumerate(mergeable):
                if j <= i or j in used_idx:
                    continue
                if not cluster_tokens or not b['_tokens']:
                    continue
                jaccard = len(cluster_tokens & b['_tokens']) / len(cluster_tokens | b['_tokens'])
                if jaccard >= 0.4:
                    cluster.append(b)
                    cluster_tokens |= b['_tokens']
                    used_idx.add(j)
            clusters.append(cluster)

        # Rebuild hot_items: representative from each cluster + protected corrections
        merged_hot = []
        for cluster in clusters:
            rep = cluster[0]
            if len(cluster) > 1:
                rep = dict(rep)
                rep['content'] = rep['content'] + f" (+{len(cluster)-1} related)"
            merged_hot.append(rep)
        hot_items = protected + merged_hot

    lines = []
    used = 0

    if hot_items:
        lines.append("## Past session learnings")
        lines.append("")
        lines.append("Relevant context from previous sessions:")
        lines.append("")
        used += 60  # headers
        for item in hot_items:
            entry_text = f"### [{item['type']}]\n{item['content']}\n"
            if used + len(entry_text) > budget:
                break
            lines.append(f"### [{item['type']}]")
            lines.append(item['content'])
            lines.append("")
            used += len(entry_text)

    if warm_items and used < budget:
        lines.append(f"## POSSIBLE MATCHES (may be relevant)")
        used += 40
        for item in warm_items:
            entry_text = f"- [{item['type']}] {item['preview']}"
            if used + len(entry_text) > budget:
                break
            lines.append(entry_text)
            used += len(entry_text)

    return '\n'.join(lines), hot_items


def main():
    try:
        try:
            input_data = json.load(sys.stdin)
        except Exception:
            return

        # Skip for subagents and classifier subprocess
        if os.environ.get('CLAUDE_AGENT_ID'):
            return
        # WHY: Check both env var names for backward compatibility during migration
        if os.environ.get('CORTEX_CLASSIFYING') == '1':
            return
        if os.environ.get('AI_CONTROLLER_CLASSIFYING') == '1':
            return

        prompt = input_data.get('prompt', '')

        # Skip short prompts and slash commands
        if len(prompt) < 15 or prompt.strip().startswith('/'):
            return

        # Skip system noise: agent prompts, extraction directives, task notifications
        if prompt.lstrip().startswith('<task-notification>'):
            return
        if any(marker in prompt[:100] for marker in [
            'You are a lead engineer reviewing',
            'You are a plan reviewer',
            'Output ONLY valid JSON',
            'Extract learnings from this session',
        ]):
            return

        if not memory_store:
            return

        # Extract intent
        intent = extract_intent(prompt)
        if len(intent) < 3:
            return

        session_id = input_data.get('session_id', 'unknown')
        _log("SEARCH", session=session_id[:8], intent=intent[:60])

        # Intent bridge: write current intent for auto_learning hook correlation
        try:
            session_id = input_data.get('session_id', 'unknown')
            # WHY: Cache dir lives under MEMORY_DIR so all Cortex state is co-located
            cache_dir = MEMORY_DIR / "cache"
            intent_file = cache_dir / f"current-intent-{session_id}.json"
            intent_file.parent.mkdir(parents=True, exist_ok=True)
            intent_file.write_text(json.dumps({
                "prompt": prompt[:500],
                "intent": intent[:200],
                "session_id": session_id,
                "timestamp": __import__('time').time(),
            }))
        except Exception:
            pass

        # Build context parts with total budget tracking
        context_parts = []
        total_used = 0

        # 0.5. Recent session summary (bridges session gap)
        session_summary = get_recent_session_summary(session_id, max_chars=250)
        if session_summary:
            context_parts.append(f"## Session Context\n{session_summary}")
            context_parts.append("")
            total_used += len(session_summary) + 25  # +25 for header

        # 1. Search core_tagged.jsonl for intent-relevant distilled knowledge
        #    (universals now compete here as UNIVERSAL_DIRECTIVE entries with priority 5)
        remaining = TOTAL_BUDGET_CHARS - total_used
        tagged_results = memory_store.search_core_tagged(intent, max_results=5)
        # WHY: Compute exact match counts for gating (same as memory_log results below).
        # Core_tagged had NO exact match filtering — high-scoring stem/substring matches
        # could inject irrelevant distilled knowledge.
        intent_tokens_ct = set(re.findall(r'\w{3,}', intent.lower()))
        for r in tagged_results:
            entry = r['entry']
            ct_content = entry.get('content', '').lower()
            ct_tags = ' '.join(t.lower() for t in entry.get('tags', []))
            ct_tokens = set(re.findall(r'\w{3,}', f"{ct_content} {ct_tags}"))
            r['exact_matches'] = len(intent_tokens_ct & ct_tokens)
        if tagged_results and remaining > 200:
            hot_tagged = [r for r in tagged_results if r['score'] >= HOT_THRESHOLD and r.get('exact_matches', 0) >= 2]
            warm_tagged = [r for r in tagged_results if WARM_THRESHOLD <= r['score'] < HOT_THRESHOLD and r.get('exact_matches', 0) >= 2]
            if hot_tagged or warm_tagged:
                # WHY: Retrieved memory is useful context, but it is not proof for the
                # current task. Keep the trust boundary explicit in injected context.
                context_parts.append("## Relevant Knowledge (from past sessions, advisory only — revalidate locally)")
                context_parts.append("")
                for r in hot_tagged:
                    entry = r['entry']
                    section = entry.get('section', '')
                    content_text = entry.get('content', '')
                    if len(content_text) > 300:
                        content_text = content_text[:297] + '...'
                    entry_text = f"**[{section}]** {content_text}"
                    if total_used + len(entry_text) > TOTAL_BUDGET_CHARS:
                        break
                    context_parts.append(entry_text)
                    context_parts.append("")
                    total_used += len(entry_text)
                for r in warm_tagged:
                    entry = r['entry']
                    section = entry.get('section', '')
                    preview = ' '.join(entry.get('content', '').split('\n')[:2])[:150]
                    if len(entry.get('content', '')) > 150:
                        preview += '...'
                    entry_text = f"- [{section}] {preview}"
                    if total_used + len(entry_text) > TOTAL_BUDGET_CHARS:
                        break
                    context_parts.append(entry_text)
                    total_used += len(entry_text)
                context_parts.append("")

        # CROSS-SOURCE DEDUP: core_tagged.jsonl and memory_log.jsonl can contain
        # the same content (distilled entries exist in both). Without dedup, the
        # same fact appears twice in injected context, wasting ~400 chars per dupe.
        # We track content[:80] keys from core_tagged results and skip matching
        # memory_log results during the merge step below.
        _core_tagged_keys = set()
        if tagged_results:
            for r in tagged_results:
                _core_tagged_keys.add(r.get('entry', {}).get('content', '')[:80].lower())

        # 2. Parallel fused search: keyword + LLM-expanded semantic search
        remaining = TOTAL_BUDGET_CHARS - total_used
        search_context = ""
        hot_items = []

        def _keyword_search():
            return memory_store.search_memory_log(intent, max_results=5)

        def _semantic_search():
            try:
                expanded = memory_store.expand_query(intent)
                if expanded and expanded != intent:
                    raw_results = memory_store.search_memory_log(expanded, max_results=5)
                    # Two-pass: expansion finds candidates, scoring uses original intent
                    return memory_store.rescore_results(raw_results, intent)
            except Exception:
                pass
            return []

        from concurrent.futures import ThreadPoolExecutor
        kw_results = []
        sem_results = []
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                kw_future = pool.submit(_keyword_search)
                sem_future = pool.submit(_semantic_search)
                kw_results = kw_future.result(timeout=5) or []
                # Always collect semantic results — ThreadPoolExecutor.__exit__
                # waits for all threads anyway, so we pay the latency regardless.
                # Semantic results add value even when keyword search found HOT matches.
                try:
                    sem_results = sem_future.result(timeout=4) or []
                except Exception:
                    sem_results = []
        except Exception:
            kw_results = memory_store.search_memory_log(intent, max_results=5) or []

        # Merge and dedupe: keep highest score per unique content
        seen_content = {}
        for r in kw_results + sem_results:
            key = r.get('entry', {}).get('content', '')[:80]
            # Cross-source dedup: skip entries already shown from core_tagged
            if key.lower() in _core_tagged_keys:
                continue
            if key not in seen_content or r.get('score', 0) > seen_content[key].get('score', 0):
                seen_content[key] = r
        search_results = sorted(seen_content.values(), key=lambda x: x.get('score', 0), reverse=True)[:5]

        # Compute exact match counts against original intent for WARM filtering
        intent_tokens = set(re.findall(r'\w{3,}', intent.lower()))
        for r in search_results:
            entry = r.get('entry', {})
            entry_content = entry.get('content', '').lower()
            tags_str = ' '.join(t.lower() for t in entry.get('tags', []))
            entry_tokens = set(re.findall(r'\w{3,}', f"{entry_content} {tags_str}"))
            r['exact_matches'] = len(intent_tokens & entry_tokens)

        if search_results and remaining > 200:
            search_context, hot_items = format_tiered_context(search_results, intent, char_budget=remaining)
            if search_context:
                context_parts.append(search_context)
                total_used += len(search_context)
            sem_count = len(sem_results)
            _log("RESULTS", hot=len(hot_items), warm=len(search_results)-len(hot_items),
                 semantic_matches=sem_count, session=session_id[:8])

        # 3. MEMORY DIRECTIVE (compact, ~60 chars).
        # WHY compressed (not removed): Without this line, Claude sometimes wastes
        # time trying to manually write to memory files or asking the user about
        # memory storage. Full removal was tested and caused this behavior.
        # WHY not the old 5-line version: The old <memory-directive> block was ~200
        # chars of boilerplate injected on EVERY prompt. This 1-line version saves
        # ~140 chars per prompt while conveying the same message.
        context_parts.append("")
        context_parts.append("(Memory is auto-captured in background — no manual action needed.)")

        context = '\n'.join(context_parts)

        # Write marker file for PreToolUse gate when HOT matches exist
        if hot_items:
            try:
                session_id = input_data.get('session_id', 'unknown')
                # WHY: Marker dir lives under MEMORY_DIR so all Cortex state is co-located
                marker_dir = MEMORY_DIR
                marker_dir.mkdir(parents=True, exist_ok=True)
                marker_file = marker_dir / f"pending-memory-ack-{session_id}.txt"
                summary_lines = []
                for item in hot_items:
                    preview = item['content'].split('\n')[0][:100]
                    summary_lines.append(f"- [{item['type']}] {preview}")
                marker_data = {
                    "count": len(hot_items),
                    "summary": '\n'.join(summary_lines),
                    "intent": intent
                }
                marker_file.write_text(json.dumps(marker_data))
                _log("MARKER_CREATED", file=marker_file.name, hot=len(hot_items), session=session_id[:8])
            except Exception:
                pass
        else:
            _log("NO_HOT", session=session_id[:8])

        # Output for Claude
        output = {
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": context
            }
        }
        print(json.dumps(output))
    except Exception:
        # Fail-safe: never crash the hook process
        print("{}")


if __name__ == "__main__":
    main()
