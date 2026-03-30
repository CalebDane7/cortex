# Cortex Architecture

A technical deep dive into how Cortex works, why each design decision was made, and how to port the system to any LLM tool.

---

## 1. System Overview

```
                          CAPTURE-STORE-SEARCH-INJECT LOOP

    +-----------+     +-----------+     +-----------+     +-----------+
    |  CAPTURE  |     |   STORE   |     |  SEARCH   |     |  INJECT   |
    +-----------+     +-----------+     +-----------+     +-----------+
    |           |     |           |     |           |     |           |
    | Hooks:    |     | Dedup:    |     | Scoring:  |     | Tiered:   |
    | - PostTool| --> | - Hash    | ... | - IDF     | --> | - HOT     |
    | - Error   |     | - Jaccard |     | - Stems   |     | - WARM    |
    | - Correct |     | - 3-way   |     | - Tags    |     | - COLD    |
    | - Stop    |     |   decide  |     | - Decay   |     |           |
    |           |     |           |     | - Boost   |     | Cluster   |
    | Classify: |     | Append OR |     |           |     | merge HOT |
    | - Pattern |     | Supersede |     | Cache:    |     | items     |
    | - LLM     |     | OR Skip   |     | - SQLite  |     |           |
    |           |     |           |     | - 12-56x  |     | Budget:   |
    +-----------+     +-----------+     | speedup   |     | 4000 char |
         |                |             +-----------+     | hard cap  |
         |                |                  |            +-----------+
         |                |                  |                 |
         v                v                  v                 v
    +---------------------------------------------------------------+
    |                    memory_log.jsonl                            |
    |  (append-only JSONL -- source of truth for all entries)       |
    +---------------------------------------------------------------+
    |                    memory_cache.db                             |
    |  (SQLite acceleration layer -- pre-computed tokens/stems)     |
    +---------------------------------------------------------------+
    |                    core_tagged.jsonl                           |
    |  (distilled knowledge -- no recency decay applied)            |
    +---------------------------------------------------------------+

    Read path:  User prompt --> extract_intent --> parallel search -->
                score --> tier --> cluster-merge --> inject context
    Write path: Hook fires --> classify --> dedup check --> store -->
                invalidate cache
```

The read path fires on every `UserPromptSubmit` hook. The write path fires on `PostToolUse` (errors, test results), `UserPromptSubmit` (corrections, preferences), and `Stop` (session-end extraction). The loop closes because better memories produce better Claude responses, which produce better memories.

---

## 2. The Scoring Algorithm

The scoring pipeline is the heart of Cortex. Every search result passes through this function:

```
score = match_score * tag_boost * type_boost * priority_boost * decay * coverage_factor
```

### Three-Tier Token Matching

Token matching happens in a strict priority cascade. If a higher tier matches, lower tiers are only used for the remaining unmatched tokens.

| Tier | Method | Weight | Description |
|------|--------|--------|-------------|
| 1 | Exact match | 1.0x (IDF-weighted) | Query token appears verbatim in entry |
| 2 | Stem match | 0.7x (IDF-weighted) | `_stem(query_token) == _stem(entry_token)` |
| 3 | Substring match | 0.5x (IDF-weighted) | One token is a substring of the other (min 50% length ratio) |

The stemmer is a lightweight suffix-stripping function (no NLTK dependency). It handles common English suffixes: `-tion`, `-ing`, `-ed`, `-ly`, `-er`, `-ment`, `-ness`, `-able`, `-ful`, `-ous`, `-ive`, plurals, and about 30 others. This is intentionally simple -- it catches morphological variants ("deploying" -> "deploy") without the weight of a full NLP pipeline.

### IDF Weighting

```python
idf(token) = log(1 + N / df)
```

Where `N` is the total number of entries and `df` is the count of entries containing that token. Smoothed IDF (`log(1 + ...)`) ensures tokens appearing in every entry still have positive weight (~0.69) rather than zero.

The match score uses IDF as both numerator and denominator weights:

```python
# Numerator: sum of IDF weights for matched tokens (at their tier weight)
exact_w = sum(idf[t] for t in exact_matches)            # 1.0x
stem_w  = sum(idf[t] for t in stem_only_matches) * 0.7  # 0.7x
sub_w   = substring_score * 0.5                          # 0.5x

# Denominator: total IDF weight of all query tokens
total_w = sum(idf[t] for t in query_tokens)

match_score = (exact_w + stem_w + sub_w) / total_w
```

The effect: rare, domain-specific tokens (like "kubernetes" or "dockerfile") contribute far more than common tokens (like "config" or "error"). If a 4,400-entry corpus has "kubernetes" in 12 entries and "error" in 800, a match on "kubernetes" is worth ~40x more than a match on "error".

IDF weights are cached in SQLite and refreshed when >50 new entries are added or >24 hours have elapsed. The change from a single supersede is <0.01% across thousands of entries -- negligible.

### Alias Expansion

Aliases are a static JSON map (e.g., `{"myapp": ["my-app", "my_app"]}`) that allows named-entity matching without polluting the scoring denominator. When a query token has aliases, those aliases are checked against entry tokens, but the denominator stays as the count of original query tokens. This means aliases improve recall without diluting precision.

### Boost Multipliers

| Boost | Multiplier | Condition |
|-------|-----------|-----------|
| Tag boost | 2.0x | Query tokens overlap with entry's `tags` field |
| Type boost | 1.5x | Entry type is `user_correction`, `repeated_correction`, `user_preference`, `frustrated_correction`, or `universal_directive` |
| Priority boost | 1.0 + (priority * 0.1) | For distilled entries with explicit priority field |

All boosts are gated by **multiplier eligibility**:

```python
multiplier_eligible = (
    match_score >= 0.3                                    # strong match
    or (match_score >= 0.15 and len(exact_matches) > 0)   # decent match + real overlap
) and len(query_tokens) >= 3
```

This prevents weak matches (stem-only or substring-only with low base scores) from being promoted to HOT tier by a 2x tag boost. A match_score of 0.12 with a 2x tag boost would be 0.24 -- just under HOT threshold. Without the gate, this would inject full content for a marginal match.

### Recency Decay

```python
decay = e^(-0.03 * age_in_days)
```

| Age | Retention |
|-----|-----------|
| 1 day | 97% |
| 7 days | 81% |
| 30 days | 41% |
| 60 days | 17% |
| 90 days | 7% |

The rate of 0.03 was tuned empirically. Fast enough that old entries naturally fade. Slow enough that a correction from last week still matters. Core tagged entries (distilled knowledge) are exempt from decay -- they are timeless.

### Coverage Factor

A bidirectional relevance penalty for superficial matches in large entries:

```python
if exact_matches and len(entry_tokens) > 3:
    entry_coverage = len(exact_matches & entry_tokens) / sqrt(len(entry_tokens))
    coverage_factor = max(0.15, min(1.0, entry_coverage * 3))
else:
    coverage_factor = 1.0
```

An entry with 200 tokens that matches 2 query tokens is probably a cross-topic coincidence. The coverage factor penalizes this. The square root in the denominator provides diminishing penalty -- an entry twice as long is not penalized twice as harshly.

---

## 3. Why Not FTS5?

SQLite FTS5 was A/B tested as both a primary search replacement and a pre-filter. The test used 10 representative queries against a 4,400+ entry corpus.

### As Primary Search: 4/10 Correct

FTS5 misses:
- **Alias expansion**: "myapp" doesn't FTS5-match "my-app"
- **Stem matching**: FTS5's built-in stemmer is English Porter, but the scoring pipeline uses custom suffix stripping tuned for technical terms
- **Tag boosting**: FTS5 ranks by BM25 only -- no way to boost entries whose tags match the query
- **Recency decay**: FTS5 has no temporal signal
- **Coverage factor**: FTS5 has no bidirectional relevance check

### As Pre-Filter: 19-96% Miss Rates

The idea was to use FTS5 to narrow candidates, then score them with `_score_entry()`. But FTS5 filters out entries that only match via stems, aliases, or substrings -- the exact entries the custom pipeline exists to find. Miss rates ranged from 19% (simple queries) to 96% (queries relying on alias/stem matching).

### Conclusion

Custom scoring beats general-purpose full-text search for this use case. FTS5 is designed for document retrieval where the query vocabulary closely matches the corpus vocabulary. Memory entries are written by hooks in a different session context than the user's current query, so vocabulary mismatch is the norm, not the exception.

---

## 4. Why Soundex Was Removed

Soundex (American phonetic encoding) was originally used for fuzzy matching. It maps words to 4-character codes based on pronunciation: "kubernetes" and "kubernates" both encode to `K165`.

### Worked at Small Scale

Below ~2,000 entries, Soundex was useful. It caught typos and phonetic variants that exact matching would miss.

### Failed at 4,400+ Entries

The Soundex alphabet produces only ~7,000 unique codes for the English language. With 4,400+ entries across diverse domains (Rust/iced GUI, memory system, Docker deployment, DNS configuration), collisions became rampant:

- Rust/iced GUI entries matching memory system queries
- Docker entries matching database queries
- Completely unrelated entries sharing Soundex codes

The false positive rate made search results unreliable.

### What Replaced It

Three existing features already covered every legitimate Soundex use case:

| Use Case | Before (Soundex) | After |
|----------|------------------|-------|
| Typos ("kubernates") | Phonetic match | Substring match (50% length ratio gate) |
| Named entities ("myapp" for "my-app") | Phonetic match | Alias expansion (aliases.json) |
| Morphological variants ("deploying"/"deploy") | Phonetic match (incidental) | Stem matching |

Soundex was removed from both the scoring function and the cache pre-filter. The `_soundex()` function remains in the codebase as dead code (SQLite schema still references the column) but is never used in scoring decisions.

---

## 5. Deduplication

Every new entry goes through a three-way decision before being stored. This is inspired by Mem0's ADD/UPDATE/DELETE/NOOP pattern, but uses Jaccard similarity instead of an LLM call.

### Decision Tree

```
                    new entry arrives
                          |
                    [O(1) hash check]
                     /            \
                  match          no match
                   |                |
                 SKIP         [Jaccard scan of last 50 entries]
             (exact dupe)        /          |           \
                           J > 0.5      0.3-0.5       J < 0.3
                              |        AND new         |
                            SKIP      is longer      APPEND
                        (near-dupe)      |          (new topic)
                                     SUPERSEDE
                                   (update in-place)
```

### Jaccard Similarity

```python
def jaccard(a, b):
    tokens_a = set(tokenize(a))
    tokens_b = set(tokenize(b))
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
```

### The Three Outcomes

| Outcome | Condition | Action |
|---------|-----------|--------|
| **SKIP** | Jaccard > 0.5 OR content hash match | Do nothing. Near-duplicate. |
| **SUPERSEDE** | Jaccard 0.3-0.5 AND new entry is longer | Replace old entry in-place. Mark cache dirty. |
| **APPEND** | Jaccard < 0.3 | Append as new entry. Append to cache inline. |

### O(1) Fast Path

Before the Jaccard scan, a content hash check runs against the SQLite cache:

```python
content_hash = md5(content.lower())[:16]
SELECT 1 FROM entry_cache WHERE content_hash = ? LIMIT 1
```

This catches exact duplicates in constant time. The Jaccard scan only runs for entries that pass the hash check.

### Cache Invalidation After Supersede

When an entry is superseded (replaced in-place), the line-based search cache is invalidated via `mark_dirty()`. This creates a `.cache-dirty` marker file that triggers a full cache rebuild on the next search. Append-only operations update the cache inline without a rebuild.

---

## 6. Tiered Injection

Search results are divided into three tiers based on their final score:

| Tier | Score Range | Injection Style | Context Cost |
|------|------------|-----------------|--------------|
| **HOT** | >= 0.3 | Full content | 100-500 chars per entry |
| **WARM** | 0.15 - 0.3 | Preview (first 2 lines, 150 chars) | ~150 chars per entry |
| **COLD** | < 0.15 | Not injected | 0 |

### Exact Match Gate

Even if an entry scores >= 0.3, it must have **at least 2 exact token matches** against the original query to be injected as HOT:

```python
if r.get('exact_matches', 0) < 2:
    continue  # Demoted to nothing, not even WARM
```

This is defense in depth. An entry that scores high purely via stem or substring matching without real token overlap is often a cross-domain false positive. The gate prevents these from wasting context budget.

### Injection Format

```markdown
## Past session learnings

Relevant context from previous sessions:

### [USER_CORRECTION]
Never run docker prune -a without explicit user confirmation.

### [WORKING_SOLUTION]
The kubernetes networking fix requires updating CoreDNS configmap. (+2 related)

## POSSIBLE MATCHES (may be relevant)
- [CONFIG_INSIGHT] DNS propagation takes up to 48h for TTL changes...
```

---

## 7. Cluster-Merge

When multiple HOT items overlap significantly, they are clustered to reduce context consumption.

### Algorithm

1. Separate corrections from mergeable entries. Corrections are **never** merged -- every word in "never run docker prune -a" matters.
2. For each mergeable entry, compute its token set.
3. Greedy clustering: for each unassigned entry, find all other entries with > 40% Jaccard overlap against the growing cluster token set.
4. The first entry in each cluster is the representative. Others are summarized as `(+N related)`.

### Why 40% Jaccard Threshold

Lower thresholds (20-30%) catch unrelated entries that happen to share common technical vocabulary. Higher thresholds (50-60%) miss genuine duplicates that use different phrasing. 40% was validated against real search results: it correctly groups entries about the same topic while keeping distinct topics separate.

### Natural Brake on Mega-Clusters

As a cluster grows, its union token set grows faster than the intersection with new candidates. This means Jaccard naturally decreases for each additional candidate, creating a self-limiting cluster size. A hard cap of 5 items per cluster provides a safety net.

### Impact

Cluster-merge reduces context injection by ~40-50% in multi-result scenarios with zero information loss. The representative entry's full content is preserved; only the redundant entries are summarized.

---

## 8. Query Expansion

Query expansion adds semantic recall that keyword matching alone cannot provide. "Delete the deployment" should find entries about "remove", "erase", "teardown", and "destroy".

### Two-Pass Pattern

```
Pass 1: Expand query with LLM-generated synonyms
Pass 2: Score results against the ORIGINAL query (not expanded)
```

This is the critical design decision. Expansion improves **recall** -- it finds entries that use different vocabulary for the same concept. But scoring against the expanded query would hurt **precision** -- an entry matching "erase" but not "delete" would score as well as one matching "delete" directly.

The two-pass pattern gives you both: expanded vocabulary for candidate discovery, anchored scoring for relevance ranking.

### LLM Backend Priority

| Priority | Backend | Latency | Cost | Notes |
|----------|---------|---------|------|-------|
| 1 | Gemini Flash Lite | ~1s | Free tier | Cheapest, fastest |
| 2 | Gemini Flash | ~1s | Free tier | Smarter, still fast |
| 3 | Anthropic Haiku API | ~2s | Per-token | If ANTHROPIC_API_KEY set |
| 4 | `claude -p` subprocess | ~12s | Included | Always available, Node.js cold start overhead |

### Expansion Prompt

```
Query: "delete the deployment"
Generate 3-5 direct synonyms or alternate spellings for the key concepts.
Focus on what the user literally means, not tangentially related topics.
Return ONLY comma-separated keywords, nothing else.
```

### Disk Cache

Expansions are cached on disk (JSON file, max 100 entries, content hash keys). Identical queries across sessions skip the LLM call entirely. No TTL -- expansion synonyms do not change.

### Fail-Open

If no LLM backend is available (no API keys, no network), expansion is skipped silently. The system falls back to keyword + stem + substring matching, which handles most queries. Expansion is an enhancement, not a requirement.

---

## 9. Search Acceleration

The SQLite cache provides 12-56x speedup over full JSONL scanning with identical results. Verified across 10 test queries.

### What It Pre-Computes

For each entry in `memory_log.jsonl`, the cache stores:

| Column | Content |
|--------|---------|
| `tokens` | Space-separated set of 3+ char tokens from content + tags + type + section |
| `stems` | Space-separated set of stemmed tokens |
| `soundex_codes` | Space-separated Soundex codes (legacy, not used in scoring) |
| `tag_tokens` | Space-separated tokens from tags only |
| `content_hash` | MD5 prefix for O(1) dedup check |
| `raw_json` | Full JSON line for deserialization |

### Search Flow

```
1. Load cache into memory (one-time per process)
2. For each cached entry:
   a. Set intersection: query_tokens & entry_tokens
   b. Set intersection: query_stems & entry_stems
   c. If no overlap: skip (costs ~0.001ms)
   d. If overlap: call _score_entry() (costs ~0.1ms)
3. Sort by score, return top N
```

The set intersection pre-filter skips ~90% of entries before the expensive `_score_entry()` call. This is why the speedup is 12-56x rather than 2-3x.

### JSONL Scan as Authoritative Fallback

The cache is an acceleration layer. If the cache DB is missing, corrupt, or stale, `search_memory_log()` falls back to scanning the raw JSONL file. The results are identical -- the cache merely pre-computes what the JSONL scan computes on the fly.

This is a deliberate architectural choice. The JSONL file is the source of truth. The cache can be deleted and regenerated at any time with no data loss.

### Cache Invalidation

| Event | Invalidation Strategy |
|-------|----------------------|
| New entry appended | `append_entry()` -- adds to cache inline, no rebuild |
| Entry superseded | `mark_dirty()` -- creates `.cache-dirty` marker, triggers full rebuild on next search |
| Cache DB missing | Falls back to JSONL scan, triggers async rebuild |
| Stale detection | Checks `(mtime, fsize, .cache-dirty)` on load |
| Concurrent builds | `BEGIN EXCLUSIVE` in SQLite prevents double-rebuild |

### IDF Persistence

IDF weights are stored alongside the cache in the same SQLite database:

```sql
CREATE TABLE idf_weights (token TEXT PRIMARY KEY, weight REAL);
CREATE TABLE idf_meta (key TEXT PRIMARY KEY, value TEXT);
```

Refreshed when >50 new entries or >24 hours since last refresh. The 24-hour/50-entry window catches IDF drift from normal usage without recomputing after every append.

---

## 10. Session Bridging

Sessions in Claude Code are ephemeral. When a user starts a new session, Claude has no context about what happened in the previous one. Session bridging provides that context in ~250 characters.

### SESSION_CHECKPOINT (Fast Path)

At session end, the `stop_learning_extractor.py` hook writes a `SESSION_CHECKPOINT` entry:

```json
{
  "type": "SESSION_CHECKPOINT",
  "content": "Session abc12345: Fixed DNS propagation for cloudflare | Updated docker-compose.yml ports | Resolved CORS issue in API gateway",
  "session_id": "abc12345-...",
  "timestamp": "2026-03-29T15:30:00Z"
}
```

On next session start, `get_recent_session_summary()` finds the most recent checkpoint from a **different** session and injects it:

```
## Session Context
Session abc12345: Fixed DNS propagation for cloudflare | Updated docker-compose.yml ports | Resolved CORS issue in API gateway
```

### Tag Scanning (Fallback)

For entries written before checkpointing was added, or if checkpoint writing failed, the fallback reconstructs topics from raw entry tags:

1. Read last 4KB of `memory_log.jsonl`
2. Find entries from the most recent different session (compare `session_id[:8]`)
3. Extract unique tags, skip generic ones (`stop_hook`, `llm_extracted`, `auto_detected`, etc.)
4. Format: `"Last session topics: kubernetes, dns, cloudflare, docker"`

The fallback must stay permanently. Old sessions will never have checkpoints, and checkpoint writing is wrapped in try/except (it must never block session end).

---

## 11. Data Flow Diagram

Complete pipeline from user message to context injection:

```
USER TYPES: "How do I fix the kubernetes networking issue?"
    |
    v
[UserPromptSubmit hook fires]
    |
    v
[extract_intent()]
    |  Strip stop words: "fix", "kubernetes", "networking", "issue"
    |  IDF selection: if >10 tokens, keep top 10 by rarity
    |
    v
[Parallel search via ThreadPoolExecutor]
    |
    +--> [Keyword search: search_memory_log(intent)]
    |       |
    |       +--> [Cache path: set intersection pre-filter]
    |       |    skip ~90% of entries
    |       |    _score_entry() on remaining ~10%
    |       |
    |       +--> [JSONL fallback if no cache]
    |            scan all entries, _score_entry() each
    |
    +--> [Semantic search: expand_query(intent)]
            |
            +--> [LLM: "kubernetes, k8s, networking, CNI, pod network"]
            +--> [search_memory_log(expanded)]
            +--> [rescore_results(raw, original_intent)]
                  ^-- scoring anchored to original query
    |
    v
[Merge + dedup keyword and semantic results]
    |  keep highest score per content[:80] key
    |  cross-source dedup vs core_tagged results
    |
    v
[Compute exact_matches per result]
    |
    v
[format_tiered_context()]
    |
    +--> HOT (>= 0.3 AND >= 2 exact matches): full content
    +--> WARM (0.15-0.3 AND >= 2 exact matches): preview
    +--> COLD (< 0.15 or < 2 exact matches): dropped
    |
    v
[cluster_merge() on HOT items]
    |  group by Jaccard > 0.4 overlap
    |  corrections exempt from merging
    |
    v
[Budget check: total < 4000 chars]
    |  truncate if over budget
    |
    v
[Inject as additionalContext in hook output]
    |
    v
CLAUDE SEES:
    ## Session Context
    Last session topics: kubernetes, dns, docker

    ## Relevant Knowledge (from past sessions, advisory only)
    **[Networking]** CoreDNS configmap must be updated after...

    ## Past session learnings
    ### [WORKING_SOLUTION]
    The kubernetes networking fix requires updating CoreDNS configmap...
    ### [USER_CORRECTION]
    Never modify kube-system namespace resources without backup...

    (Memory is auto-captured in background -- no manual action needed.)
```

---

## Porting Guide

To implement Cortex for a different LLM tool (Cursor, Copilot, Windsurf, or any tool with a hook/plugin system):

1. **Hook into the prompt lifecycle.** You need a way to inject context before the LLM sees the user's message, and a way to observe tool outputs after execution. In Claude Code, these are `UserPromptSubmit` and `PostToolUse` hooks.

2. **Implement `_score_entry()`.** This is the core algorithm. Copy the multi-signal scoring pipeline. The function is ~120 lines of Python with zero dependencies.

3. **Use JSONL for storage.** Append-only, one JSON object per line. No schema migrations, no database setup, no versioning. The JSONL file is human-readable and trivially parseable.

4. **Add the three-way dedup.** Without it, the memory log grows with duplicates and the signal-to-noise ratio degrades. Jaccard similarity at 0.5/0.3 thresholds is the minimum viable dedup.

5. **Add the SQLite cache** only if search latency matters. At <1,000 entries, JSONL scanning is fast enough. Above that, the cache provides 12-56x speedup. The cache is optional -- it is an acceleration layer, not a feature.

6. **Add query expansion** only if you have an LLM API available. The system works well without it. Expansion helps most when the memory corpus uses different vocabulary than the user's current query.

7. **Tune thresholds for your corpus size.** The defaults (HOT=0.3, WARM=0.15, decay=0.03) were tuned for a 4,400+ entry corpus. Smaller corpora may benefit from lower thresholds; larger corpora may need higher ones.
