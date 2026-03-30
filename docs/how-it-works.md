# How Cortex Works

A step-by-step walkthrough of the complete pipeline, using a concrete example.

---

## The Scenario

You open a new Claude Code session and type:

```
How do I fix the kubernetes networking issue?
```

Your memory log has 4,400+ entries from months of work across multiple projects. Somewhere in there is an entry from two weeks ago:

```json
{
  "type": "WORKING_SOLUTION",
  "content": "The kubernetes networking fix requires updating the CoreDNS configmap. Run kubectl edit configmap coredns -n kube-system and add the forward block for the custom domain. This resolved the pod-to-service DNS resolution failure.",
  "tags": ["kubernetes", "networking", "dns", "coredns"],
  "timestamp": "2026-03-16T14:30:00Z",
  "session_id": "abc12345-def6-7890-ghij-klmnopqrstuv"
}
```

Here is exactly what happens, step by step.

---

## Step 1: Hook Fires

The `UserPromptSubmit` hook (`memory_awareness_tiered.py`) fires before Claude sees your message. It receives the raw prompt as JSON on stdin:

```json
{
  "hook": "UserPromptSubmit",
  "prompt": "How do I fix the kubernetes networking issue?",
  "session_id": "xyz98765-..."
}
```

---

## Step 2: Intent Extraction

`extract_intent()` strips the prompt down to meaningful keywords.

**Input:** `"How do I fix the kubernetes networking issue?"`

**Process:**
1. Strip quoted text and code blocks (prevents feedback loops from previously-injected memory)
2. Remove punctuation, lowercase
3. Split into words: `["how", "do", "i", "fix", "the", "kubernetes", "networking", "issue"]`
4. Remove stop words (`how`, `do`, `the` are stopped; `fix` and `issue` are also stopped as common action words)
5. Keep words with 3+ characters that are not in the stop list

**Output:** `"kubernetes networking"`

Two tokens. Clean signal.

---

## Step 3: IDF Token Selection

If intent extraction produced more than 10 tokens (e.g., from a long prompt), the top 10 by IDF weight are kept. IDF gives higher weight to rarer tokens.

In this case, we only have 2 tokens, so no selection is needed. But if we had 15 tokens, the process would be:

```python
# For each token, look up its IDF weight
# idf(token) = log(1 + N / df)
# where N = 4400 total entries, df = entries containing this token

# "kubernetes" appears in 12 entries: idf = log(1 + 4400/12) = 5.91
# "config" appears in 800 entries:   idf = log(1 + 4400/800) = 1.79
# "error" appears in 1200 entries:   idf = log(1 + 4400/1200) = 1.50

# Keep top 10 by IDF -- rare, specific tokens are kept; common ones are dropped
```

---

## Step 4: Parallel Search

Two searches run simultaneously via `ThreadPoolExecutor`:

### Branch A: Keyword Search

`search_memory_log("kubernetes networking")` runs against the memory log.

**Cache path (fast):** The SQLite cache has pre-computed token sets for all 4,400 entries. For each cached entry:

```
query_tokens  = {"kubernetes", "networking"}
query_stems   = {"kubernetes", "network"}  (stemmer strips "-ing")

entry_tokens  = {"kubernetes", "networking", "fix", "requires", "updating", ...}
entry_stems   = {"kubernetes", "network", "fix", "requir", "updat", ...}

Set intersection: query_tokens & entry_tokens = {"kubernetes", "networking"}
Overlap found --> pass to _score_entry()
```

For ~90% of entries, the set intersection is empty. Those are skipped without calling `_score_entry()`. This is the 12-56x speedup.

### Branch B: Semantic Search

`expand_query("kubernetes networking")` calls the LLM for synonyms.

**Prompt sent to Gemini Flash:**
```
Query: "kubernetes networking"
Generate 3-5 direct synonyms or alternate spellings for the key concepts.
Focus on what the user literally means, not tangentially related topics.
Return ONLY comma-separated keywords, nothing else.
```

**Response (~1 second):** `"k8s, pod network, CNI, cluster networking, kube-dns"`

**Expanded query:** `"kubernetes networking k8s pod network CNI cluster networking kube-dns"`

This expanded query runs through `search_memory_log()` -- finding entries that mention "k8s" or "kube-dns" but not "kubernetes" literally.

**Critical: rescoring.** The results from the expanded search are then re-scored against the **original** query `"kubernetes networking"`:

```python
rescored = rescore_results(expanded_results, "kubernetes networking")
```

This means an entry matching "CNI" but not "kubernetes" will score lower than one matching "kubernetes" directly. Expansion improves recall without corrupting precision.

---

## Step 5: Scoring Walkthrough

Let us score our target entry step by step.

**Entry:**
```json
{
  "content": "The kubernetes networking fix requires updating the CoreDNS configmap...",
  "tags": ["kubernetes", "networking", "dns", "coredns"],
  "type": "WORKING_SOLUTION",
  "timestamp": "2026-03-16T14:30:00Z"
}
```

**Query tokens:** `{"kubernetes", "networking"}`
**Query stems:** `{"kubernetes", "network"}`

### 5a. Token Extraction

Build the searchable text from all entry fields:

```
content:  "the kubernetes networking fix requires updating the coredns configmap..."
tags:     "kubernetes networking dns coredns"
type:     "working_solution"
section:  ""

searchable = content + tags + type + section
entry_tokens = {"kubernetes", "networking", "fix", "requires", "updating",
                "coredns", "configmap", "resolved", "pod", "service",
                "dns", "resolution", "failure", "working", "solution", ...}
```

### 5b. Exact Match Check

```
exact_matches = query_tokens & entry_tokens
              = {"kubernetes", "networking"} & {..."kubernetes", "networking"...}
              = {"kubernetes", "networking"}
```

Both query tokens match exactly. This is the best case.

### 5c. IDF-Weighted Match Score

```python
# IDF weights (from corpus statistics)
idf["kubernetes"] = 5.91   # rare -- only 12 entries
idf["networking"] = 4.20   # uncommon -- 30 entries

# Numerator: IDF weight of exact matches (1.0x tier)
exact_w = idf["kubernetes"] + idf["networking"]
        = 5.91 + 4.20
        = 10.11

# No stem-only or substring matches (both tokens matched exactly)
stem_w = 0
sub_w  = 0

# Denominator: total IDF weight of query tokens
total_w = idf["kubernetes"] + idf["networking"]
        = 5.91 + 4.20
        = 10.11

match_score = (10.11 + 0 + 0) / 10.11 = 1.0
```

A perfect match score of 1.0 -- both query tokens matched exactly, and both are rare.

### 5d. Multiplier Eligibility

```python
multiplier_eligible = (
    match_score >= 0.3                                     # True (1.0 >= 0.3)
    or (match_score >= 0.15 and len(exact_matches) > 0)    # Also true
) and len(query_tokens) >= 3                               # False (only 2 tokens)
```

Wait -- `len(query_tokens) >= 3` is False. With only 2 query tokens, multiplier boosts are disabled. This prevents very short queries from getting inflated scores via tag/type multipliers.

```
tag_boost = 1.0      # disabled (not multiplier-eligible)
type_boost = 1.0     # disabled (not multiplier-eligible)
```

### 5e. Priority Boost

```python
priority = entry.get("priority", 0)  # 0 for memory_log entries
priority_boost = 1.0                 # no boost
```

### 5f. Recency Decay

```python
age_days = (now - entry_timestamp) / 86400
         = 14 days  # entry is from 2 weeks ago

decay = e^(-0.03 * 14)
      = e^(-0.42)
      = 0.657
```

The entry retains 65.7% of its score after two weeks.

### 5g. Coverage Factor

```python
# How much of the entry is covered by query matches?
exact_in_entry = len(exact_matches & entry_tokens)  # 2
entry_size = len(entry_tokens)                       # ~25 tokens

entry_coverage = 2 / sqrt(25) = 2 / 5 = 0.4
coverage_factor = max(0.15, min(1.0, 0.4 * 3))
                = max(0.15, min(1.0, 1.2))
                = 1.0  # capped at 1.0
```

Good coverage -- the entry is not so large that our 2-token match feels superficial.

### 5h. Final Score

```python
score = match_score * tag_boost * type_boost * priority_boost * decay * coverage_factor
      = 1.0 * 1.0 * 1.0 * 1.0 * 0.657 * 1.0
      = 0.657
```

Score: **0.657**. Well above the HOT threshold of 0.3.

> Note: If the query had 3+ tokens (e.g., "fix kubernetes networking issue" after stop-word removal yielded 3 keywords), multiplier eligibility would be True. The tag boost would kick in (2.0x) because "kubernetes" and "networking" appear in both the query and the entry's tags. The score would be `1.0 * 2.0 * 1.0 * 1.0 * 0.657 * 1.0 = 1.314`. Boosts can push scores above 1.0.

---

## Step 6: Result Merging

The keyword search and semantic search both return their top 5 results. These are merged:

```python
# Merge by content[:80] key -- keep highest score per unique entry
seen = {}
for r in keyword_results + semantic_results:
    key = r["entry"]["content"][:80]
    if key not in seen or r["score"] > seen[key]["score"]:
        seen[key] = r

# Cross-source dedup: skip entries already found in core_tagged search
# (prevents the same fact appearing twice from two data sources)

# Sort by score, keep top 5
results = sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:5]
```

---

## Step 7: Tiering

Each result is assigned a tier based on its score:

```
Result 1: score=0.657, type=WORKING_SOLUTION  --> HOT  (>= 0.3)
Result 2: score=0.42,  type=CONFIG_INSIGHT     --> HOT  (>= 0.3)
Result 3: score=0.23,  type=DEBUGGING_INSIGHT  --> WARM (0.15-0.3)
Result 4: score=0.11,  type=CODEBASE_PATTERN   --> COLD (< 0.15, dropped)
Result 5: score=0.08,  type=WORKING_SOLUTION   --> COLD (< 0.15, dropped)
```

**Exact match gate:** Before accepting a HOT or WARM result, check that it has at least 2 exact token matches against the original intent. This prevents entries that scored high via stem/substring matching alone (potential cross-domain false positives) from being injected.

```python
# For each result, count exact token overlap with intent
intent_tokens = {"kubernetes", "networking"}
entry_tokens  = set(tokenize(entry_content + entry_tags))

exact_matches = len(intent_tokens & entry_tokens)

if exact_matches < 2:
    continue  # skip this result entirely
```

---

## Step 8: Cluster-Merge

If multiple HOT results are similar, they are clustered to save context budget.

```python
# Result 1 tokens: {"kubernetes", "networking", "coredns", "configmap", "dns", ...}
# Result 2 tokens: {"kubernetes", "networking", "coredns", "forward", "dns", ...}

jaccard = |intersection| / |union|
        = 8 / 15
        = 0.53  # > 0.4 threshold --> cluster together
```

Result 2 is merged into Result 1's cluster. The output becomes:

```markdown
### [WORKING_SOLUTION]
The kubernetes networking fix requires updating the CoreDNS configmap.
Run kubectl edit configmap coredns -n kube-system and add the forward
block for the custom domain. This resolved the pod-to-service DNS
resolution failure. (+1 related)
```

The `(+1 related)` suffix tells Claude there is additional information available without consuming more context budget.

**Corrections are never merged.** If Result 1 were a `USER_CORRECTION` like "Never modify kube-system namespace resources without backup", it would be shown separately regardless of Jaccard overlap. Every word in a correction matters.

---

## Step 9: Context Assembly and Injection

All context parts are assembled with a total budget of 4,000 characters:

```markdown
## Session Context
Last session topics: kubernetes, dns, cloudflare, docker

## Relevant Knowledge (from past sessions, advisory only)

**[Networking]** CoreDNS requires custom forward blocks for internal
domains when running in EKS clusters.

## Past session learnings

Relevant context from previous sessions:

### [WORKING_SOLUTION]
The kubernetes networking fix requires updating the CoreDNS configmap.
Run kubectl edit configmap coredns -n kube-system and add the forward
block for the custom domain. This resolved the pod-to-service DNS
resolution failure. (+1 related)

## POSSIBLE MATCHES (may be relevant)
- [DEBUGGING_INSIGHT] DNS resolution failures in kubernetes often caused by...

(Memory is auto-captured in background -- no manual action needed.)
```

### Budget Check

```python
total_chars = len(context)  # ~850 characters
budget = 4000

# Under budget. No truncation needed.
# If over budget, entries are dropped from bottom (lowest score first)
```

This context is returned as `additionalContext` in the hook's JSON output. Claude Code prepends it to your message before sending to the model.

---

## Step 10: What Claude Sees

Claude receives your prompt with the injected context. It now knows:

1. What you worked on last session (session bridging)
2. A specific working solution for kubernetes networking from 2 weeks ago
3. A related debugging insight about DNS resolution
4. That memory capture is automatic (so it does not try to manually manage memory files)

Claude can now reference the CoreDNS fix directly instead of giving generic kubernetes networking advice. If it makes a mistake, your correction will be captured by the `correction_detector.py` hook and stored with a 1.5x type boost -- ensuring it ranks higher than the general solution next time.

---

## The Write Side

While the read side runs on every prompt, the write side captures new memories during the session:

| Trigger | Hook | What It Captures |
|---------|------|-----------------|
| Tool returns an error | `auto_learning.py` (PostToolUse) | Error-resolution pairs when the next tool succeeds |
| User corrects Claude | `correction_detector.py` (UserPromptSubmit) | The correction, tagged as `USER_CORRECTION` with 1.5x boost |
| User states a preference | `correction_detector.py` (UserPromptSubmit) | The preference, tagged as `USER_PREFERENCE` |
| Test passes after failing | `auto_learning.py` (PostToolUse) | The fix that made the test pass |
| Session ends | `stop_learning_extractor.py` (Stop) | LLM-extracted learnings + session checkpoint |

Each captured memory goes through the three-way dedup before storage:

```
New memory arrives
    |
    v
[Content hash check] -- O(1) against SQLite
    |
    match --> SKIP (exact duplicate)
    |
    no match
    |
    v
[Jaccard scan of last 50 entries]
    |
    J > 0.5  --> SKIP (near-duplicate)
    |
    0.3-0.5 AND longer --> SUPERSEDE (replace old entry)
    |
    < 0.3 --> APPEND (new topic, add to log)
```

The loop closes: better memories produce better Claude responses, which produce better memories.

---

## Timing

Typical latency for the full read pipeline:

| Step | With Cache | Without Cache |
|------|-----------|---------------|
| Intent extraction | <1ms | <1ms |
| Keyword search | 5-20ms | 200-800ms |
| Semantic expansion | 1-2s (Gemini) or 10-12s (Haiku) | Same |
| Scoring + tiering | <1ms | <1ms |
| Cluster-merge | <1ms | <1ms |
| **Total (no expansion)** | **5-25ms** | **200-800ms** |
| **Total (with expansion)** | **1-2s** | **1-2s** |

Query expansion dominates latency when enabled. Without it, the entire pipeline completes in under 25ms with a warm cache. The keyword search and semantic search run in parallel via `ThreadPoolExecutor`, so expansion latency does not add to keyword search latency -- they overlap.
