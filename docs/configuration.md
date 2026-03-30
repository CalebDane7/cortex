# Cortex Configuration

All configuration is done through environment variables. Cortex ships with sensible defaults that work well for most users — you only need to change these if you have a specific reason.

## Environment Variables

### CORTEX_MEMORY_DIR

**Default:** `~/.cortex/`

The root directory where Cortex stores all memory data: logs, tagged memories, aliases, cache, and the generated MEMORY.md file.

**Why this default:** Follows the XDG-adjacent `~/.toolname/` convention used by Claude Code itself (`~/.claude/`). Keeps memory data separate from hook configuration.

**When to change:** If you want memory data on a different filesystem (e.g., an encrypted volume), or if you run multiple isolated Cortex instances for different projects.

```bash
export CORTEX_MEMORY_DIR="/path/to/custom/memory"
```

---

### GEMINI_API_KEY

**Default:** Not set (query expansion disabled)

API key for Google Gemini. When set, Cortex uses Gemini to expand search queries — turning a prompt like "fix the auth bug" into expanded terms like "authentication, OAuth, JWT, login, session." This improves recall when your memory entries use different terminology than your current prompt.

**Why optional:** Cortex works well without query expansion. The tag-based and keyword matching catches most relevant memories. Expansion helps with terminology gaps in larger memory stores (500+ entries).

**When to enable:** When you notice Cortex missing relevant memories because the prompt wording doesn't match how the memory was stored. For example, you say "database" but the memory says "PostgreSQL."

```bash
export GEMINI_API_KEY="your-api-key-here"
```

---

### ANTHROPIC_API_KEY

**Default:** Not set

API key for Anthropic's Claude API. Alternative LLM backend for query expansion (used when GEMINI_API_KEY is not set but this is). Also used by the stop_learning_extractor hook to summarize session learnings at the end of a conversation.

**Why optional:** The stop_learning_extractor can operate in a lightweight mode without an API key, extracting learnings through pattern matching rather than LLM summarization.

**When to enable:** When you want higher-quality session summaries and query expansion, especially if you already have an Anthropic API key from other usage.

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

---

### CORTEX_DECAY_RATE

**Default:** `0.03`

Controls how quickly old memories lose relevance in scoring. The score multiplier for an entry is `exp(-rate * age_in_days)`.

At the default rate of 0.03:
- 7-day-old entry retains **81%** of its score
- 30-day-old entry retains **41%** of its score
- 90-day-old entry retains **7%** of its score
- 180-day-old entry retains **0.5%** of its score

**Why 0.03:** Balances recency bias against long-term memory. Fast enough that yesterday's debugging session ranks above last month's, slow enough that important corrections (like "never force-push to main") survive for weeks before needing reinforcement.

**When to change:**
- **Increase (e.g., 0.05):** If you work on rapidly changing projects where week-old context is usually stale.
- **Decrease (e.g., 0.01):** If you work on stable, long-running projects where months-old decisions are still relevant.

```bash
export CORTEX_DECAY_RATE="0.01"
```

---

### CORTEX_HOT_THRESHOLD

**Default:** `0.3`

Minimum relevance score for a memory to be classified as "HOT" and receive full injection into the prompt. HOT memories appear with full content and are given the most prominent placement.

Scores are 0.0-1.0, combining keyword match strength, tag overlap, recency (decay), and priority weight.

**Why 0.3:** Empirically tuned to produce 1-4 HOT results per prompt on a memory store of ~500 entries. Below 0.3, too many marginally relevant memories get full injection, wasting context budget on noise.

**When to change:**
- **Increase (e.g., 0.4):** If you have a very large memory store (2000+) and see too many injected memories per prompt.
- **Decrease (e.g., 0.2):** If Cortex rarely surfaces memories and you want more aggressive recall.

```bash
export CORTEX_HOT_THRESHOLD="0.4"
```

---

### CORTEX_WARM_THRESHOLD

**Default:** `0.15`

Minimum relevance score for a memory to be classified as "WARM." WARM memories appear as condensed one-line summaries below the HOT results — enough to remind Claude they exist without consuming much context.

**Why 0.15:** Set at roughly half the HOT threshold. Catches memories that are partially relevant — right topic area but not a direct hit. Below 0.15 on stores with 2000+ entries, you start seeing irrelevant matches.

**When to change:**
- **Increase (e.g., 0.2):** If WARM results are mostly noise and not helpful.
- **Decrease (e.g., 0.1):** If you want broader recall and don't mind occasional false positives in the condensed section.

```bash
export CORTEX_WARM_THRESHOLD="0.2"
```

---

### CORTEX_MAX_INJECTION_CHARS

**Default:** `4000`

Hard cap on the total number of characters injected into Claude's context by the memory_awareness hook. This includes both HOT (full content) and WARM (condensed) memories, plus section headers.

**Why 4000:** Claude's context window is 200K tokens. 4000 characters is roughly 1000 tokens, or 0.5% of the context. This is enough for approximately 3 full HOT memories plus 2-3 WARM one-liners. Going higher produces diminishing returns — Claude is less likely to act on memory #8 than memory #1, and the extra context adds latency.

**When to change:**
- **Increase (e.g., 6000):** If you work on complex domains where more context is consistently useful (e.g., a project with many interrelated gotchas).
- **Decrease (e.g., 2000):** If you want minimal overhead per prompt, or if you notice the injected memories are distracting Claude from the actual task.

```bash
export CORTEX_MAX_INJECTION_CHARS="6000"
```

---

### MEMORY_EXPANSION_DISABLED

**Default:** Not set (expansion enabled when an API key is available)

Set to `1` to completely disable LLM-based query expansion, even when GEMINI_API_KEY or ANTHROPIC_API_KEY is set. Cortex will fall back to keyword-only matching.

**Why this exists:** Query expansion adds 200-500ms of latency per prompt (one LLM API call). If you have an API key set for other purposes but don't want Cortex using it, or if you're on a metered connection and want to minimize API calls, this lets you disable just the expansion.

**When to set:**
- You have API keys configured for other tools but don't want Cortex making API calls.
- You're debugging Cortex behavior and want to isolate keyword matching from expansion.
- You're in an environment with high API latency and the 10-second hook timeout is tight.

```bash
export MEMORY_EXPANSION_DISABLED=1
```

---

## File Locations

All paths are relative to `CORTEX_MEMORY_DIR` (default `~/.cortex/`):

| File | Purpose |
|------|---------|
| `memory_log.jsonl` | Append-only log of all learned memories (runtime) |
| `core_tagged.jsonl` | Curated high-priority memories with tags and sections |
| `aliases.json` | Term aliases for query expansion (e.g., "k8s" -> "kubernetes") |
| `MEMORY.md` | Auto-generated markdown summary (consumed by Claude Code) |
| `logs/memory-pipeline.log` | Debug log for hook execution |
| `memory_cache.db` | SQLite cache for scored results |
| `.cache-dirty` | Marker file that triggers cache rebuild |
| `.query-expansion-cache.json` | Cached query expansions to avoid repeated API calls |
| `archive/` | Rotated old memory_log.jsonl files |
