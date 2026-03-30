# Cortex vs. Alternatives

An honest comparison of Cortex against every memory system we researched during development. Each system makes different tradeoffs. Cortex is not universally better -- it occupies a specific niche.

---

## Cortex vs. Mem0

[Mem0](https://github.com/mem0ai/mem0) is the most popular open-source LLM memory library. It stores memories as embeddings in a vector database and uses an LLM to decide what to remember.

**Where Mem0 wins:**
- Semantic search via embeddings is fundamentally richer than keyword matching. "The deployment broke" can match "production outage" even with zero token overlap.
- Mem0's LLM-based ADD/UPDATE/DELETE/NOOP decision is more nuanced than Jaccard thresholds. It can detect semantic duplicates that share no vocabulary.
- Graph memory (Mem0 v2) can represent entity relationships that flat JSONL cannot.

**Where Cortex wins:**
- **Zero per-operation cost.** Every Mem0 `add()` and `search()` calls an LLM (for extraction/classification) and an embedding model (for vectorization). At scale, this adds up. Cortex's scoring pipeline is pure arithmetic -- no API calls for search.
- **No infrastructure.** Mem0 requires an OpenAI API key (or compatible) and a vector database (Qdrant, Chroma, Pinecone, etc.). Cortex requires Python and nothing else.
- **Deterministic scoring.** Cortex's `_score_entry()` is a pure function. Same inputs always produce the same score. Mem0's LLM-based decisions are probabilistic -- the same memory might be classified differently on retry.
- **Transparency.** You can read `memory_log.jsonl` in a text editor and understand exactly what Cortex remembers. Vector embeddings are opaque.

**Cortex borrowed from Mem0:** The three-way dedup decision (SKIP/SUPERSEDE/APPEND) was directly inspired by Mem0's ADD/UPDATE/DELETE/NOOP pattern, implemented with Jaccard similarity instead of LLM calls.

---

## Cortex vs. MemGPT / Letta

[Letta](https://github.com/letta-ai/letta) (formerly MemGPT) is a full agent platform where the LLM manages its own memory through tool calls. The LLM decides what to store, what to retrieve, and what to forget.

**Where Letta wins:**
- The LLM has full autonomy over memory management. It can make context-sensitive decisions about what matters, perform complex reasoning about what to forget, and organize memories into hierarchical structures.
- Letta supports conversation memory, working memory, and archival memory as distinct tiers with different retrieval strategies.
- Built-in agent orchestration, tool use, and multi-step reasoning.

**Where Cortex wins:**
- **Cortex is a plugin, not a platform.** Letta requires Docker, PostgreSQL, and a running server. It replaces your existing tool chain. Cortex adds memory to Claude Code (or any LLM tool) without replacing anything.
- **No LLM judgment dependency.** Letta's memory quality depends entirely on the model's judgment about what to store and retrieve. If the model decides something is not worth remembering, it is lost. Cortex captures memories via deterministic hooks (errors, corrections, test results) -- the model's opinion is not consulted for capture.
- **Predictable context cost.** Letta's context usage depends on the model's retrieval decisions. Cortex guarantees a hard cap of 4,000 characters (~1,000 tokens) per prompt.
- **Startup time.** Cortex adds ~50-200ms to each prompt (search + scoring). Letta adds server round-trip latency plus LLM inference for every memory operation.

---

## Cortex vs. CLAUDE.md

CLAUDE.md is Claude Code's built-in memory mechanism. It is a markdown file that gets loaded into every prompt automatically.

**Where CLAUDE.md wins:**
- Zero latency. The file is loaded directly, no search or scoring needed.
- Complete control. You write exactly what Claude sees. No algorithm decides what is relevant.
- No dependencies. No hooks, no Python, no configuration.
- Works for project-specific context that should always be loaded (coding standards, architecture decisions, team conventions).

**Where Cortex wins:**
- **Search vs. load-all.** CLAUDE.md loads its entire contents into every prompt, whether relevant or not. A 200-line CLAUDE.md costs 5,000-15,000+ tokens per prompt. Cortex searches 6,000+ entries and injects only what matches the current query, at 250-750 tokens.
- **Automatic capture.** CLAUDE.md requires manual maintenance. You write the entries, you organize them, you prune stale content. Cortex captures memories automatically from errors, corrections, test results, and session-end extraction.
- **Deduplication.** CLAUDE.md has no dedup. If you add the same fact twice, it is injected twice. Cortex's three-way dedup prevents this.
- **Relevance ranking.** CLAUDE.md has no concept of relevance. Line 1 and line 200 are equally important. Cortex scores every entry against the current context and only surfaces what matters.
- **Self-correction.** When you correct Claude, Cortex gives that correction a 1.5x priority boost. Old, incorrect memories decay exponentially. CLAUDE.md has no temporal signal -- a wrong entry from three months ago has the same weight as a correction from today.
- **Scale.** CLAUDE.md has a practical cap around 200 lines before the context cost becomes prohibitive. Cortex scales to thousands of entries because only relevant ones are injected.

**When to use both:** CLAUDE.md for project-level constants (coding standards, architecture rules, team conventions). Cortex for session-learned knowledge (error resolutions, discovered patterns, user corrections).

---

## Cortex vs. Zep

[Zep](https://github.com/getzep/zep) is the most sophisticated memory system we researched. It combines cosine similarity, BM25 keyword search, and knowledge graph traversal (via Neo4j) for retrieval.

**Where Zep wins:**
- **Graph-based entity resolution.** Zep can track entities across conversations and resolve relationships ("Alice works at Acme" + "Acme's CTO is Bob" = knowledge about Alice's colleague). Cortex has no entity graph.
- **Conflict detection.** Zep can identify contradictions between stored facts and flag them. Cortex relies on recency decay and supersede to implicitly resolve conflicts.
- **Multi-signal retrieval.** Zep combines three retrieval signals (cosine, BM25, graph) and uses reciprocal rank fusion. This is more robust than any single retrieval method.
- **Enterprise features.** User management, API keys, fact rating, structured data extraction.

**Where Cortex wins:**
- **Zero infrastructure.** Zep requires Neo4j (graph database) and an LLM API for entity extraction. That is enterprise infrastructure for what should be a developer tool. Cortex is `pip install` and done.
- **Context cost.** Zep's benchmark shows ~1,600 tokens per injection. Cortex averages 250-750 tokens. For tools with 200K context windows this matters less, but for tools with smaller windows the difference is significant.
- **Offline operation.** Cortex works without network access (expansion disabled, everything else works). Zep requires API connectivity for both storage and retrieval.
- **Simplicity.** Zep's architecture has more moving parts (graph DB, embedding model, BM25 index, LLM extractor). Each part can fail independently. Cortex has one JSONL file, one SQLite cache, and one scoring function.

**If you need entity graphs and conflict detection, use Zep.** It is genuinely more capable for knowledge management at scale. Cortex trades that sophistication for zero-dependency simplicity.

---

## Cortex vs. claude-mem

[claude-mem](https://github.com/nicobailey/claude-mem) is the closest architectural cousin to Cortex. It also uses Claude Code hooks for automatic memory capture and retrieval.

**Where claude-mem wins:**
- Progressive context injection. Instead of fixed HOT/WARM/COLD tiers, claude-mem injects memories progressively as context budget allows, from most to least relevant. This can be more efficient for borderline cases.
- Hybrid semantic + keyword search via ChromaDB embeddings and keyword matching. Semantic search is fundamentally more capable than keyword-only matching.

**Where Cortex wins:**
- **Single runtime.** claude-mem requires three runtimes: Bun (for hooks), uv (for Python), and Chroma (for vector storage). Cortex requires Python and nothing else.
- **No vector database.** ChromaDB is lightweight compared to Qdrant or Pinecone, but it is still an external dependency that must be installed, updated, and maintained.
- **Multi-signal scoring.** Cortex's scoring pipeline (IDF weighting, tag boost, type boost, recency decay, coverage factor) provides comparable precision to hybrid semantic + keyword search, at zero per-query cost.
- **Self-correcting via priority boost.** Cortex specifically boosts corrections and preferences. claude-mem treats all memories equally in scoring.

---

## Full Comparison Table

| Feature | Cortex | Mem0 | Letta | CLAUDE.md | Zep | claude-mem |
|---------|--------|------|-------|-----------|-----|------------|
| **Context cost** | 250-750 tokens | 200-800 + LLM/op | ~2K base + inference | 5K-15K+ (all loaded) | ~1,600 (benchmark) | 50-1,000 (progressive) |
| **Dependencies** | None (stdlib) | OpenAI + vector DB | Docker + PostgreSQL | None | Neo4j + LLM API | Bun + uv + Chroma |
| **Auto-capture** | Yes (hooks) | SDK call required | Agent tool calls | Manual / semi-auto | Automatic | Yes (hooks) |
| **Search method** | IDF + stems + tags + decay | Embeddings | Agent decides | None (loads all) | Cosine + BM25 + graph | Hybrid semantic + keyword |
| **Self-correcting** | Yes (1.5x boost + decay) | No | Model-dependent | No | Conflict detection | No |
| **Deduplication** | 3-way (hash + Jaccard) | LLM-based | Model-dependent | None | Entity merging | Basic |
| **Offline capable** | Yes (expansion disabled) | No | No | Yes | No | No |
| **Setup** | `pip install` | pip + API keys + vector DB | Docker Compose | Create a file | Docker + Neo4j + API keys | npm + pip + chroma |
| **Transparency** | JSONL (human-readable) | Embeddings (opaque) | Agent-managed | Markdown (readable) | Graph + embeddings | ChromaDB |
| **Entity resolution** | No (flat entries) | No | No | No | Yes (knowledge graph) | No |
| **Scale tested** | 6,000+ entries | Production scale | Production scale | ~200 lines practical | Enterprise scale | Hundreds of entries |

---

## When to Choose What

- **You want zero dependencies and maximum simplicity:** Cortex
- **You need semantic search and have API budget:** Mem0
- **You want the LLM to manage its own memory:** Letta
- **You have <200 lines of context and want full control:** CLAUDE.md
- **You need entity graphs and enterprise features:** Zep
- **You are already in the Bun/uv ecosystem:** claude-mem
- **You want Cortex's approach but with embeddings:** Fork Cortex, replace `_score_entry()` with an embedding call. The rest of the architecture (hooks, dedup, tiering, caching) still applies.
