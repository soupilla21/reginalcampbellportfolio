# Pipeline Architecture

The Leadership Agent runs a 9-stage sequential pipeline, executing entirely
on-premises with no third-party API calls. Each stage is a discrete Python
module with typed inputs and outputs. The orchestrator manages transitions,
retries, and cross-agent context injection.

```
PHASE 1 — INTELLIGENCE GATHERING
┌──────────────┐    ┌─────────────┐    ┌────────────┐
│  Researcher  │───▶│  Gatekeeper │───▶│   Scorer   │
│  (RSS Fetch) │    │  (LLM Filter│    │ (Semantic  │
│  25+ feeds   │    │   by brand  │    │  Ranking)  │
└──────────────┘    │   fitness)  │    └────────────┘
                    └─────────────┘
                           │
                    Deduplication layer
                    (item / theme / similarity)
                           │
PHASE 2 — STRATEGY
                    ┌──────────────┐
                    │  Strategist  │
                    │  (selects    │
                    │   top 3,     │
                    │   picks angle│
                    │   + format)  │
                    └──────────────┘
                           │
PHASE 3/4 — WRITING + CRITIC LOOP
                    ┌──────────────┐    ┌────────────┐
                    │    Writer    │───▶│   Critic   │
                    │  (post or    │    │  (7-axis   │◀─┐
                    │   article)   │    │   scoring) │  │
                    └──────────────┘    └────────────┘  │
                                              │ reject   │
                                        ┌────▼──────┐   │
                                        │  Rewrite  │───┘
                                        │  (up to   │  max 3x
                                        │  3 loops) │
                                        └───────────┘
                                              │ pass
PHASE 5–7 — POLISH + COMPLIANCE
                    ┌──────────────┐    ┌────────────┐    ┌──────────────┐
                    │    Editor    │───▶│  Formatter │───▶│ Publisher QC │
                    │  (clarity,   │    │ (LinkedIn  │    │ (word count, │
                    │   hook,      │    │  structure)│    │  banned      │
                    │   brand      │    │            │    │  phrases,    │
                    │   polish)    │    │            │    │  CTA check)  │
                    └──────────────┘    └────────────┘    └──────────────┘
                                                                  │
CONTINUOUS LEARNING LOOP                                          │
                    ┌──────────────────────────────────────────┐  │
                    │  Feedback Ingester                        │◀─┘
                    │  - Diffs human edits vs. agent output     │
                    │  - Logs structured observations           │
                    │  - Promotes patterns (≥3 occurrences)     │
                    │    into brand/guide.yaml automatically    │
                    └──────────────────────────────────────────┘
```

## Key Design Decisions

**Why local LLMs (Ollama)?**
Healthcare and federal deployment contexts prohibit sending operational data
to third-party APIs. Local inference via Ollama on commodity hardware delivers
strong output quality at near-zero marginal cost, and keeps every byte within
the organization's control boundary.

**Why a critic loop instead of one-shot generation?**
Single-pass LLM output rarely meets production brand standards consistently.
The critic evaluates on 7 axes (hook strength, specificity, voice match,
argument coherence, novelty, CTA quality, brand compliance) and triggers a
progressive rewrite loop — each retry passes the rejection reasons *and* the
history of failed attempts so the model avoids repeating the same mistakes.

**Why a Feedback Ingester?**
Manual prompt engineering after every bad output is expensive and inconsistent.
The Feedback Ingester performs a semantic diff between the agent's output and
the human's final edited version. Patterns appearing 3+ times are automatically
promoted into brand/guide.yaml, so the system improves from real usage without
requiring a human to reverse-engineer what changed.

## Stack

| Component | Technology |
|---|---|
| OS / runtime | Ubuntu 24.04, Python 3.11 |
| LLM inference | Ollama (qwen2.5:7b) |
| Vector store | ChromaDB (mxbai-embed-large embeddings) |
| Feed ingestion | feedparser + httpx |
| Config | PyYAML (config.yaml) |
| Scheduling | systemd timer + cron |
| SaaS layer | FastAPI + Supabase (PostgreSQL RLS, ES256/JWKS auth) |
