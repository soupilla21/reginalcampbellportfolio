# Leadership Agent

An autonomous, on-premises AI content pipeline that transforms RSS feed data
into executive-ready LinkedIn posts and long-form articles — entirely within
your organization's control boundary.

No third-party API calls. No per-token billing. No data leaving the machine.

---

## What It Does

The pipeline runs 9 agents in sequence, from raw feed ingestion to
publisher-ready output with governance checks at every stage.

```
Researcher → Gatekeeper → Scorer → Strategist → Writer →
Critic → Editor → Formatter → Publisher QC → Feedback Ingester
```

Each run:
1. Pulls articles from 25+ RSS sources
2. Filters them for brand relevance via an LLM gatekeeper
3. Scores and deduplicates (by URL, theme, and semantic similarity)
4. Selects the top 3 items and synthesizes a content angle
5. Writes a post and/or article in the author's voice
6. Runs a 7-axis critic loop — rewriting up to 3 times if needed
7. Edits, formats, and validates against brand rules
8. Logs human editorial changes to improve future runs automatically

See [`docs/pipeline-architecture.md`](docs/pipeline-architecture.md) for
the full architecture diagram and design rationale.

---

## Key Files

| File | Purpose |
|---|---|
| `agents/orchestrator.py` | State machine orchestrating the full 9-agent pipeline, including async LLM calls, timeout handling, deduplication, and run logging |
| `agents/feedback_ingester.py` | Continuous learning loop — diffs human edits against agent output, logs structured observations, and auto-promotes patterns into the brand guide |
| `config.yaml` | Central configuration for all models, thresholds, token limits, and deduplication parameters |
| `docs/pipeline-architecture.md` | Full pipeline diagram and design decision log |

---

## Architecture Decisions

**Local-first inference**
The system uses Ollama (qwen2.5:7b) for all LLM calls. This was a deliberate
constraint-driven decision: healthcare and federal deployment environments
prohibit sending operational data to external APIs. Local inference solves
for both the privacy requirement and the cost model simultaneously.

**Critic loop over one-shot generation**
Single-pass output rarely meets brand standards consistently at scale. The
Critic agent evaluates drafts across 7 axes and triggers a progressive rewrite
loop — passing rejection reasons *and* failed attempt history to the model so
it avoids repeating the same mistakes.

**Automated brand evolution**
The Feedback Ingester diffs every human-edited output against the agent
baseline. Patterns that appear 3 or more times are automatically promoted
into `brand/guide.yaml` — so the system improves from real editorial decisions
without manual prompt re-engineering.

---

## Stack

- **Runtime**: Ubuntu 24.04, Python 3.11
- **LLM inference**: Ollama (`qwen2.5:7b`, `mxbai-embed-large`)
- **Vector store**: ChromaDB (semantic deduplication)
- **Feed ingestion**: feedparser, httpx, BeautifulSoup4
- **Scheduling**: systemd timer
- **SaaS layer**: FastAPI + Supabase (PostgreSQL RLS, ES256/JWKS auth)

See [`requirements-core.txt`](requirements-core.txt) for core dependencies.

---

## Running the Pipeline

```bash
# Activate virtualenv
source venv/bin/activate

# Run full pipeline (post + article)
python -m agents.orchestrator --type both

# Run post only
python -m agents.orchestrator --type post

# Run with a specific theme
python -m agents.orchestrator --type post --theme "AI Governance"

# Ingest human edits after reviewing output
python -m agents.feedback_ingester --type post

# View feedback log summary + promotion candidates
python -m agents.feedback_ingester --summary

# Promote patterns into brand guide
python -m agents.feedback_ingester --promote
```

> **Note**: Run with `python -m agents.orchestrator` (not `python agents/orchestrator.py`)
> to correctly resolve relative imports from the project root.

---

## SaaS Layer

The pipeline is wrapped in a production-grade multi-tenant backend:

- **Auth**: Supabase ES256/JWKS JWT — cryptographically verifiable tokens,
  no round-trip to auth server
- **Data isolation**: PostgreSQL Row Level Security enforced at the database
  level — a misconfigured route cannot leak tenant data
- **RBAC**: `viewer`, `editor`, `tenant_admin` roles with separate policies
  per table
- **Audit logging**: Every pipeline run, stage result, and QC outcome is
  logged and exportable for compliance review

The SaaS layer separates what an enterprise product requires from what a
prototype delivers. Architecture details on the
[portfolio site](https://www.reginalcampbell.com).

---

## Author

**Reginal Campbell, PMP**
Enterprise AI & Data Strategy · [reginalcampbell.com](https://www.reginalcampbell.com)
