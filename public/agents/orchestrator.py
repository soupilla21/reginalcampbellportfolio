#!/usr/bin/env python3
import argparse
import re
import asyncio
import sys
import os
import json
import yaml
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from agents.researcher import fetch_all_feeds
from agents.gatekeeper import run_gatekeeper_async
from agents.scorer import semantic_scorer
from agents.critic import run_critic_logic
from agents.strategist import build_prompt as build_strategist_prompt, parse_strategist_output
from agents.post_writer import run_post_writer
from agents.article_writer import run_article_writer
from agents.editor import run_editor
from agents.linkedin_formatter import run_formatter
from agents.publisher_qc import run_publisher_qc
from agents.theme_tracker import (
    load_used_themes, save_used_themes, purge_expired_themes,
    filter_by_theme, classify_theme, mark_theme_used, THEME_BLOCK_DAYS,
    THEME_TAXONOMY,
)
from agents.similarity_filter import filter_by_similarity
from agents.researcher import (
    fetch_all_feeds, scored_items_are_fresh,
    load_scored_items, FRESHNESS_HOURS,
)
from utils.ollama_client import (
    generate_text, REASONING_MODEL, WRITING_MODEL,
    POST_MAX_TOKENS, ARTICLE_MAX_TOKENS
)

MAX_CRITIC_RETRIES     = 3
ITEM_BLOCK_DAYS        = 30
SYNTHESIS_ITEM_COUNT   = 3  # number of top items to synthesize into each post/article

# ── LLM call timeout (seconds) ───────────────────────────────────────────────
# FIX: Increased from 90s to 180s. qwen2.5:7b regularly exceeds 90s on
# article writing, editor, and formatter calls — the old value caused all
# three to time out and skip on nearly every run.
LLM_TIMEOUT_S = 300

DRAFT_FILES = {
    "post":    "data/outputs/post_draft.txt",
    "article": "data/outputs/article_draft.txt",
}
FINAL_FILES = {
    "post":    "data/outputs/post_final.txt",
    "article": "data/outputs/article_final.txt",
}
# Immutable agent baseline — written by the pipeline, never edited by the human.
# The feedback ingester diffs the human's edited post_final.txt against these.
AGENT_FILES = {
    "post":    "data/outputs/post_agent.txt",
    "article": "data/outputs/article_agent.txt",
}
ARCHIVE_DIR = "data/archive"
USED_LOG    = "data/processed/used_items.json"
RUN_LOG          = "data/logs/pipeline_runs.json"
SCORED_ITEMS_FILE = "data/processed/scored_items.json"

VALID_CONTENT_TYPES = {"post", "article", "both"}
VALID_POST_FORMATS  = {"harsh_truth", "leadership_moment", "contrarian_breakdown", "framework_story"}


# ── Async-safe LLM wrapper ────────────────────────────────────────────────────

async def llm_async(prompt: str, model: str = None,
                    temperature: float = None,
                    max_tokens: int = None,
                    timeout: float = LLM_TIMEOUT_S) -> str:
    """
    Run the blocking generate_text() in a thread executor with a timeout.

    FIX: generate_text() is synchronous (Ollama SDK). Calling it bare inside
    an async function blocks the entire event loop with no timeout, causing
    the silent hang observed at Phase 2 and Phase 3/4 rewrite calls.

    This wrapper offloads the call to a thread and wraps it in asyncio.wait_for()
    so any stall surfaces as a TimeoutError rather than an infinite freeze.
    """
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: generate_text(prompt, model=model,
                                      temperature=temperature,
                                      max_tokens=max_tokens)
            ),
            timeout=timeout,
        )
        return result.strip() if result else ""
    except asyncio.TimeoutError:
        print(f"  ⚠️  [LLM] Call timed out after {timeout}s.", flush=True)
        return ""


# ── Timing helpers ────────────────────────────────────────────────────────────

def format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.1f}s"


def phase_timer(phase_name: str, start: float) -> None:
    elapsed = time.time() - start
    print(f"[Timing] {phase_name} completed in {format_elapsed(elapsed)}", flush=True)


def _load_run_log() -> list:
    os.makedirs(os.path.dirname(RUN_LOG), exist_ok=True)
    if os.path.exists(RUN_LOG):
        try:
            with open(RUN_LOG, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return []


def _save_run_log(runs: list) -> None:
    with open(RUN_LOG, "w") as f:
        json.dump(runs, f, indent=2)


def start_run_log(content_type: str) -> int:
    """
    Write a 'started' entry immediately when the pipeline launches.
    Returns the index of the entry so update_run_log() can patch it later.
    Captures runs that crash, hang, or get killed — not just completed ones.
    """
    runs = _load_run_log()
    entry = {
        "run_date":        datetime.now().strftime("%Y-%m-%d"),
        "run_time":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "content_type":    content_type,
        "status":          "started",
        "reasoning_model": REASONING_MODEL,
        "writing_model":   WRITING_MODEL,
    }
    runs.append(entry)
    _save_run_log(runs)
    idx = len(runs) - 1
    print(f"[Run Log] Run #{idx + 1} started — {RUN_LOG}")
    return idx


def update_run_log(idx: int, updates: Dict) -> None:
    """
    Patch the run entry at idx with final details (status, timing, topic, etc.).
    Called at pipeline end — success or failure.
    """
    runs = _load_run_log()
    if idx < len(runs):
        runs[idx].update(updates)
        _save_run_log(runs)
        print(f"[Run Log] Run #{idx + 1} updated → status: "
              f"{updates.get('status', '?')} ({len(runs)} total runs logged.)")


# ── Used items log ────────────────────────────────────────────────────────────

def load_used_items() -> Dict:
    if not os.path.exists(USED_LOG):
        return {}
    try:
        with open(USED_LOG, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_used_items(used: Dict) -> None:
    os.makedirs(os.path.dirname(USED_LOG), exist_ok=True)
    with open(USED_LOG, "w") as f:
        json.dump(used, f, indent=2)


def item_key(item: Dict) -> str:
    return item.get("link") or item.get("title", "unknown")


def purge_expired_items(used: Dict) -> Dict:
    cutoff = datetime.now() - timedelta(days=ITEM_BLOCK_DAYS)
    purged = {k: v for k, v in used.items()
              if datetime.fromisoformat(v) > cutoff}
    removed = len(used) - len(purged)
    if removed:
        print(f"[Used Items] Purged {removed} expired item(s) "
              f"(>{ITEM_BLOCK_DAYS} days old).")
    return purged


def filter_unused_items(scored_items: List[Dict], used: Dict) -> List[Dict]:
    unused = [item for item in scored_items if item_key(item) not in used]

    if unused:
        print(f"[Used Items] {len(scored_items) - len(unused)} item(s) skipped "
              f"(used in last {ITEM_BLOCK_DAYS} days). "
              f"{len(unused)} fresh item(s) available.", flush=True)
        return unused

    print(f"[Used Items] ⚠️  All items used in last {ITEM_BLOCK_DAYS} days. "
          f"Falling back to oldest.", flush=True)

    def last_used_date(item):
        key = item_key(item)
        return datetime.fromisoformat(used.get(key, "2000-01-01T00:00:00"))

    oldest = min(scored_items, key=last_used_date)
    return [oldest] + [i for i in scored_items if i != oldest]


def mark_item_used(item: Dict, used: Dict) -> Dict:
    key = item_key(item)
    used[key] = datetime.now().isoformat()
    return used


# ── Archive ───────────────────────────────────────────────────────────────────

def archive_previous_drafts():
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archived  = []

    all_files = {**DRAFT_FILES, **{f"{k}_final": v for k, v in FINAL_FILES.items()}}

    for label, path in all_files.items():
        if os.path.exists(path):
            archive_name = f"{label}_{timestamp}.txt"
            archive_path = os.path.join(ARCHIVE_DIR, archive_name)
            with open(path, "r") as src, open(archive_path, "w") as dst:
                dst.write(src.read())
            archived.append(archive_path)

    if archived:
        print(f"[Archive] Saved {len(archived)} previous file(s) to {ARCHIVE_DIR}/")
        for path in archived:
            print(f"          → {path}")
    else:
        print(f"[Archive] No previous drafts to archive — first run.")


# ── Pipeline state ────────────────────────────────────────────────────────────

@dataclass
class PipelineState:
    raw_items: List[Dict] = field(default_factory=list)
    filtered_items: List[Dict] = field(default_factory=list)
    scored_items: List[Dict] = field(default_factory=list)
    content_plan: Dict = field(default_factory=dict)
    drafts: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── Critic rewrite prompt ─────────────────────────────────────────────────────

def build_rewrite_prompt(draft: str, content_type: str, feedback: list,
                         attempt_history: list) -> str:
    feedback_text = "\n".join(f"- {f}" for f in feedback)

    history_text = ""
    if attempt_history:
        history_text = "\n\nPREVIOUS FAILED ATTEMPTS (do not repeat these approaches):\n"
        for i, h in enumerate(attempt_history, 1):
            history_text += f"\nAttempt {i} was rejected for:\n"
            history_text += "\n".join(f"  - {r}" for r in h["reasons"])

    return f"""
You are revising a LinkedIn {content_type} draft based on editorial feedback.

CURRENT REJECTION REASONS (fix all of these):
{feedback_text}
{history_text}

Rules:
- Fix every current rejection reason listed above
- Do not repeat any approach from the previous failed attempts
- Do not introduce new banned phrases
- Preserve the core argument and operator-level tone
- No em-dashes, no hashtags, no generic motivational language
- Output only the revised {content_type} body, nothing else

Original draft:
{draft}
""".strip()


# ── Theme filter ──────────────────────────────────────────────────────────────

def filter_by_requested_theme(items: List[Dict], theme: str) -> List[Dict]:
    """
    Filter scored items to those whose classified theme matches the requested
    theme. Uses the theme already classified on each item if present; otherwise
    classifies on the fly using classify_theme().

    If no items match, returns the full list with a warning so the pipeline
    never hard-fails on a theme request.
    """
    theme_lower = theme.strip().lower()

    # Find the closest valid taxonomy match (allow partial input)
    matched_taxonomy = None
    for t in THEME_TAXONOMY:
        if theme_lower == t.lower() or theme_lower in t.lower():
            matched_taxonomy = t
            break

    if not matched_taxonomy:
        print(f"⚠️  [Theme Filter] '{theme}' not found in taxonomy. "
              f"Valid themes: {', '.join(THEME_TAXONOMY)}", flush=True)
        print(f"  Proceeding with unfiltered item pool.", flush=True)
        return items

    print(f"[Theme Filter] Filtering for theme: '{matched_taxonomy}'...", flush=True)

    matched = []
    for item in items:
        # Use pre-classified theme if available
        item_theme = item.get("theme", "")
        if not item_theme:
            item_theme = classify_theme(item) or "general"
            item["theme"] = item_theme
        if item_theme.lower() == matched_taxonomy.lower():
            matched.append(item)

    if matched:
        print(f"  [Theme Filter] {len(matched)} item(s) match '{matched_taxonomy}'.",
              flush=True)
        return matched

    print(f"  ⚠️  [Theme Filter] No items matched '{matched_taxonomy}'. "
          f"Falling back to full item pool.", flush=True)
    return items


# ── Synthesis pool deduplication ─────────────────────────────────────────────

def deduplicate_by_title(items: List[Dict]) -> List[Dict]:
    """
    FIX: Remove items with duplicate titles before building the synthesis pool.
    The researcher can return the same article from multiple RSS feeds under the
    same title, which causes the strategist to synthesize near-identical content
    and reliably land back on the same theme.
    Preserves original score order — first occurrence of each title wins.
    """
    seen_titles: set = set()
    unique: List[Dict] = []
    dupes = 0
    for item in items:
        title = item.get("title", "").strip().lower()
        if title and title not in seen_titles:
            seen_titles.add(title)
            unique.append(item)
        else:
            dupes += 1
    if dupes:
        print(f"  [Title Dedup] Removed {dupes} duplicate title(s) from synthesis pool.",
              flush=True)
    return unique


# ── Main pipeline ─────────────────────────────────────────────────────────────

async def run_pipeline(content_type: str, requested_theme: str = "", forced_format: str = None):
    state = PipelineState()
    pipeline_start = time.time()
    phase_times    = {}
    print(f"--- Starting Leadership Agent Pipeline (In-Memory Mode) ---")
    print(f"[Config] Content type:    {content_type}")
    print(f"[Config] Reasoning model: {REASONING_MODEL}")
    print(f"[Config] Writing model:   {WRITING_MODEL}")
    print(f"[Config] Started at:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if requested_theme:
        print(f"[Config] Requested theme: {requested_theme}")

    # Write 'started' log entry immediately — captures runs that never finish
    run_log_idx = start_run_log(content_type)

    try:
        await _run_pipeline_body(
            state, content_type, pipeline_start, phase_times, run_log_idx,
            requested_theme=requested_theme,
            forced_format=forced_format,
        )
    except SystemExit:
        # sys.exit() calls inside the pipeline — log as failed then re-raise
        total_elapsed = time.time() - pipeline_start
        update_run_log(run_log_idx, {
            "status":        "failed",
            "total_seconds": round(total_elapsed, 1),
            "total_time":    format_elapsed(total_elapsed),
            "phase_times":   phase_times,
        })
        raise
    except Exception as e:
        total_elapsed = time.time() - pipeline_start
        update_run_log(run_log_idx, {
            "status":        "failed",
            "error":         str(e),
            "total_seconds": round(total_elapsed, 1),
            "total_time":    format_elapsed(total_elapsed),
            "phase_times":   phase_times,
        })
        print(f"PIPELINE FAILED: {e}")
        raise


async def _run_pipeline_body(state: PipelineState, content_type: str,
                              pipeline_start: float, phase_times: Dict,
                              run_log_idx: int,
                              requested_theme: str = "",
                              forced_format: str = None):

    archive_previous_drafts()

    with open("brand/guide.yaml", "r") as f:
        brand = yaml.safe_load(f)

    # PHASE 1: RESEARCH & FILTERING
    # If the standalone researcher has run recently and left fresh scored items,
    # skip Phase 1 entirely — items are already filtered, scored, and waiting.
    phase1_start = time.time()
    if scored_items_are_fresh() and os.path.exists(SCORED_ITEMS_FILE):
        try:
            state.scored_items = load_scored_items()
            print(f"[Phase 1] ⚡ Skipping — using pre-scored items from researcher "
                  f"({len(state.scored_items)} items, fresher than {FRESHNESS_HOURS}h)",
                  flush=True)
            phase_times["feed_fetch_s"]  = 0.0
            phase_times["gatekeeper_s"]  = 0.0
            phase_times["scorer_s"]      = 0.0
            phase_times["phase1_s"]      = round(time.time() - phase1_start, 1)
            phase_times["phase1_source"] = "researcher_cache"
            if not state.scored_items:
                print("❌ Pre-scored items file is empty. Falling through to live fetch.")
                raise ValueError("empty_cache")
        except ValueError:
            pass  # fall through to live fetch below
        except Exception as e:
            print(f"  ⚠️  Could not load pre-scored items: {e}. Running live fetch.")
            state.scored_items = []

    if not state.scored_items:
        try:
            t = time.time()
            print("[Phase 1] Fetching feeds...")
            state.raw_items = await fetch_all_feeds("sources/rss_feeds.txt")
            phase_times["feed_fetch_s"] = round(time.time() - t, 1)
            print(f"[Timing]  Feed fetch: {format_elapsed(phase_times['feed_fetch_s'])} "
                  f"({len(state.raw_items)} items)", flush=True)

            t = time.time()
            print("[Phase 1] Filtering 'Organizational Theater'...")
            state.filtered_items, _ = await run_gatekeeper_async(state.raw_items, brand)
            phase_times["gatekeeper_s"] = round(time.time() - t, 1)
            print(f"[Timing]  Gatekeeper: {format_elapsed(phase_times['gatekeeper_s'])} "
                  f"({len(state.filtered_items)} items kept)", flush=True)

            t = time.time()
            print("[Phase 1] Semantic Vibe Check...")
            state.scored_items = await semantic_scorer(state.filtered_items)
            phase_times["scorer_s"] = round(time.time() - t, 1)
            print(f"[Timing]  Scorer: {format_elapsed(phase_times['scorer_s'])} "
                  f"({len(state.scored_items)} items scored)", flush=True)

            if not state.scored_items:
                print("❌ No items passed the gatekeeper/scorer. Exiting.")
                return

        except Exception as e:
            print(f"CRITICAL FAIL in Phase 1: {e}")
            sys.exit(1)

        phase_times["phase1_s"]      = round(time.time() - phase1_start, 1)
        phase_times["phase1_source"] = "live_fetch"

    phase_timer("Phase 1 (Research + Filter + Score)", phase1_start)

    # PHASE 1d: DEDUPLICATION
    print("[Phase 1] Running deduplication filters...")

    used_items = load_used_items()
    used_items = purge_expired_items(used_items)
    state.scored_items = filter_unused_items(state.scored_items, used_items)

    used_themes = load_used_themes()
    used_themes = purge_expired_themes(used_themes)
    state.scored_items = filter_by_theme(state.scored_items, used_themes)

    t = time.time()
    state.scored_items = filter_by_similarity(state.scored_items)
    print(f"[Timing]  Similarity filter: {format_elapsed(time.time() - t)}", flush=True)

    if not state.scored_items:
        print("❌ No items survived deduplication filters. Exiting.")
        return

    # THEME SELECTION — filter pool to requested theme if provided
    if requested_theme:
        state.scored_items = filter_by_requested_theme(
            state.scored_items, requested_theme
        )

    if len(state.scored_items) <= 3:
        print(f"⚠️  [Content Warning] Only {len(state.scored_items)} fresh item(s) "
              f"available after deduplication. Consider adding more RSS feeds.", flush=True)

    # FIX: Deduplicate synthesis pool by title before passing to strategist.
    # The researcher can surface the same article from multiple feeds under the
    # same title. When 2-of-3 synthesis items are identical, the strategist
    # reliably converges on the same theme every run.
    state.scored_items = deduplicate_by_title(state.scored_items)

    # PHASE 2: STRATEGIST
    # FIX: generate_text() is synchronous. Running it bare inside async blocks
    # the event loop indefinitely if Ollama is slow. llm_async() wraps it in
    # run_in_executor() with a 90s timeout so hangs surface as errors, not freezes.
    phase2_start = time.time()
    try:
        print("[Phase 2] Determining Content Strategy...")

        blocked_themes = set(used_themes.keys()) if used_themes else set()

        # FIX: Post-strategist theme guard. The item-level theme filter runs
        # before the strategist, but the strategist classifies its *output angle*
        # independently — and can land on a blocked theme even with clean inputs.
        # We try up to 2 batches before accepting the best available result.
        state.content_plan = None
        batch_offset = 0
        MAX_STRATEGY_ATTEMPTS = 2

        for strategy_attempt in range(MAX_STRATEGY_ATTEMPTS):
            batch_start = batch_offset
            batch_end   = batch_start + SYNTHESIS_ITEM_COUNT
            top_items   = state.scored_items[batch_start:batch_end]

            if not top_items:
                print(f"  [Theme Guard] No items in batch {strategy_attempt + 1}. "
                      f"Using previous plan.", flush=True)
                break

            top_item          = top_items[0]
            strategist_prompt = build_strategist_prompt(top_items, brand)

            raw_plan = await llm_async(
                strategist_prompt,
                model=REASONING_MODEL,
                timeout=LLM_TIMEOUT_S,
            )

            if not raw_plan:
                print("⚠️  Strategist LLM call timed out or returned empty. Using defaults.")

            candidate_plan = parse_strategist_output(raw_plan, top_items)

            if not candidate_plan:
                print("⚠️ Strategist output could not be parsed. Using defaults.")
                candidate_plan = {
                    "content_type":    content_type,  # CLI arg
                    "post_format":     "harsh_truth",
                    "reason":          "Fallback: could not parse strategist output.",
                    "topic_title":     top_item.get("title", ""),
                    "synthesis_angle": "",
                }

            # Classify the angle the strategist chose
            print("[Phase 2] Classifying topic theme...")
            angle_item = {
                "title":   candidate_plan.get("topic_title", top_item.get("title", "")),
                "summary": candidate_plan.get("synthesis_angle", ""),
            }
            theme = classify_theme(angle_item) or classify_theme(top_item) or "general"
            candidate_plan["classified_theme"] = theme

            # Check if the strategist landed on a blocked theme
            from agents.theme_tracker import get_blocked_themes
            currently_blocked = get_blocked_themes(used_themes)

            if theme in currently_blocked and strategy_attempt < MAX_STRATEGY_ATTEMPTS - 1:
                print(f"  [Theme Guard] Strategist chose blocked theme '{theme}' "
                      f"(attempt {strategy_attempt + 1}). "
                      f"Retrying with next batch of items...", flush=True)
                batch_offset += SYNTHESIS_ITEM_COUNT
                continue  # try next batch

            if theme in currently_blocked:
                print(f"  [Theme Guard] ⚠️  All batches landed on blocked themes. "
                      f"Proceeding with best available plan.", flush=True)

            candidate_plan["content_type"] = content_type  # CLI arg always wins
            state.content_plan = candidate_plan
            top_item["theme"]  = theme
            break

        # Ensure we always have a plan
        if not state.content_plan:
            top_items = state.scored_items[:SYNTHESIS_ITEM_COUNT]
            top_item  = top_items[0]
            state.content_plan = {
                "content_type":    content_type,  # CLI arg
                "post_format":     "harsh_truth",
                "reason":          "Fallback: all strategy attempts exhausted.",
                "topic_title":     top_item.get("title", ""),
                "synthesis_angle": "",
                "classified_theme": "general",
            }
            theme    = "general"
            top_item = top_items[0]

        theme    = state.content_plan.get("classified_theme", "general")
        top_item = state.scored_items[0]  # always mark the top scored item as used

        if "selected_items" not in state.content_plan:
            state.content_plan["selected_items"] = state.scored_items[:SYNTHESIS_ITEM_COUNT]
        if "selected_item" not in state.content_plan:
            state.content_plan["selected_item"] = top_item

        print(f"✅ Synthesis across:  {len(state.content_plan['selected_items'])} item(s)")
        for i, it in enumerate(state.content_plan["selected_items"], 1):
            print(f"   {i}. {it.get('title', '')[:65]}")
        print(f"✅ Synthesis angle:   {state.content_plan.get('synthesis_angle', 'n/a')[:65]}")
        print(f"✅ Content type:      {state.content_plan.get('content_type')}")
        print(f"✅ Post format:       {state.content_plan.get('post_format')}")

        used_items  = mark_item_used(top_item, used_items)
        save_used_items(used_items)

        used_themes = mark_theme_used(theme, used_themes)
        save_used_themes(used_themes)
        print(f"[Used Items]    Marked as used: {top_item.get('title', '')[:60]}")
        print(f"[Theme Tracker] Theme '{theme}' blocked for {THEME_BLOCK_DAYS} days.")

    except Exception as e:
        print(f"CRITICAL FAIL in Phase 2: {e}")
        sys.exit(1)

    phase_times["phase2_s"] = round(time.time() - phase2_start, 1)
    phase_timer("Phase 2 (Strategist)", phase2_start)

    # PHASE 3 & 4: WRITING & CRITIC LOOP
    phase34_start = time.time()
    if content_type == "both":
        content_types = ["post", "article"]
    else:
        content_types = [content_type]

    items = state.content_plan.get("selected_items") or [state.content_plan["selected_item"]]

    for c_type in content_types:
        c_type_start    = time.time()
        attempt_history = []
        print(f"[Phase 3] Writing {c_type}...")

        try:
            loop = asyncio.get_event_loop()
            if c_type == "post":
                post_format     = state.content_plan.get("post_format", "harsh_truth")
                if forced_format and forced_format in VALID_POST_FORMATS:
                    post_format = forced_format
                    print(f"  [Post Writer] ⚡ Format override: {post_format}", flush=True)
                synthesis_angle = state.content_plan.get("synthesis_angle", "")
                print(f"  [Post Writer] Running with {LLM_TIMEOUT_S}s timeout...")
                try:
                    draft = await asyncio.wait_for(
                        loop.run_in_executor(
                            None, lambda: run_post_writer(items, brand, post_format, synthesis_angle=synthesis_angle)
                        ),
                        timeout=LLM_TIMEOUT_S,
                    )
                except asyncio.TimeoutError:
                    print(f"  ⚠️  Post writer timed out after {LLM_TIMEOUT_S}s. Skipping.")
                    continue
                draft = re.sub(
                    r"^(Harsh Truth|Leadership Moment|Contrarian Breakdown)\s*\n+",
                    "", draft, flags=re.IGNORECASE
                ).strip()
                state.drafts[c_type] = draft
            else:
                print(f"  [Article Writer] Running with {LLM_TIMEOUT_S}s timeout...")
                try:
                    draft = await asyncio.wait_for(
                        loop.run_in_executor(
                            None, lambda: run_article_writer(items, brand, synthesis_angle=synthesis_angle)
                        ),
                        timeout=LLM_TIMEOUT_S,
                    )
                except asyncio.TimeoutError:
                    print(f"  ⚠️  Article writer timed out after {LLM_TIMEOUT_S}s. Skipping.")
                    continue
                state.drafts[c_type] = draft
        except Exception as e:
            print(f"  ❌ Writer failed for {c_type}: {e}")
            continue

        print(f"[Phase 4] Critic loop for {c_type}...")
        for attempt in range(1, MAX_CRITIC_RETRIES + 1):
            draft = state.drafts.get(c_type, "")

            if not draft:
                print(f"  ⚠️  No draft for {c_type} — skipping critic loop.")
                break

            print(f"  Attempt {attempt}: Evaluating brand alignment...")
            is_accepted, feedback = run_critic_logic(draft, c_type, attempt=attempt)

            if is_accepted:
                print(f"  ✅ {c_type} passed Critic on attempt {attempt}.")
                break

            print(f"  ⚠️ {c_type} rejected: {feedback}")
            attempt_history.append({"attempt": attempt, "reasons": feedback})

            if attempt == MAX_CRITIC_RETRIES:
                print(f"  ❌ Max retries reached for {c_type}. "
                      f"Using best available draft.")
                break

            print(f"  🔄 Rewriting {c_type} (attempt {attempt + 1} of "
                  f"{MAX_CRITIC_RETRIES}) with progressive feedback...")
            rewrite_prompt = build_rewrite_prompt(
                draft, c_type, feedback, attempt_history[:-1]
            )

            # FIX: same async wrapper applied to rewrite calls
            revised = await llm_async(
                rewrite_prompt,
                model=REASONING_MODEL,
                timeout=LLM_TIMEOUT_S,
            )

            if revised:
                state.drafts[c_type] = revised
                draft_path = DRAFT_FILES.get(c_type)
                if draft_path:
                    os.makedirs(os.path.dirname(draft_path), exist_ok=True)
                    with open(draft_path, "w") as f:
                        f.write(revised)
                print(f"  📝 Revised draft ready for attempt {attempt + 1}.")
            else:
                print(f"  ⚠️ Rewrite returned empty or timed out. "
                      f"Retrying with original draft.")

        phase_timer(f"Phase 3/4 ({c_type})", c_type_start)

    phase_times["phase34_s"] = round(time.time() - phase34_start, 1)
    phase_timer("Phase 3/4 (Writing + Critic total)", phase34_start)

    # PHASE 5-7: EDITOR → FORMATTER → PUBLISHER QC
    phase57_start = time.time()
    print("[Phase 5] Editing drafts...")

    qc_results    = {}
    final_content = {}

    for c_type in content_types:
        draft = state.drafts.get(c_type, "")
        if not draft:
            print(f"  ⚠️  No draft for {c_type} — skipping editor/formatter/QC.")
            continue

        loop = asyncio.get_event_loop()

        try:
            print(f"  [Editor] Running with {LLM_TIMEOUT_S}s timeout...")
            edited = await asyncio.wait_for(
                loop.run_in_executor(
                    None, lambda: run_editor(draft, c_type, brand)
                ),
                timeout=LLM_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            print(f"  ⚠️  Editor timed out after {LLM_TIMEOUT_S}s. Using critic-approved draft.")
            edited = draft
        except Exception as e:
            print(f"  ⚠️  Editor failed for {c_type}: {e}. Using critic-approved draft.")
            edited = draft

        try:
            print(f"  [Formatter] Running with {LLM_TIMEOUT_S}s timeout...")
            formatted = await asyncio.wait_for(
                loop.run_in_executor(
                    None, lambda: run_formatter(edited, c_type, brand)
                ),
                timeout=LLM_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            print(f"  ⚠️  Formatter timed out after {LLM_TIMEOUT_S}s. Using edited draft.")
            formatted = edited
        except Exception as e:
            print(f"  ⚠️  Formatter failed for {c_type}: {e}. Using edited draft.")
            formatted = edited

        final_content[c_type] = formatted

        print(f"[Phase 7] Publisher QC for {c_type}...")
        try:
            qc_passed, qc_result = run_publisher_qc(formatted, c_type)
            qc_results[c_type]   = qc_result
            if not qc_passed:
                print(f"  ⚠️  QC issues for {c_type}: {qc_result['issues']}")
                print(f"       Using best available output — review before posting.")
        except Exception as e:
            print(f"  ⚠️  QC failed for {c_type}: {e}.")

    try:
        os.makedirs("data/outputs", exist_ok=True)
        with open("data/outputs/publisher_qc.json", "w") as f:
            json.dump({
                "run_time":     datetime.now().isoformat(),
                "overall_pass": all(r.get("passed", False)
                                    for r in qc_results.values()),
                "results":      qc_results,
            }, f, indent=2)
    except Exception as e:
        print(f"  ⚠️  Could not save QC results: {e}")

    phase_timer("Phase 5-7 (Edit + Format + QC)", phase57_start)

    print("\n[Finalizing output files...]")
    created = []
    for c_type in content_types:
        final_text = final_content.get(c_type) or state.drafts.get(c_type, "")
        if not final_text:
            continue

        # Human-editable final (post_final.txt / article_final.txt)
        final_path = FINAL_FILES.get(c_type)
        if final_path:
            os.makedirs(os.path.dirname(final_path), exist_ok=True)
            with open(final_path, "w") as f:
                f.write(final_text)
            created.append((c_type, final_path))

        # Immutable agent baseline (post_agent.txt / article_agent.txt)
        # Written once, never touched again. The feedback ingester diffs
        # the human's edited final against this to measure what changed.
        agent_path = AGENT_FILES.get(c_type)
        if agent_path:
            os.makedirs(os.path.dirname(agent_path), exist_ok=True)
            with open(agent_path, "w") as f:
                f.write(final_text)

    if created:
        print(f"\n[Ready for editing]")
        for c_type, path in created:
            print(f"  ✏️  {c_type.capitalize()} → {path}")
        print(f"\n  Edit the final file(s) above, then run the ingester:")
        for c_type, _ in created:
            print(f"    python agents/feedback_ingester.py --type {c_type}")

    total_elapsed = time.time() - pipeline_start
    top_item      = state.content_plan.get("selected_item", {})

    update_run_log(run_log_idx, {
        "status":           "complete",
        "post_format":      state.content_plan.get("post_format", "n/a"),
        "selected_topic":   top_item.get("title", "unknown")[:80],
        "selected_source":  top_item.get("source", "unknown"),
        "theme":            top_item.get("theme", "unknown"),
        "requested_theme":  requested_theme or "auto",
        "raw_items":        len(state.raw_items),
        "filtered_items":   len(state.filtered_items),
        "scored_items":     len(state.scored_items),
        "total_seconds":    round(total_elapsed, 1),
        "total_time":       format_elapsed(total_elapsed),
        "phase_times":      phase_times,
        "drafts_generated": list(state.drafts.keys()),
        "end_time":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })

    print(f"\n--- Pipeline Complete ---")
    print(f"[Timing] Total run time: {format_elapsed(total_elapsed)}")
    print(f"[Timing] Finished at:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Leadership Agent content pipeline."
    )
    parser.add_argument(
        "--type",
        choices=["post", "article", "both"],
        default="both",
        help="Content type to generate (default: both)"
    )
    parser.add_argument(
        "--format",
        choices=[
            "harsh_truth", "leadership_moment", "contrarian_breakdown",
            "framework_story", "data_driven", "case_study",
            "failure_autopsy", "vs_breakdown", "cultural_lens"
        ],
        default=None,
        help=(
            "Force a specific post/article format. If not set, the strategist selects automatically. "
            "Options: harsh_truth, leadership_moment, contrarian_breakdown, framework_story, "
            "data_driven, case_study, failure_autopsy, vs_breakdown, cultural_lens"
        )
    )
    parser.add_argument(
        "--theme",
        default="",
        help=(
            "Filter content pool to a specific theme before selection. "
            "Valid themes: accountability, governance, execution, data leadership, "
            "organizational dysfunction, leadership failure, federal healthcare, "
            "team performance, decision making, strategy, technology leadership, "
            "workforce management, project management, change management, crisis leadership. "
            "If no items match, falls back to the full pool."
        )
    )
    args = parser.parse_args()
    asyncio.run(run_pipeline(content_type=args.type, requested_theme=args.theme, forced_format=args.format))
