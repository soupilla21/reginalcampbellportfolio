#!/usr/bin/env python3
"""
agents/feedback_ingester.py

Human-in-the-loop feedback agent. Run this after you've finalized a draft
to your standard. It saves the final to brand/examples/, compares it against
the agent's final output to log what changed, and flags persistent patterns
for optional promotion into brand/guide.yaml.

Usage:
    # Ingest a finalized post (defaults to data/outputs/post_final.txt)
    python agents/feedback_ingester.py --type post

    # Ingest a finalized article (defaults to data/outputs/article_final.txt)
    python agents/feedback_ingester.py --type article

    # Ingest a specific file
    python agents/feedback_ingester.py --type post --file path/to/my_post.txt

    # Skip diff comparison (just save as example)
    python agents/feedback_ingester.py --type post --no-compare

    # View the current feedback log summary
    python agents/feedback_ingester.py --summary

    # Write promoted patterns directly into brand/guide.yaml
    python agents/feedback_ingester.py --promote
"""

import argparse
import difflib
import os
import re
import sys
import yaml
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────

BRAND_FILE           = "brand/guide.yaml"
FEEDBACK_LOG         = "brand/feedback_log.yaml"
EXAMPLES_POST_DIR    = "brand/examples/posts"
EXAMPLES_ARTICLE_DIR = "brand/examples/articles"

# FIX: Compare against the immutable agent baseline files written by the
# orchestrator at the end of each run (post_agent.txt / article_agent.txt).
# These are identical to post_final.txt at pipeline completion but are never
# edited by the human, giving the ingester a stable diff target.
AGENT_FINAL_POST_FILE    = "data/outputs/post_agent.txt"
AGENT_FINAL_ARTICLE_FILE = "data/outputs/article_agent.txt"

PATTERN_PROMOTION_THRESHOLD = 3


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_text(filepath):
    with open(filepath, "r") as f:
        return f.read().strip()


def load_yaml(filepath):
    if not os.path.exists(filepath):
        return {}
    with open(filepath, "r") as f:
        return yaml.safe_load(f) or {}


def save_yaml(filepath, data):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        yaml.dump(data, f, default_flow_style=False,
                  allow_unicode=True, sort_keys=False)


def save_text(filepath, text):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(text)


# ── Brand utilities ───────────────────────────────────────────────────────────

def get_cta(brand_data: dict) -> str:
    """Return the CTA template string, stripped of emoji for comparison."""
    raw = brand_data.get("cta_template", "").strip()
    # Strip emoji and leading punctuation for loose matching
    return re.sub(r"[^\w\s,.]", "", raw).strip().lower()


def strip_cta(text: str, brand_data: dict) -> str:
    """
    Remove the CTA follow line from the end of a post before comparison.
    The CTA is now appended automatically by the formatter — it is not a
    meaningful human edit and should not trigger closing_line_changed.
    """
    cta_core = get_cta(brand_data)
    if not cta_core:
        return text
    lines = text.strip().splitlines()
    # Walk from the bottom, drop any lines that are part of the CTA
    while lines:
        clean = re.sub(r"[^\w\s,.]", "", lines[-1]).strip().lower()
        if cta_core[:30] in clean or clean in cta_core:
            lines.pop()
        else:
            break
    return "\n".join(lines).strip()


# FIX: Use the same fuzzy keyword match as critic.py instead of exact
# string matching. The old `concept not in agent_lower` check missed
# concepts expressed slightly differently, generating false positives.
def find_signature_hits(text: str, signature_concepts: list) -> list:
    """Fuzzy keyword match — identical to critic.py logic."""
    stop_words = {"the", "is", "are", "and", "for", "that", "this",
                  "with", "from", "have"}
    lowered = re.sub(r"\s+", " ", text.lower()).strip()
    hits    = []
    for concept in signature_concepts:
        words    = re.findall(r"[a-zA-Z]+", concept)
        keywords = [w.lower() for w in words
                    if len(w) > 3 and w.lower() not in stop_words]
        if any(re.search(rf"\b{re.escape(kw)}\b", lowered)
               for kw in keywords):
            hits.append(concept)
    return hits


# ── Example management ────────────────────────────────────────────────────────

def save_as_example(content_type, final_text):
    """
    Save the finalized draft into brand/examples/{posts|articles}/ with a
    timestamp so writers always have fresh, dated references to draw from.
    """
    directory = EXAMPLES_POST_DIR if content_type == "post" else EXAMPLES_ARTICLE_DIR
    os.makedirs(directory, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"{content_type}_{timestamp}.txt"
    filepath  = os.path.join(directory, filename)
    save_text(filepath, final_text)
    print(f"  ✅ Saved final as example: {filepath}")
    return filepath


def prune_old_examples(content_type, max_examples=10):
    """Keep only the most recent N examples per content type."""
    directory = EXAMPLES_POST_DIR if content_type == "post" else EXAMPLES_ARTICLE_DIR
    if not os.path.exists(directory):
        return
    files = sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".txt")
    ])
    if len(files) > max_examples:
        to_remove = files[:len(files) - max_examples]
        for f in to_remove:
            os.remove(f)
            print(f"  🗑️  Pruned old example: {f}")


# ── Diff analysis ─────────────────────────────────────────────────────────────

def compute_diff(agent_final, human_final):
    """Produce a unified diff between the agent final and the human final."""
    agent_lines = agent_final.splitlines(keepends=True)
    final_lines = human_final.splitlines(keepends=True)
    diff = difflib.unified_diff(
        agent_lines, final_lines,
        fromfile="agent_final",
        tofile="human_final",
        lineterm=""
    )
    return "".join(diff)


def analyze_changes(agent_final, human_final, brand):
    """
    Compare the agent's final output to the human's final and extract
    structured observations.

    FIX 1: Compares against agent_final (post_final.txt) not the raw draft.
    FIX 2: Strips CTA before closing line comparison so automatic CTA
           appending never triggers a false closing_line_changed observation.
    FIX 3: Uses fuzzy keyword match for signature concepts (same as critic).
    FIX 4: Tracks engagement question change separately from CTA presence.
    """
    brand_data         = brand.get("brand", {})
    words_to_avoid     = [w.lower() for w in brand_data.get("words_to_avoid", [])]
    signature_concepts = brand_data.get("signature_concepts", [])

    agent_lower = agent_final.lower()
    final_lower = human_final.lower()

    observations = []

    # 1. Banned phrases that slipped through and were removed by the human
    slipped_banned = [
        phrase for phrase in words_to_avoid
        if phrase in agent_lower and phrase not in final_lower
    ]
    if slipped_banned:
        observations.append({
            "type":   "banned_phrase_removed",
            "detail": slipped_banned,
            "note":   "These banned phrases were in the agent final and removed by the human."
        })

    # 2. Signature concepts added by the human that the agent missed
    # FIX: fuzzy match instead of exact string — same logic as critic.py
    agent_sig_hits = set(find_signature_hits(agent_final, signature_concepts))
    final_sig_hits = set(find_signature_hits(human_final, signature_concepts))
    added_concepts = sorted(final_sig_hits - agent_sig_hits)
    if added_concepts:
        observations.append({
            "type":   "signature_concept_added",
            "detail": added_concepts,
            "note":   "Human added these signature concepts that the agent missed."
        })

    # 3. Word count delta
    agent_wc  = len(agent_final.split())
    final_wc  = len(human_final.split())
    delta     = final_wc - agent_wc
    direction = "expanded" if delta > 0 else "reduced" if delta < 0 else "unchanged"
    observations.append({
        "type":   "word_count_delta",
        "detail": {"agent": agent_wc, "final": final_wc,
                   "delta": delta, "direction": direction},
        "note":   f"Human {direction} the draft by {abs(delta)} words."
    })

    # 4. Opening line changed
    agent_opening = agent_final.strip().splitlines()[0] if agent_final.strip() else ""
    final_opening = human_final.strip().splitlines()[0] if human_final.strip() else ""
    if agent_opening != final_opening:
        observations.append({
            "type":   "opening_line_changed",
            "detail": {"agent": agent_opening, "final": final_opening},
            "note":   "Human rewrote the opening line."
        })

    # 5. Engagement question changed (closing line, CTA stripped first)
    # FIX: strip CTA from both texts before comparing so the auto-appended
    # follow line never triggers this observation spuriously.
    agent_stripped = strip_cta(agent_final, brand_data)
    final_stripped = strip_cta(human_final, brand_data)

    agent_question = agent_stripped.strip().splitlines()[-1] if agent_stripped.strip() else ""
    final_question = final_stripped.strip().splitlines()[-1] if final_stripped.strip() else ""

    if agent_question != final_question:
        observations.append({
            "type":   "closing_question_changed",
            "detail": {"agent": agent_question, "final": final_question},
            "note":   "Human rewrote the closing engagement question."
        })

    return observations


# ── Feedback log ──────────────────────────────────────────────────────────────

def append_to_feedback_log(content_type, final_path, observations, diff):
    """Append a structured entry to brand/feedback_log.yaml."""
    log = load_yaml(FEEDBACK_LOG)
    if "entries" not in log:
        log["entries"] = []

    entry = {
        "date":         datetime.now().isoformat(),
        "content_type": content_type,
        "final_file":   final_path,
        "observations": observations,
        "diff_preview": diff[:800] if diff else None,
    }
    log["entries"].append(entry)
    save_yaml(FEEDBACK_LOG, log)
    print(f"  ✅ Feedback entry appended to {FEEDBACK_LOG}")
    return log


def check_promotion_candidates(log):
    """
    Scan the feedback log for patterns that have occurred >=
    PATTERN_PROMOTION_THRESHOLD times and return them as candidates
    for promotion into brand/guide.yaml.
    """
    if "entries" not in log:
        return []

    pattern_counts = {}

    for entry in log["entries"]:
        for obs in entry.get("observations", []):
            obs_type = obs.get("type")
            detail   = obs.get("detail")

            if obs_type == "banned_phrase_removed" and isinstance(detail, list):
                for phrase in detail:
                    key = f"banned_phrase_slippage::{phrase}"
                    pattern_counts[key] = pattern_counts.get(key, 0) + 1

            elif obs_type == "signature_concept_added" and isinstance(detail, list):
                for concept in detail:
                    key = f"concept_always_added::{concept}"
                    pattern_counts[key] = pattern_counts.get(key, 0) + 1

            elif obs_type == "opening_line_changed":
                key = "opening_line_consistently_changed"
                pattern_counts[key] = pattern_counts.get(key, 0) + 1

            elif obs_type == "closing_question_changed":
                key = "closing_question_consistently_changed"
                pattern_counts[key] = pattern_counts.get(key, 0) + 1

            elif obs_type == "word_count_delta":
                direction = detail.get("direction") if isinstance(detail, dict) else None
                if direction and direction != "unchanged":
                    key = f"word_count_consistently_{direction}"
                    pattern_counts[key] = pattern_counts.get(key, 0) + 1

    return [
        {"pattern": k, "count": v}
        for k, v in pattern_counts.items()
        if v >= PATTERN_PROMOTION_THRESHOLD
    ]


def print_promotion_suggestions(candidates):
    """Print actionable suggestions for patterns ready to be promoted."""
    if not candidates:
        return

    print(f"\n  ⚠️  PATTERN PROMOTION SUGGESTIONS")
    print(f"  The following patterns have appeared {PATTERN_PROMOTION_THRESHOLD}+ times.")
    print(f"  Run with --promote to write these into brand/guide.yaml automatically.\n")

    for c in candidates:
        pattern = c["pattern"]
        count   = c["count"]

        if pattern.startswith("banned_phrase_slippage::"):
            phrase = pattern.split("::", 1)[1]
            print(f"    [{count}x] Agent keeps using banned phrase '{phrase}'.")
            print(f"          → Strengthen the critic check for this phrase.\n")

        elif pattern.startswith("concept_always_added::"):
            concept = pattern.split("::", 1)[1]
            print(f"    [{count}x] You consistently add '{concept}' to drafts.")
            print(f"          → Add stronger weighting in the writer prompt.\n")

        elif pattern == "opening_line_consistently_changed":
            print(f"    [{count}x] You consistently rewrite the opening line.")
            print(f"          → Add more specific opening-line rules to the writer prompt.\n")

        elif pattern == "closing_question_consistently_changed":
            print(f"    [{count}x] You consistently rewrite the closing question.")
            print(f"          → Add closing question examples to brand/guide.yaml.\n")

        elif pattern.startswith("word_count_consistently_"):
            direction = pattern.split("_")[-1]
            print(f"    [{count}x] You consistently {direction} the draft word count.")
            print(f"          → Adjust the word count constraint in the writer prompt.\n")


# ── Promotion writer ──────────────────────────────────────────────────────────

def run_promote(candidates):
    """
    FIX: Actually write confirmed promotions into brand/guide.yaml.
    Previously, promotion suggestions only printed to stdout and were
    never acted on. This command reads the candidates and applies them.

    Currently handles:
    - banned_phrase_slippage  → add phrase to words_to_avoid if missing
    - concept_always_added    → no-op (concepts already in guide.yaml)
    - word_count patterns     → prints advisory only (requires human judgment)
    - opening/closing patterns → prints advisory only
    """
    if not candidates:
        print("  ✅ No patterns have reached the promotion threshold yet.")
        return

    brand     = load_yaml(BRAND_FILE)
    brand_data = brand.get("brand", {})
    modified  = False

    for c in candidates:
        pattern = c["pattern"]
        count   = c["count"]

        if pattern.startswith("banned_phrase_slippage::"):
            phrase       = pattern.split("::", 1)[1]
            avoid_list   = brand_data.get("words_to_avoid", [])
            avoid_lower  = [w.lower() for w in avoid_list]
            if phrase.lower() not in avoid_lower:
                avoid_list.append(phrase)
                brand_data["words_to_avoid"] = avoid_list
                brand["brand"] = brand_data
                modified = True
                print(f"  ✅ Added '{phrase}' to words_to_avoid in brand/guide.yaml "
                      f"({count}x slippage).")
            else:
                print(f"  ℹ️  '{phrase}' already in words_to_avoid — skipping.")

        elif pattern.startswith("concept_always_added::"):
            concept = pattern.split("::", 1)[1]
            print(f"  ℹ️  '{concept}' already in signature_concepts. "
                  f"Consider strengthening its weight in the writer prompt.")

        elif pattern == "opening_line_consistently_changed":
            print(f"  ℹ️  Opening line rewritten {count}x. "
                  f"Review the writer prompt's opening-line rules manually.")

        elif pattern == "closing_question_consistently_changed":
            print(f"  ℹ️  Closing question rewritten {count}x. "
                  f"Consider adding a closing question example to brand/guide.yaml.")

        elif pattern.startswith("word_count_consistently_"):
            direction = pattern.split("_")[-1]
            print(f"  ℹ️  Draft consistently {direction} {count}x. "
                  f"Adjust word count targets in config.yaml manually.")

    if modified:
        save_yaml(BRAND_FILE, brand)
        print(f"\n  ✅ brand/guide.yaml updated.")
    else:
        print(f"\n  ℹ️  No automated changes made — review advisory notes above.")


# ── Summary view ──────────────────────────────────────────────────────────────

def print_summary():
    """Print a human-readable summary of the feedback log."""
    log     = load_yaml(FEEDBACK_LOG)
    entries = log.get("entries", [])

    if not entries:
        print("No feedback entries logged yet.")
        return

    print(f"\n📊 Feedback Log Summary — {len(entries)} entries\n")
    print(f"  {'Date':<25} {'Type':<10} {'Observations'}")
    print(f"  {'-'*24} {'-'*9} {'-'*40}")

    for entry in entries[-10:]:
        date  = entry.get("date", "")[:19]
        ctype = entry.get("content_type", "")
        obs   = entry.get("observations", [])
        types = ", ".join(set(o.get("type", "") for o in obs))
        print(f"  {date:<25} {ctype:<10} {types}")

    candidates = check_promotion_candidates(log)
    print_promotion_suggestions(candidates)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ingest a human-finalized draft to improve the Leadership Agent pipeline."
    )
    parser.add_argument(
        "--type",
        choices=["post", "article"],
        help="Content type to ingest"
    )
    parser.add_argument(
        "--file",
        default="auto",
        help=(
            "Path to your finalized draft file. "
            "Defaults to data/outputs/{type}_final.txt — "
            "edit that file in place after the pipeline runs, then call this."
        )
    )
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="Skip comparison against agent final (just save as example)"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print feedback log summary and exit"
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help=(
            "Write pattern promotions directly into brand/guide.yaml. "
            "Only patterns that have appeared 3+ times are promoted."
        )
    )
    args = parser.parse_args()

    # --summary and --promote don't need --type
    if args.summary:
        print_summary()
        sys.exit(0)

    if args.promote:
        log        = load_yaml(FEEDBACK_LOG)
        candidates = check_promotion_candidates(log)
        run_promote(candidates)
        sys.exit(0)

    if not args.type:
        parser.error("--type is required (use 'post' or 'article')")

    # Resolve the file path
    if args.file == "auto":
        args.file = ("data/outputs/post_final.txt" if args.type == "post"
                     else "data/outputs/article_final.txt")
        print(f"  ℹ️  No --file specified. Using: {args.file}")

    if not os.path.exists(args.file):
        print(f"❌ File not found: {args.file}")
        print(f"   Edit data/outputs/{args.type}_final.txt after the pipeline runs,")
        print(f"   then call this script to ingest it.")
        sys.exit(1)

    print(f"\n🔄 Feedback Ingester — processing {args.type}...")

    # 1. Load the human final
    human_final = load_text(args.file)
    print(f"  📄 Loaded final: {args.file} ({len(human_final.split())} words)")

    # 2. Save as a brand example
    final_path = save_as_example(args.type, human_final)
    prune_old_examples(args.type, max_examples=10)

    # 3. Compare against the agent final
    observations = []
    diff         = None

    if not args.no_compare:
        # FIX: compare against the pipeline's final output, not the raw draft
        agent_file = (AGENT_FINAL_POST_FILE if args.type == "post"
                      else "data/outputs/article_final.txt")

        # If the user passed their own file AND it happens to be the same as
        # the agent final path, we have nothing meaningful to compare.
        if args.file == agent_file:
            print(f"  ⚠️  Cannot compare — your file and the agent baseline are the same path.")
            print(f"       This means the pipeline hasn't run yet, or post_agent.txt is missing.")
            print(f"       Run the pipeline first, then edit post_final.txt, then run the ingester.")
        elif os.path.exists(agent_file):
            agent_final = load_text(agent_file)
            brand       = load_yaml(BRAND_FILE)

            print(f"  🔍 Comparing against agent final: {agent_file}")
            diff         = compute_diff(agent_final, human_final)
            observations = analyze_changes(agent_final, human_final, brand)

            print(f"\n  📋 Change Analysis:")
            for obs in observations:
                print(f"     • {obs['note']}")
                if obs["type"] == "word_count_delta":
                    d = obs["detail"]
                    print(f"       Agent: {d['agent']} words → "
                          f"Final: {d['final']} words (Δ {d['delta']:+d})")
                elif obs["type"] in ("banned_phrase_removed",
                                     "signature_concept_added"):
                    print(f"       Items: {obs['detail']}")
                elif obs["type"] in ("opening_line_changed",
                                     "closing_question_changed"):
                    d = obs["detail"]
                    print(f"       Agent: {d['agent'][:80]}")
                    print(f"       Final: {d['final'][:80]}")
        else:
            print(f"  ⚠️  No agent final found at {agent_file}. Skipping comparison.")
            print(f"       Run the pipeline first, or use --no-compare.")

    # 4. Append to feedback log and check for promotion candidates
    if observations or diff:
        log        = append_to_feedback_log(args.type, final_path,
                                             observations, diff)
        candidates = check_promotion_candidates(log)
        print_promotion_suggestions(candidates)
    else:
        print(f"  ℹ️  No observations to log.")

    print(f"\n✅ Ingestion complete. "
          f"The pipeline will use your final as a reference on the next run.")


if __name__ == "__main__":
    main()
