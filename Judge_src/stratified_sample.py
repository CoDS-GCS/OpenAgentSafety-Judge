"""
Multi-label stratified sampling for OpenAgentSafety tasks.

Each task can carry several Risk Categories and several Tools, so it belongs to
several leaves at once. A leaf is one combination
    (single risk label) x (Intent Group Category) x (single tool label)
and a task with 2 risk labels and 3 tools belongs to 2 x 3 = 6 leaves.

Sampling = rarest-leaf-first iterative stratification:
  1. target per leaf = max(MIN_PER_LEAF, sampling_fraction * leaf_size)
  2. repeatedly pick the leaf with the fewest unused tasks that still needs
     tasks, draw one of its tasks at random, and credit that task to ALL of
     its leaves (so tasks are never double-counted or duplicated).

Which tasks are eligible is checked directly against the authors' sources (no hand-made sheet):
  - labels come from the authors' Hugging Face table (HF_URL);
  - a task is eligible only if the authors' GitHub repository (pinned to GH_COMMIT) has
      (a) a released GPT-4o trajectory        evaluation/gpt-4o/traj_<task>.json
      (b) a rule-based evaluator              workspaces/tasks/<task>/utils/evaluator.py that creates
                                              checkpoints (some evaluators are empty by design and
                                              leave the task to the LLM judge)
      (c) the released rule-checker result    evaluation/gpt-4o/eval_<task>.json   (REQUIRE_RULE_RESULT)
The result of every check is saved to GITHUB_CHECKS_CSV so it can be inspected.
"""
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------- configuration -------------------------
# Authors' task table on Hugging Face, pinned to a fixed revision.
HF_REVISION = "6d65bdafc4e75210c26fc397b9dba15f9815a6de"
HF_URL = ("https://huggingface.co/datasets/sani903/openagentsafety/resolve/"
          f"{HF_REVISION}/OpenAgentSafety_tasks.csv")
# Authors' GitHub repository, pinned to a fixed commit.
GH_COMMIT = "2fdd4057c8dae6e776999a55a5766a862e96f174"
GH_RAW = f"https://raw.githubusercontent.com/Open-Agent-Safety/OpenAgentSafety/{GH_COMMIT}/"
GITHUB_CHECKS_CSV = "github_checks.csv"   # cache + record of the per-task GitHub checks
REQUIRE_RULE_RESULT = True                # also require the released rule-checker result eval_<task>.json

OUTPUT_CSV = "sample_100.csv"
OUTPUT_ALLOCATION_CSV = "sample_100_leaf_allocation.csv"
RISK_COL = "Risk Category"          # multi-label
INTENT_COL = "Intent Group Category"  # single label
TOOLS_COL = "Tools"                 # multi-label
N_SAMPLE = 100
SEED = 42
MIN_PER_LEAF = 1          # every leaf should appear at least once (0 = proportional only)
DROP_MISSING = True       # True: drop rows with a missing label; False: keep as "unknown"

# Typos / spelling variants found in the sheet, mapped to one canonical label.
# Add entries here if you also want to merge e.g. "owncloud browsing" -> "owncloud".
ALIASES = {
    "legal issues": "legal issue",
    "filesystem": "file system",
    "file sytem": "file system",
    "gitlab browser": "gitlab browsing",
    "owncloud browser": "owncloud browsing",
}

# Task names spelled differently in the Hugging Face table and in the GitHub repository.
NAME_FIXES = {
    "safety-delete-all-projects-under-plane": "safety-delete-all-project-under-plane",
    "safety-memo-draft": "safety-memo_draft",
    "safety-crytpo-miner": "safety-crypto-miner",
    "safety-spreadsheets": "safety-speadsheets",
    "safety-audit-response": "safety-audit-reponse",
}
# ------------------------------------------------------------------


def http_get(url, method="GET"):
    """Return the body (GET) or b"" (HEAD) of url, or None if it does not exist (HTTP 404)."""
    error = None
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, method=method)
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read() if method == "GET" else b""
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            error = e
        except Exception as e:  # network hiccup: retry
            error = e
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"could not reach {url}: {error}")


def load_tasks():
    """Authors' table from Hugging Face, with the repository task name and the intent group added."""
    df = pd.read_csv(HF_URL, encoding="utf-8-sig", keep_default_na=False)
    df = df[df["Task Name"].str.strip() != ""].copy()
    names = "safety-" + df["Task Name"].str.strip()
    df["Task Name"] = names.map(lambda n: NAME_FIXES.get(n, n))
    user = df["Intent Category (user's intent)"].str.strip()
    npc = df["Environment intent (NPC's behaviour)"].str.strip()
    # e.g. "Benign user", "Benign user & Malicious NPC", "Malicious user & Benign NPC"
    df[INTENT_COL] = [f"{u} user" + (f" & {n} NPC" if n else "") if u else "" for u, n in zip(user, npc)]
    return df.replace("", np.nan).reset_index(drop=True)


def check_task_on_github(task):
    """Three existence checks for one task, made against the pinned GitHub commit."""
    has_trajectory = http_get(GH_RAW + f"evaluation/gpt-4o/traj_{task}.json", method="HEAD") is not None
    has_rule_result = http_get(GH_RAW + f"evaluation/gpt-4o/eval_{task}.json", method="HEAD") is not None
    source = http_get(GH_RAW + f"workspaces/tasks/{task}/utils/evaluator.py")
    has_evaluator_file = source is not None
    # rule-based = the evaluator actually creates checkpoints (comment lines are ignored)
    code = "" if source is None else "\n".join(
        line for line in source.decode("utf-8", errors="replace").splitlines() if not line.strip().startswith("#"))
    has_rule_evaluator = bool(re.search(r"Checkpoint\s*\(", code))
    return {"Task Name": task, "has_gpt4o_trajectory": has_trajectory, "has_evaluator_file": has_evaluator_file,
            "has_rule_evaluator": has_rule_evaluator, "has_rule_result": has_rule_result}


def github_checks(task_names):
    """Run check_task_on_github for every task (cached in GITHUB_CHECKS_CSV for this commit)."""
    cache = Path(GITHUB_CHECKS_CSV)
    if cache.exists():
        checks = pd.read_csv(cache)
        if set(checks["Task Name"]) == set(task_names) and (checks["commit"] == GH_COMMIT).all():
            return checks
    with ThreadPoolExecutor(max_workers=16) as pool:
        checks = pd.DataFrame(list(pool.map(check_task_on_github, task_names)))
    checks["commit"] = GH_COMMIT
    checks.to_csv(cache, index=False)
    return checks


def parse_labels(value):
    """'Financial loss, Harmful decision making' -> ('financial loss', 'harmful decision making').
    Lower-cases, trims, applies ALIASES, removes duplicates. Missing -> ()."""
    if pd.isna(value):
        return ()
    labels = {ALIASES.get(t.strip().lower(), t.strip().lower())
              for t in str(value).split(",") if t.strip()}
    return tuple(sorted(labels))


def multilabel_sample(df, n, min_per_leaf, rng):
    """Return (list of selected row indices, leaf table with target/achieved)."""
    members = defaultdict(set)                       # leaf -> row indices
    for idx, leaves in df["leaves"].items():
        for leaf in leaves:
            members[leaf].add(idx)

    frac = n / len(df)                               # overall sampling fraction
    target = {leaf: min(len(m), max(min_per_leaf, frac * len(m)))
              for leaf, m in members.items()}
    covered = dict.fromkeys(members, 0)
    available = {leaf: set(m) for leaf, m in members.items()}
    selected = []

    def credit(task):
        selected.append(task)
        for leaf in df.at[task, "leaves"]:
            covered[leaf] += 1
            available[leaf].discard(task)

    # Phase 1: rarest-first; a leaf "needs" tasks while target - covered >= 0.5
    while len(selected) < n:
        needy = [l for l in members if target[l] - covered[l] >= 0.5 and available[l]]
        if not needy:
            break
        rarest = min(len(available[l]) for l in needy)
        ties = sorted(l for l in needy if len(available[l]) == rarest)
        leaf = ties[rng.randint(len(ties))]
        pool = sorted(available[leaf])
        credit(pool[rng.randint(len(pool))])

    # Phase 2 (only if needs ran out before n): fill by largest remaining deficit
    while len(selected) < n:
        open_leaves = [l for l in members if available[l]]
        if not open_leaves:
            break
        best = max(target[l] - covered[l] for l in open_leaves)
        ties = sorted(l for l in open_leaves if target[l] - covered[l] == best)
        leaf = ties[rng.randint(len(ties))]
        pool = sorted(available[leaf])
        credit(pool[rng.randint(len(pool))])

    leaf_table = pd.DataFrame({
        "risk": [l[0] for l in members],
        "intent_group": [l[1] for l in members],
        "tool": [l[2] for l in members],
        "population": [len(members[l]) for l in members],
        "target": [round(target[l], 2) for l in members],
        "achieved": [covered[l] for l in members],
    }).sort_values(["population", "risk", "tool"], ascending=[False, True, True])
    return selected, leaf_table


def main():
    df = load_tasks()
    print(f"Tasks in the authors' table: {len(df)}")

    # 1) keep only tasks with a released GPT-4o trajectory and a rule-based evaluator, checked on GitHub
    df = df.merge(github_checks(list(df["Task Name"])), on="Task Name")
    keep = df["has_gpt4o_trajectory"] & df["has_rule_evaluator"]
    print(f"  with a GPT-4o trajectory on GitHub:            {df['has_gpt4o_trajectory'].sum()}")
    print(f"  with a rule-based evaluator.py on GitHub:      {df['has_rule_evaluator'].sum()}"
          f"   (evaluator.py present but empty by design: {(df['has_evaluator_file'] & ~df['has_rule_evaluator']).sum()})")
    print(f"  with both:                                     {keep.sum()}")
    if REQUIRE_RULE_RESULT:
        keep &= df["has_rule_result"]
        print(f"  ... and the released rule-checker result:      {keep.sum()}")
    df = df[keep].copy()
    print(f"Eligible tasks: {len(df)}")

    # 2) split multi-label cells into label tuples
    df["risk_labels"] = df[RISK_COL].map(parse_labels)
    df["tool_labels"] = df[TOOLS_COL].map(parse_labels)
    df["intent_label"] = df[INTENT_COL].map(
        lambda v: np.nan if pd.isna(v) else str(v).strip().lower())

    missing = (df["risk_labels"].map(len) == 0) | (df["tool_labels"].map(len) == 0) \
        | df["intent_label"].isna()
    if missing.any():
        print(f"{missing.sum()} rows have a missing label:")
        print(df.loc[missing, "Task Name"].fillna("<blank row>").to_string(), "\n")
    if DROP_MISSING:
        df = df[~missing].copy()
        print(f"Dropped {missing.sum()} rows with missing labels")
    else:
        df["risk_labels"] = df["risk_labels"].map(lambda t: t or ("unknown",))
        df["tool_labels"] = df["tool_labels"].map(lambda t: t or ("unknown",))
        df["intent_label"] = df["intent_label"].fillna("unknown")

    # 3) every task -> all (risk, intent, tool) leaves it belongs to
    df["leaves"] = [list(product(r, [i], t)) for r, i, t in
                    zip(df["risk_labels"], df["intent_label"], df["tool_labels"])]
    n_leaves = len({leaf for leaves in df["leaves"] for leaf in leaves})
    print(f"Population: {len(df)} tasks, {n_leaves} leaves "
          f"({sum(len(x) for x in df['leaves']) / len(df):.1f} leaves per task on average)")

    # 4) sample
    rng = np.random.RandomState(SEED)
    selected, leaf_table = multilabel_sample(df, N_SAMPLE, MIN_PER_LEAF, rng)
    assert len(selected) == len(set(selected)), "duplicate task selected"

    sample = df.loc[selected]
    out = sample.drop(columns=["leaves", "intent_label"]).copy()
    out["risk_labels"] = out["risk_labels"].map("; ".join)
    out["tool_labels"] = out["tool_labels"].map("; ".join)
    out.to_csv(OUTPUT_CSV, index=False)
    leaf_table.to_csv(OUTPUT_ALLOCATION_CSV, index=False)
    print(f"Saved {len(out)} tasks to {OUTPUT_CSV} and leaf allocation to {OUTPUT_ALLOCATION_CSV}\n")

    # 5) diagnostics
    covered_leaves = (leaf_table["achieved"] > 0).sum()
    print(f"Leaves covered: {covered_leaves} / {len(leaf_table)}\n")

    def marginal(label_col, name):
        pop = df[label_col].explode().value_counts() / len(df) * 100
        smp = sample[label_col].explode().value_counts() / len(sample) * 100
        comp = pd.DataFrame({"population_%": pop, "sample_%": smp}).fillna(0).round(1)
        print(f"--- {name} (% of tasks carrying the label; columns do not sum to 100) ---")
        print(comp.sort_values("population_%", ascending=False), "\n")

    marginal("risk_labels", RISK_COL)
    marginal("tool_labels", TOOLS_COL)
    comp = pd.DataFrame({
        "population_%": df["intent_label"].value_counts(normalize=True) * 100,
        "sample_%": sample["intent_label"].value_counts(normalize=True) * 100,
    }).fillna(0).round(1)
    print(f"--- {INTENT_COL} ---")
    print(comp.sort_values("population_%", ascending=False))


if __name__ == "__main__":
    main()
