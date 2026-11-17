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
"""
from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd

# ------------------------- configuration -------------------------
INPUT_CSV = "HF_OpenAgentSafety_tasks.csv"
OUTPUT_CSV = "sample_100.csv"
OUTPUT_ALLOCATION_CSV = "sample_100_leaf_allocation.csv"
FILTER_COL = "has_usable_gpt4o_run"
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
# ------------------------------------------------------------------


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
    df = pd.read_csv(INPUT_CSV)

    # 1) keep only usable GPT-4o runs
    df = df[df[FILTER_COL].astype(str).str.strip().str.lower() == "yes"].copy()
    print(f"Rows with {FILTER_COL} = yes: {len(df)}")

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
