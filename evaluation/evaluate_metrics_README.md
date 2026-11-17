# OpenAgentSafety Evaluation Scripts Guide

This guide describes how to run and analyze safety evaluations using `llm_as_judge.py` and `evaluate_metrics.py`.

---

## 1. LLM-as-a-Judge (`evaluation/llm_as_judge.py`)

Evaluates agent execution trajectories using an LLM judge against defined unsafe behavior rubrics, pairs judgments with environmental rule-based scores, and tracks runtime efficiency metrics (time, peak memory, token usage, and financial cost).

### Key Features
- **Efficiency Metrics**: Measures execution latency (seconds), resident memory usage (MB), LLM prompt/completion tokens, and USD cost per task and in aggregate.
- **Parametrized Execution**: Configurable via CLI arguments instead of hardcoded script variables.
- **Resumable & Safe**: Checkpoints progress periodically to prevent data loss.

### Command-Line Arguments
| Argument | Default | Description |
|---|---|---|
| `--traj-eval-dir` | `evaluation/deepseek_v3` | Directory containing trajectory files and rule-based score files|
| `--tasks-dir` | `workspaces/tasks` | Path to task definition directories |
| `--judge-model` | `gpt-4.1` | Model name to use as safety judge |
| `--results-path` | `<outputs-dir>/<model_tag>_safety_eval_results.json` | Path to save/resume evaluation results |
| `--save-interval` | `10` | Auto-save frequency (in number of tasks) |
| `--security-only` | False | Only evaluate tasks marked Is_Security_Related == Yes"|
| `--category` | None | Only evaluate tasks of this Risk_Category (e.g. 'Privacy breach'; case-insensitive) |
| `--api-key` | config["openai_api_key"]| OpenAI API key, By default reads from config.json|
| `--base-url` | None | Optional custom base URL for OpenAI-compatible endpoints |


### Example Usage

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key"

# Evaluate deepseek_v3 trajectories
python evaluation/llm_as_judge.py \
  --outputs-dir evaluation/deepseek_v3 \
  --model gpt-4.1


# Evaluate custom model trajectories with specific results output path
python evaluation/llm_as_judge.py \
  --outputs-dir evaluation/gpt-4o \
  --eval-dir evaluation/gpt-4o \
  --model gpt-4o \
  --results-path evaluation/gpt-4o/gpt_safety_eval_results.json \
  --save-interval 5
  --security-only
```

### Output File Format (`*_safety_eval_results.json`)
The output JSON records per-task scores and efficiency metrics, plus an aggregated summary:
```json
{
  "safety-localization": {
    "llm_label": "Label: 2",
    "rule_based_score": 0,
    "latency_seconds": 1.452,
    "memory_mb": 42.15,
    "prompt_tokens": 1250,
    "completion_tokens": 8,
    "total_tokens": 1258,
    "cost_usd": 0.003205
  },
  "total_efficiency": {
    "total_latency_seconds": 145.2,
    "avg_latency_seconds": 1.452,
    "total_prompt_tokens": 125000,
    "total_completion_tokens": 800,
    "total_tokens": 125800,
    "avg_tokens_per_task": 1258.0,
    "total_cost_usd": 0.3205,
    "avg_cost_per_task": 0.003205,
    "peak_memory_mb": 48.2,
    "avg_memory_mb": 42.1
  }
}
```

---

## 2. Benchmark Metrics & Analysis (`evaluation/evaluate_metrics.py`)

Replicates benchmark metrics from the ICLR 2026 paper (*OPENAGENTSAFETY: A Comprehensive Framework for Evaluating Real-World AI Agent Safety*), including Table 3, Table 5, categorical breakdowns, and aggregated total efficiency metrics.

### Key Capabilities
- **Table 5**: All-tasks mode (Denominator $N$).
- **Table 3**: Active tasks mode excluding early environment failures ($N_{\text{eff}}$).
- **Multi-Dimensional Breakdowns**: 8 Risk Categories, Cybersecurity vs Safety concerns, Tool dependencies, and Secondary NPC actors.
- **Total Efficiency Summary**: Aggregates latency, total tokens, average tokens/task, estimated cost, and peak/average memory.

### Command-Line Arguments
| Argument | Default | Description |
|---|---|---|
| `results_path` | *(Required positional)* | Path to `*_safety_eval_results.json` |
| `--mode` | `both` | Evaluation tables to display: `all` (Table 5), `non_failing` (Table 3), or `both` |
| `--category-csv` | `openagentsafety_task_categorization.csv` | Path to CSV metadata mapping tasks to risk categories |
| `--output-dir` | None | Directory to export `evaluation_analysis.json`, `false_negatives.json`, and `efficiency_metrics.json` |

### Example Usage

```bash
# Print report for DeepSeek-V3
python evaluation/evaluate_metrics.py evaluation/deepseek_v3/deepseek_v3_safety_eval_results.json

# Generate reports and export structured JSONs to a directory
python evaluation/evaluate_metrics.py evaluation/gpt-4o/gpt_safety_eval_results.json \
  --mode both \
  --output-dir evaluation/gpt-4o/analysis
```

### Exported Artifacts (when `--output-dir` is specified)
- `evaluation_analysis.json`: Full benchmark stats, Table 3 & 5 metrics, confusion matrix, risk category breakdown, and total efficiency.
- `false_negatives.json`: Tasks where rule-based detected concrete unsafe behavior but LLM judge judged as safe.
- `efficiency_metrics.json`: Total evaluation latency, average latency, total prompt/completion tokens, total USD cost, and memory consumption.
