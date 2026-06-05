"""Aggregate per-task results and emit summary.json + report.md.

Supports pass@k: when --runs K > 1, each task has K result files suffixed
_r0, _r1, ..., _r{K-1}. Aggregations are computed across all runs and
pass@k metrics (k=1..K) are included in the output.

Files written:
  - summary.json   machine-readable, fully structured
  - report.md      human-readable Markdown
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _safe_load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _load_records(result_dir: Path, tasks: list[dict], runs: int) -> list[dict]:
    """Load per-task records, one entry per (task, run_idx) pair.

    Each record has:
      task_id, domain, sites, run_idx, agent (dict), verify (dict)

    When runs=1 the old suffix-free filenames are tried first for backwards
    compatibility, then _r0.
    """
    records: list[dict] = []
    for task in tasks:
        tid = task["task_id"]
        for j in range(runs):
            suffix = f"_r{j}"
            # backwards-compat: runs=1 may have been written without suffix
            agent = (
                _safe_load(result_dir / f"{tid}{suffix}.json")
                or (j == 0 and _safe_load(result_dir / f"{tid}.json"))
                or {"task_id": tid, "status": "missing", "trajectory": [], "agent_output": ""}
            )
            verify = (
                _safe_load(result_dir / f"{tid}{suffix}_verify.json")
                or (j == 0 and _safe_load(result_dir / f"{tid}_verify.json"))
                or {"task_id": tid, "status": "SKIP", "score": 0.0, "checks": []}
            )
            records.append({
                "task_id": tid,
                "domain":  task.get("category_id", "UNKNOWN"),
                "sites":   task.get("meta", {}).get("meta_data", {}).get("sites", []),
                "run_idx": j,
                "agent":   agent,
                "verify":  verify,
            })
    return records


def _group_by_task(records: list[dict]) -> dict[str, list[dict]]:
    """Group records by task_id, preserving run order."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        grouped[r["task_id"]].append(r)
    for runs in grouped.values():
        runs.sort(key=lambda r: r["run_idx"])
    return dict(grouped)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
    return s[k]


def _step_stats_dict(steps: list[int]) -> dict:
    if not steps:
        return {"avg": 0.0, "median": 0, "p90": 0, "max": 0, "min": 0}
    return {
        "avg":    round(statistics.mean(steps), 1),
        "median": int(statistics.median(steps)),
        "p90":    int(_percentile(steps, 0.9)),
        "max":    int(max(steps)),
        "min":    int(min(steps)),
    }


_LLM_COUNT_KEYS = ("llm_call_count", "usage_call_count", "priced_call_count", "unpriced_call_count")
_LLM_TOKEN_KEYS = (
    "prompt_tokens",
    "prompt_cached_tokens",
    "prompt_cache_creation_tokens",
    "prompt_image_tokens",
    "completion_tokens",
    "total_tokens",
)


def _safe_int(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _new_llm_bucket() -> dict:
    bucket = {key: 0 for key in _LLM_COUNT_KEYS + _LLM_TOKEN_KEYS}
    bucket.update({
        "spending_usd": 0.0,
        "priced_task_runs": 0,
        "unpriced_task_runs": 0,
    })
    return bucket


def _llm_usage(records: list[dict]) -> dict:
    summary = _new_llm_bucket()
    summary["task_runs"] = len(records)
    summary["missing_stats_task_runs"] = 0
    by_model: dict[str, dict] = defaultdict(_new_llm_bucket)

    for record in records:
        agent = record.get("agent", {})
        stats = agent.get("llm_stats") if isinstance(agent, dict) else None
        if not isinstance(stats, dict):
            summary["missing_stats_task_runs"] += 1
            continue

        model = str(stats.get("model") or "unknown")
        model_bucket = by_model[model]
        for key in _LLM_COUNT_KEYS + _LLM_TOKEN_KEYS:
            value = _safe_int(stats.get(key))
            summary[key] += value
            model_bucket[key] += value

        spending = _safe_float(stats.get("spending_usd"))
        if spending is not None:
            summary["spending_usd"] += spending
            summary["priced_task_runs"] += 1
            model_bucket["spending_usd"] += spending
            model_bucket["priced_task_runs"] += 1
        elif _safe_int(stats.get("usage_call_count")) > 0:
            summary["unpriced_task_runs"] += 1
            model_bucket["unpriced_task_runs"] += 1

    for bucket in [summary, *by_model.values()]:
        bucket["spending_usd"] = round(bucket["spending_usd"], 6)
        bucket["spending_is_partial"] = (
            bucket.get("unpriced_task_runs", 0) > 0
            or (
                bucket is summary
                and summary.get("missing_stats_task_runs", 0) > 0
            )
        )

    summary["by_model"] = dict(sorted(by_model.items()))
    return summary


def summarize_llm_usage_from_files(
    result_dir: Path,
    task_ids: list[str],
    run_indices: list[int],
) -> dict:
    """Aggregate LLM stats from agent result JSON files."""
    records: list[dict] = []
    for tid in task_ids:
        for run_idx in run_indices:
            suffix = f"_r{run_idx}"
            agent = (
                _safe_load(result_dir / f"{tid}{suffix}.json")
                or (run_idx == 0 and _safe_load(result_dir / f"{tid}.json"))
                or {"task_id": tid, "status": "missing", "trajectory": [], "agent_output": ""}
            )
            records.append({"agent": agent})
    return _llm_usage(records)


# ---------------------------------------------------------------------------
# pass@k and per-task aggregation
# ---------------------------------------------------------------------------

def _task_passk(task_runs: list[dict]) -> dict:
    """Compute all pass@k metrics for a single task across its runs.

    Returns a dict with:
      runs           : number of runs available
      pass_at_k      : {k: bool}  — did at least 1 of first-k runs pass?
      first_pass_idx : int or None — 0-based index of first passing run
      scores         : [float]    per-run scores
      best_score     : float
      mean_score     : float
      steps          : [int]      per-run step counts
      avg_steps      : float
      min_steps      : int        steps used by best-scoring run
      checkpoint_score: float     union score across runs (each check: any pass)
      agent_completed_any: bool
    """
    n = len(task_runs)
    scores = [float(r["verify"].get("score", 0.0)) for r in task_runs]
    steps  = [len(r["agent"].get("trajectory", [])) for r in task_runs]

    # pass@k: at least one of first k runs has all_pass=True
    pass_at_k: dict[int, bool] = {}
    first_pass_idx: int | None = None
    for j, r in enumerate(task_runs):
        if r["verify"].get("all_pass") or r["verify"].get("status") == "PASS":
            if first_pass_idx is None:
                first_pass_idx = j
    for k in range(1, n + 1):
        pass_at_k[k] = first_pass_idx is not None and first_pass_idx < k

    # checkpoint-wise union score: for each check label, passed if any run passed
    check_union: dict[str, dict] = {}  # label -> {weight, passed_any}
    for r in task_runs:
        for chk in r["verify"].get("checks", []):
            lbl = chk.get("label", "")
            if lbl not in check_union:
                check_union[lbl] = {"weight": chk.get("weight", 1), "passed": False}
            if chk.get("passed"):
                check_union[lbl]["passed"] = True
    if check_union:
        total_w  = sum(v["weight"] for v in check_union.values())
        earned_w = sum(v["weight"] for v in check_union.values() if v["passed"])
        checkpoint_score = earned_w / total_w if total_w else 0.0
    else:
        checkpoint_score = 0.0

    agent_completed_any = any(
        r["agent"].get("status") == "completed" for r in task_runs
    )

    return {
        "runs":               n,
        "pass_at_k":          pass_at_k,
        "first_pass_idx":     first_pass_idx,
        "scores":             scores,
        "best_score":         max(scores) if scores else 0.0,
        "mean_score":         statistics.mean(scores) if scores else 0.0,
        "steps":              steps,
        "avg_steps":          statistics.mean(steps) if steps else 0.0,
        "min_steps":          min(steps) if steps else 0,
        "checkpoint_score":   checkpoint_score,
        "agent_completed_any": agent_completed_any,
    }


# ---------------------------------------------------------------------------
# Dataset-level aggregations
# ---------------------------------------------------------------------------

def _overall(task_aggs: dict[str, dict], runs: int) -> dict:
    """Aggregate pass@k and score metrics across all tasks."""
    total = len(task_aggs)
    if total == 0:
        return {}

    # pass@k counts
    pass_at_k_counts: dict[int, int] = {k: 0 for k in range(1, runs + 1)}
    for agg in task_aggs.values():
        for k, passed in agg["pass_at_k"].items():
            if passed:
                pass_at_k_counts[k] += 1

    best_scores  = [agg["best_score"]        for agg in task_aggs.values()]
    mean_scores  = [agg["mean_score"]        for agg in task_aggs.values()]
    ck_scores    = [agg["checkpoint_score"]  for agg in task_aggs.values()]
    avg_steps_l  = [agg["avg_steps"]         for agg in task_aggs.values()]

    # first_pass distribution
    first_pass_dist: dict[str, int] = {f"r{j}": 0 for j in range(runs)}
    first_pass_dist["never"] = 0
    for agg in task_aggs.values():
        idx = agg["first_pass_idx"]
        if idx is None:
            first_pass_dist["never"] += 1
        else:
            first_pass_dist[f"r{idx}"] = first_pass_dist.get(f"r{idx}", 0) + 1

    agent_ok = sum(1 for agg in task_aggs.values() if agg["agent_completed_any"])

    # score buckets (based on best_score)
    perfect = sum(1 for s in best_scores if s >= 1.0)
    zero    = sum(1 for s in best_scores if s <= 0.0)
    partial = total - perfect - zero

    out: dict = {
        "total":   total,
        "runs":    runs,
        "agent_completed_any": agent_ok,
        "pass_at_k": {
            k: {
                "count":  pass_at_k_counts[k],
                "rate":   round(pass_at_k_counts[k] / total, 4),
            }
            for k in range(1, runs + 1)
        },
        "avg_best_score":        round(statistics.mean(best_scores), 4),
        "avg_mean_score":        round(statistics.mean(mean_scores), 4),
        "avg_checkpoint_score":  round(statistics.mean(ck_scores),   4),
        "median_best_score":     round(statistics.median(best_scores), 4),
        "score_buckets":         {"perfect": perfect, "partial": partial, "zero": zero},
        "avg_steps_per_task":    round(statistics.mean(avg_steps_l), 1),
        "first_pass_distribution": first_pass_dist,
    }
    return out


def _by_domain(task_aggs: dict[str, dict], records: list[dict], runs: int) -> dict:
    # map task_id -> domain
    tid2dom: dict[str, str] = {}
    for r in records:
        tid2dom[r["task_id"]] = r["domain"]

    by: dict[str, list[dict]] = defaultdict(list)
    for tid, agg in task_aggs.items():
        by[tid2dom.get(tid, "UNKNOWN")].append(agg)

    out: dict[str, dict] = {}
    for dom in sorted(by):
        aggs = by[dom]
        n = len(aggs)
        pass_counts = {k: sum(1 for a in aggs if a["pass_at_k"].get(k)) for k in range(1, runs + 1)}
        best_scores = [a["best_score"] for a in aggs]
        ck_scores   = [a["checkpoint_score"] for a in aggs]
        out[dom] = {
            "total":     n,
            "pass_at_k": {k: {"count": pass_counts[k],
                               "rate": round(pass_counts[k] / n, 4)} for k in range(1, runs + 1)},
            "avg_best_score":       round(statistics.mean(best_scores), 4),
            "avg_checkpoint_score": round(statistics.mean(ck_scores),   4),
        }
    return out


def _by_app(task_aggs: dict[str, dict], records: list[dict], runs: int) -> dict:
    tid2sites: dict[str, list[str]] = {}
    for r in records:
        tid2sites[r["task_id"]] = r["sites"]

    by: dict[str, list[dict]] = defaultdict(list)
    for tid, agg in task_aggs.items():
        for site in tid2sites.get(tid, []):
            by[site].append(agg)

    out: dict[str, dict] = {}
    for app in sorted(by):
        aggs = by[app]
        n    = len(aggs)
        pass_counts = {k: sum(1 for a in aggs if a["pass_at_k"].get(k)) for k in range(1, runs + 1)}
        best_scores = [a["best_score"] for a in aggs]
        out[app] = {
            "used_in": n,
            "pass_at_k": {k: {"count": pass_counts[k],
                               "rate": round(pass_counts[k] / n, 4)} for k in range(1, runs + 1)},
            "avg_best_score": round(statistics.mean(best_scores), 4),
        }
    return out


def _step_stats(task_aggs: dict[str, dict], records: list[dict]) -> dict:
    # overall: avg steps per task (averaged over runs)
    avg_steps_per_task = [agg["avg_steps"] for agg in task_aggs.values()]
    all_run_steps      = [len(r["agent"].get("trajectory", [])) for r in records]

    tid2dom: dict[str, str] = {r["task_id"]: r["domain"] for r in records}
    by_dom: dict[str, list[int]] = defaultdict(list)
    for agg_tid, agg in task_aggs.items():
        dom = tid2dom.get(agg_tid, "UNKNOWN")
        by_dom[dom].append(int(agg["avg_steps"]))

    return {
        "per_task_avg":  _step_stats_dict([int(s) for s in avg_steps_per_task]),
        "per_run":       _step_stats_dict(all_run_steps),
        "by_domain":     {dom: _step_stats_dict(by_dom[dom]) for dom in sorted(by_dom)},
    }


_BANNED_ACTION_NAMES = frozenset({
    "evaluate", "screenshot", "take_screenshot", "execute_script", "js",
})


def _action_distribution(records: list[dict]) -> dict:
    counter: Counter = Counter()
    banned:  Counter = Counter()
    nav_per_record:   list[int] = []
    sites_per_record: list[int] = []

    for r in records:
        n_nav = 0
        for step in r["agent"].get("trajectory", []):
            for action in step.get("actions", []) or []:
                if not isinstance(action, dict):
                    continue
                for name in action.keys():
                    counter[name] += 1
                    if name == "navigate":
                        n_nav += 1
                    if name in _BANNED_ACTION_NAMES:
                        banned[name] += 1
        nav_per_record.append(n_nav)
        sites_per_record.append(len(r["sites"]) if r["sites"] else 0)

    nav_avg   = round(statistics.mean(nav_per_record),   2) if nav_per_record   else 0.0
    sites_avg = round(statistics.mean(sites_per_record), 2) if sites_per_record else 0.0
    ratio     = round(nav_avg / sites_avg, 2) if sites_avg else 0.0
    violators = sum(1 for n, s in zip(nav_per_record, sites_per_record) if s > 0 and n > s)

    return {
        "by_action":          dict(counter.most_common()),
        "banned_attempted":   dict(banned),
        "navigate_per_run":   nav_avg,
        "sites_per_task":     sites_avg,
        "nav_to_sites_ratio": ratio,
        "navigate_violators": violators,
    }


def _trajectory_health(records: list[dict], max_steps: int, task_aggs: dict[str, dict]) -> dict:
    n_tasks = len(task_aggs)
    if n_tasks == 0:
        return {}

    total_steps = 0
    error_steps = 0

    # per-run signals
    max_hit_any:   set[str] = set()
    done_pres_any: set[str] = set()
    early_fail_any:set[str] = set()
    empty_out_any: set[str] = set()

    for r in records:
        tid  = r["task_id"]
        traj = r["agent"].get("trajectory", []) or []
        total_steps += len(traj)

        if len(traj) >= max_steps:
            max_hit_any.add(tid)

        any_done = False
        for step in traj:
            for res in (step.get("results", []) or []):
                if res.get("error"):
                    error_steps += 1
                if res.get("is_done"):
                    any_done = True
        if any_done:
            done_pres_any.add(tid)

        first5 = traj[:5]
        if any(res.get("error")
               for step in first5 for res in (step.get("results", []) or [])):
            early_fail_any.add(tid)

        if not (r["agent"].get("agent_output") or "").strip():
            empty_out_any.add(tid)

    return {
        "step_error_rate":    round(error_steps / total_steps, 4) if total_steps else 0.0,
        "max_steps_hit_rate": round(len(max_hit_any)    / n_tasks, 4),
        "done_present_rate":  round(len(done_pres_any)  / n_tasks, 4),
        "early_fail_rate":    round(len(early_fail_any) / n_tasks, 4),
        "empty_output_rate":  round(len(empty_out_any)  / n_tasks, 4),
        "total_steps":        total_steps,
        "total_error_steps":  error_steps,
    }


def _confidence_matrix(records: list[dict]) -> dict:
    """Each (task, run) is one sample."""
    matrix = {
        "done_success_true":  {"verify_PASS": 0, "verify_FAIL": 0, "verify_ERROR_SKIP": 0},
        "done_success_false": {"verify_PASS": 0, "verify_FAIL": 0, "verify_ERROR_SKIP": 0},
        "no_done_action":     {"verify_PASS": 0, "verify_FAIL": 0, "verify_ERROR_SKIP": 0},
    }
    for r in records:
        done_success = None
        for step in r["agent"].get("trajectory", []):
            for action in (step.get("actions", []) or []):
                if isinstance(action, dict) and "done" in action:
                    payload = action["done"]
                    if isinstance(payload, dict):
                        done_success = payload.get("success")
                    break
            if done_success is not None:
                break

        if done_success is True:
            row = "done_success_true"
        elif done_success is False:
            row = "done_success_false"
        else:
            row = "no_done_action"

        v = r["verify"].get("status", "SKIP")
        col = "verify_PASS" if v == "PASS" else ("verify_FAIL" if v == "FAIL" else "verify_ERROR_SKIP")
        matrix[row][col] += 1

    total = len(records)
    over  = (matrix["done_success_true"]["verify_FAIL"]
             + matrix["done_success_true"]["verify_ERROR_SKIP"])
    under = matrix["done_success_false"]["verify_PASS"]
    return {
        "matrix":                matrix,
        "over_confidence_rate":  round(over  / total, 4) if total else 0.0,
        "under_confidence_rate": round(under / total, 4) if total else 0.0,
    }


def _per_task_rows(task_aggs: dict[str, dict], records: list[dict], runs: int) -> list[dict]:
    tid2meta: dict[str, dict] = {}
    for r in records:
        if r["task_id"] not in tid2meta:
            tid2meta[r["task_id"]] = {"domain": r["domain"], "sites": r["sites"]}

    rows = []
    for tid, agg in sorted(task_aggs.items()):
        meta = tid2meta.get(tid, {})
        # best agent_output preview (from best-scoring run)
        best_run_idx = 0
        if agg["scores"]:
            best_run_idx = agg["scores"].index(max(agg["scores"]))
        best_rec = next(
            (r for r in records if r["task_id"] == tid and r["run_idx"] == best_run_idx),
            None,
        )
        out = (best_rec["agent"].get("agent_output") or "") if best_rec else ""
        preview = " ".join(out.split())
        if len(preview) > 80:
            preview = preview[:77] + "..."

        row: dict = {
            "task_id":              tid,
            "domain":               meta.get("domain", ""),
            "sites":                meta.get("sites", []),
            "agent_completed_any":  agg["agent_completed_any"],
            "pass_at_k":            {str(k): v for k, v in agg["pass_at_k"].items()},
            "first_pass_idx":       agg["first_pass_idx"],
            "best_score":           round(agg["best_score"],       3),
            "mean_score":           round(agg["mean_score"],       3),
            "checkpoint_score":     round(agg["checkpoint_score"], 3),
            "scores_per_run":       [round(s, 3) for s in agg["scores"]],
            "steps_per_run":        agg["steps"],
            "avg_steps":            round(agg["avg_steps"], 1),
            "agent_output_preview": preview,
        }
        # flatten pass_at_k for readability when runs=1
        if runs == 1:
            row["pass"] = agg["pass_at_k"].get(1, False)
            row["score"] = round(agg["best_score"], 3)
            row["n_steps"] = agg["steps"][0] if agg["steps"] else 0
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _md_overall(s: dict) -> str:
    o = s["overall"]
    runs = o["runs"]
    lines = [
        "## Overall\n",
        "| Metric | Value |",
        "|--------|------:|",
        f"| Tasks | {o['total']} |",
        f"| Runs per task | {runs} |",
        f"| Agent completed (any run) | {o['agent_completed_any']} / {o['total']} |",
    ]
    for k in range(1, runs + 1):
        pk = o["pass_at_k"][k]
        lines.append(
            f"| **pass@{k}** | **{pk['count']} / {o['total']} ({pk['rate']*100:.1f}%)** |"
        )
    lines += [
        f"| Avg best-of-k score | **{o['avg_best_score']:.3f}** |",
        f"| Avg mean-of-k score | {o['avg_mean_score']:.3f} |",
        f"| Avg checkpoint score (union) | {o['avg_checkpoint_score']:.3f} |",
        f"| Median best score | {o['median_best_score']:.3f} |",
        f"| Perfect / Partial / Zero (best) | {o['score_buckets']['perfect']} / {o['score_buckets']['partial']} / {o['score_buckets']['zero']} |",
        f"| Avg steps per task | {o['avg_steps_per_task']:.1f} |",
    ]
    if runs > 1:
        dist = o["first_pass_distribution"]
        dist_str = "  ".join(f"r{j}:{dist.get(f'r{j}',0)}" for j in range(runs))
        dist_str += f"  never:{dist.get('never',0)}"
        lines.append(f"| First-pass distribution | {dist_str} |")
    return "\n".join(lines) + "\n\n"


def _md_llm_usage(s: dict) -> str:
    usage = s.get("llm_usage", {})
    spending = usage.get("spending_usd", 0.0)
    spending_label = f"${spending:.6f}"
    if usage.get("spending_is_partial"):
        spending_label += " (partial)"
    return (
        "## LLM Usage\n\n"
        "| Metric | Value |\n"
        "|--------|------:|\n"
        f"| LLM calls | {usage.get('llm_call_count', 0)} |\n"
        f"| Calls with token usage | {usage.get('usage_call_count', 0)} |\n"
        f"| Prompt tokens | {usage.get('prompt_tokens', 0)} |\n"
        f"| Completion tokens | {usage.get('completion_tokens', 0)} |\n"
        f"| Total tokens | {usage.get('total_tokens', 0)} |\n"
        f"| Estimated spending | {spending_label} |\n"
        f"| Missing per-task stats | {usage.get('missing_stats_task_runs', 0)} |\n\n"
    )


def _md_by_domain(s: dict) -> str:
    runs = s["overall"]["runs"]
    # build header
    pk_headers = " | ".join(f"pass@{k}" for k in range(1, runs + 1))
    pk_sep     = " | ".join("-------:" for _ in range(1, runs + 1))
    header = (
        f"| Domain | Tasks | {pk_headers} | Avg Best Score | Avg CkPt Score |\n"
        f"|--------|------:|{pk_sep}|---------------:|---------------:|"
    )
    rows = []
    for dom, d in s["by_domain"].items():
        pk_vals = " | ".join(
            f"{d['pass_at_k'][k]['count']} ({d['pass_at_k'][k]['rate']*100:.0f}%)"
            for k in range(1, runs + 1)
        )
        rows.append(
            f"| {dom} | {d['total']} | {pk_vals} | "
            f"{d['avg_best_score']:.3f} | {d['avg_checkpoint_score']:.3f} |"
        )
    return "## By Domain\n\n" + header + "\n" + "\n".join(rows) + "\n\n"


def _md_by_app(s: dict) -> str:
    runs = s["overall"]["runs"]
    pk_headers = " | ".join(f"pass@{k}" for k in range(1, runs + 1))
    pk_sep     = " | ".join("-------:" for _ in range(1, runs + 1))
    header = (
        f"| App | Used in | {pk_headers} | Avg Best Score |\n"
        f"|-----|--------:|{pk_sep}|---------------:|"
    )
    rows = []
    for app, d in s["by_app"].items():
        pk_vals = " | ".join(
            f"{d['pass_at_k'][k]['count']} ({d['pass_at_k'][k]['rate']*100:.0f}%)"
            for k in range(1, runs + 1)
        )
        rows.append(f"| `{app}` | {d['used_in']} | {pk_vals} | {d['avg_best_score']:.3f} |")
    return "## By App\n\n" + header + "\n" + "\n".join(rows) + "\n\n"


def _md_steps(s: dict) -> str:
    rows = []
    o = s["steps"]["per_task_avg"]
    rows.append(f"| **Per-task avg** | {o['avg']:.1f} | {o['median']} | {o['p90']} | {o['max']} |")
    o2 = s["steps"]["per_run"]
    rows.append(f"| Per-run | {o2['avg']:.1f} | {o2['median']} | {o2['p90']} | {o2['max']} |")
    for dom, d in s["steps"]["by_domain"].items():
        rows.append(f"| {dom} (task avg) | {d['avg']:.1f} | {d['median']} | {d['p90']} | {d['max']} |")
    return (
        "## Step Statistics\n\n"
        "| Scope | Avg | Median | p90 | Max |\n"
        "|-------|----:|-------:|----:|----:|\n"
        + "\n".join(rows) + "\n\n"
    )


def _md_actions(s: dict) -> str:
    a = s["actions"]
    total = sum(a["by_action"].values()) or 1
    rows = [
        f"| `{name}` | {cnt} | {cnt / total * 100:.1f}% |"
        for name, cnt in a["by_action"].items()
    ]
    md = (
        "## Action Distribution\n\n"
        "| Action | Count | Share |\n"
        "|--------|------:|------:|\n"
        + "\n".join(rows) + "\n\n"
        f"- **Avg `navigate` per run**: {a['navigate_per_run']:.2f} "
        f"(avg sites/task: {a['sites_per_task']:.2f}; "
        f"ratio: **{a['nav_to_sites_ratio']:.2f}**, ideal ≈ 1.0)\n"
        f"- **Runs violating navigate-discipline** (n_nav > n_sites): "
        f"**{a['navigate_violators']}**\n"
    )
    if a["banned_attempted"]:
        items = ", ".join(f"`{k}`={v}" for k, v in a["banned_attempted"].items())
        md += f"- **Banned actions attempted**: {items}\n"
    else:
        md += "- **Banned actions attempted**: 0 ✓\n"
    return md + "\n"


def _md_health(s: dict) -> str:
    h = s["trajectory_health"]
    return (
        "## Trajectory Health\n\n"
        "| Signal | Rate |\n"
        "|--------|-----:|\n"
        f"| Step error rate ({h['total_error_steps']} / {h['total_steps']} steps) | {h['step_error_rate']*100:.2f}% |\n"
        f"| Max-steps hit (any run) | {h['max_steps_hit_rate']*100:.2f}% |\n"
        f"| `done` present (any run) | {h['done_present_rate']*100:.2f}% |\n"
        f"| Early-fail (any run, error in first 5 steps) | {h['early_fail_rate']*100:.2f}% |\n"
        f"| Empty output (any run) | {h['empty_output_rate']*100:.2f}% |\n\n"
    )


def _md_confidence(s: dict) -> str:
    c = s["confidence"]["matrix"]
    over  = s["confidence"]["over_confidence_rate"]  * 100
    under = s["confidence"]["under_confidence_rate"] * 100
    return (
        "## Self-Confidence Calibration\n\n"
        "Each (task × run) is one sample.\n\n"
        "| Self-report | verify PASS | verify FAIL | verify ERROR/SKIP |\n"
        "|-------------|------------:|------------:|------------------:|\n"
        f"| `done(success=true)` | {c['done_success_true']['verify_PASS']} | "
        f"**{c['done_success_true']['verify_FAIL']}** | "
        f"{c['done_success_true']['verify_ERROR_SKIP']} |\n"
        f"| `done(success=false)` | {c['done_success_false']['verify_PASS']} | "
        f"{c['done_success_false']['verify_FAIL']} | "
        f"{c['done_success_false']['verify_ERROR_SKIP']} |\n"
        f"| no `done` action | {c['no_done_action']['verify_PASS']} | "
        f"{c['no_done_action']['verify_FAIL']} | "
        f"{c['no_done_action']['verify_ERROR_SKIP']} |\n\n"
        f"- **Over-confidence rate**: {over:.2f}%\n"
        f"- **Under-confidence rate**: {under:.2f}%\n\n"
    )


def _md_per_task(s: dict) -> str:
    runs = s["overall"]["runs"]
    pk_headers = " | ".join(f"p@{k}" for k in range(1, runs + 1))
    header = (
        f"| Task ID | Domain | Sites | {pk_headers} | Best | CkPt | Avg Steps | Output Preview |\n"
        f"|---------|--------|-------|"
        + "".join("----:|" for _ in range(runs))
        + "-----:|-----:|----------:|----------------|"
    )
    rows = []
    for t in s["tasks"]:
        sites   = ",".join(t["sites"]) if t["sites"] else "—"
        preview = t["agent_output_preview"].replace("|", "\\|")
        pk_vals = " | ".join(
            "✓" if t["pass_at_k"].get(str(k)) else "✗"
            for k in range(1, runs + 1)
        )
        rows.append(
            f"| {t['task_id']} | {t['domain']} | {sites} | "
            f"{pk_vals} | {t['best_score']:.2f} | {t['checkpoint_score']:.2f} | "
            f"{t['avg_steps']:.0f} | {preview} |"
        )
    return (
        "## Per-task Results\n\n"
        f"<details><summary>Click to expand all {len(s['tasks'])} task results</summary>\n\n"
        + header + "\n" + "\n".join(rows) + "\n\n"
        "</details>\n"
    )


def _human_duration(seconds: float) -> str:
    s = int(seconds)
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _render_report(summary: dict) -> str:
    r   = summary["run"]
    iso = "enabled" if summary["isolation"] else "disabled"
    runs = summary["overall"]["runs"]
    header = (
        "# SaaS-Bench Eval Report\n\n"
        f"**Model:** `{summary['model']}` • "
        f"**Workers:** {summary['workers']} • "
        f"**Runs (pass@k):** {runs} • "
        f"**Isolation:** {iso} • "
        f"**Max steps:** {summary['max_steps']}\n"
        f"**Started:** {r['started_at']} • "
        f"**Duration:** {r['duration_human']} • "
        f"**Tasks:** {summary['overall']['total']}\n\n"
        "---\n\n"
    )
    body = (
        _md_overall(summary)
        + _md_llm_usage(summary)
        + _md_by_domain(summary)
        + _md_by_app(summary)
        + _md_steps(summary)
        + _md_actions(summary)
        + _md_health(summary)
        + _md_confidence(summary)
        + _md_per_task(summary)
    )
    footer = "\n---\n\n_Auto-generated from `summary.json` by saas-bench._\n"
    return header + body + footer


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_outputs(
    result_dir: Path,
    tasks: list[dict],
    run_meta: dict,
    max_steps: int,
) -> tuple[Path, Path]:
    """Build aggregations and write summary.json + report.md to result_dir."""
    runs = int(run_meta.get("runs", 1))
    records = _load_records(result_dir, tasks, runs)
    task_aggs = {tid: _task_passk(runs_list)
                 for tid, runs_list in _group_by_task(records).items()}

    summary = {
        "tasks_dir": run_meta.get("tasks_dir"),
        "model":     run_meta.get("model"),
        "workers":   run_meta.get("workers"),
        "hostname":  run_meta.get("hostname"),
        "isolation": run_meta.get("isolation"),
        "max_steps": max_steps,
        "run": {
            "started_at":     run_meta.get("started_at"),
            "ended_at":       run_meta.get("ended_at"),
            "duration_s":     run_meta.get("duration_s"),
            "duration_human": _human_duration(run_meta.get("duration_s") or 0),
        },
        "overall":           _overall(task_aggs, runs),
        "llm_usage":         _llm_usage(records),
        "by_domain":         _by_domain(task_aggs, records, runs),
        "by_app":            _by_app(task_aggs, records, runs),
        "steps":             _step_stats(task_aggs, records),
        "actions":           _action_distribution(records),
        "trajectory_health": _trajectory_health(records, max_steps, task_aggs),
        "confidence":        _confidence_matrix(records),
        "tasks":             _per_task_rows(task_aggs, records, runs),
    }

    summary_path = result_dir / "summary.json"
    report_path  = result_dir / "report.md"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    report_path.write_text(_render_report(summary))
    return summary_path, report_path
