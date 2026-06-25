#!/usr/bin/env python3
"""Score one or more SaaS-Bench result directories from current artifacts.

The harness writes per-task agent files:
  <task_id>_r0.json, <task_id>_r1.json, ...

and verifier files:
  <task_id>_r0_verify.json, <task_id>_r1_verify.json, ...

This script intentionally scores from the per-task verifier files that exist
right now, rather than trusting summary.json, so it is useful for partial or
resumed experiments.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IGNORE_JSON_NAMES = {"summary.json"}


@dataclass(frozen=True)
class ArtifactKey:
    task_id: str
    run_idx: int


@dataclass
class RunRecord:
    task_id: str
    run_idx: int
    verify: dict[str, Any] | None
    agent: dict[str, Any] | None

    @property
    def has_verify(self) -> bool:
        return self.verify is not None

    @property
    def score(self) -> float:
        if not self.verify:
            return 0.0
        return safe_float(self.verify.get("score"), 0.0)

    @property
    def passed(self) -> bool:
        if not self.verify:
            return False
        return bool(self.verify.get("all_pass") or self.verify.get("status") == "PASS")

    @property
    def steps(self) -> int:
        if not self.agent:
            return 0
        trajectory = self.agent.get("trajectory")
        return len(trajectory) if isinstance(trajectory, list) else 0


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def load_task_ids_file(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Task ids file does not exist: {path}")

    task_ids: list[str] = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        task_ids.extend(line.replace(",", " ").split())
    return task_ids


def parse_verify_name(path: Path) -> ArtifactKey | None:
    stem = path.stem
    if not stem.endswith("_verify"):
        return None

    base = stem[: -len("_verify")]
    task_id, run_idx = split_run_suffix(base)
    return ArtifactKey(task_id=task_id, run_idx=run_idx)


def parse_agent_name(path: Path) -> ArtifactKey | None:
    if path.name in IGNORE_JSON_NAMES or path.stem.endswith("_verify"):
        return None

    task_id, run_idx = split_run_suffix(path.stem)
    return ArtifactKey(task_id=task_id, run_idx=run_idx)


def split_run_suffix(stem: str) -> tuple[str, int]:
    head, sep, tail = stem.rpartition("_r")
    if sep and tail.isdigit() and head:
        return head, int(tail)
    return stem, 0


def has_result_artifacts(path: Path) -> bool:
    if not path.is_dir():
        return False
    for child in path.iterdir():
        if child.suffix != ".json":
            continue
        if parse_verify_name(child) or parse_agent_name(child):
            return True
    return False


def expand_experiment_dirs(paths: list[Path | str]) -> list[Path]:
    expanded: list[Path] = []
    seen: set[Path] = set()

    for raw_path in paths:
        path = Path(raw_path)
        if has_result_artifacts(path):
            candidates = [path]
        elif path.is_dir():
            candidates = [
                child for child in sorted(path.iterdir())
                if child.is_dir() and has_result_artifacts(child)
            ]
        else:
            candidates = [path]

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                expanded.append(candidate)

    return expanded


def metadata_from_summary(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not summary:
        return {}
    task_meta = {}
    for task in summary.get("tasks") or []:
        if isinstance(task, dict) and task.get("task_id"):
            task_meta[str(task["task_id"])] = {
                "domain": task.get("domain") or infer_domain(str(task["task_id"])),
                "sites": task.get("sites") or [],
            }
    return {
        "model": summary.get("model"),
        "tasks_dir": summary.get("tasks_dir"),
        "run": summary.get("run"),
        "summary_total": (summary.get("overall") or {}).get("total"),
        "task_meta": task_meta,
    }


def infer_domain(task_id: str) -> str:
    prefix = task_id.rsplit("_", 1)[0]
    if not prefix:
        return "UNKNOWN"
    return prefix.replace("-", " ").replace("_", " ").title()


def llm_usage(agent_records: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "llm_call_count": 0,
        "usage_call_count": 0,
        "priced_call_count": 0,
        "unpriced_call_count": 0,
        "prompt_tokens": 0,
        "prompt_cached_tokens": 0,
        "prompt_cache_creation_tokens": 0,
        "prompt_image_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "spending_usd": 0.0,
        "priced_task_runs": 0,
        "unpriced_task_runs": 0,
        "missing_stats_task_runs": 0,
    }
    by_model: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            key: 0 for key in totals
            if key not in {"spending_usd", "missing_stats_task_runs"}
        }
    )

    for agent in agent_records:
        stats = agent.get("llm_stats") if isinstance(agent, dict) else None
        if not isinstance(stats, dict):
            totals["missing_stats_task_runs"] += 1
            continue

        model = str(stats.get("model") or "unknown")
        bucket = by_model[model]
        for key in (
            "llm_call_count",
            "usage_call_count",
            "priced_call_count",
            "unpriced_call_count",
            "prompt_tokens",
            "prompt_cached_tokens",
            "prompt_cache_creation_tokens",
            "prompt_image_tokens",
            "completion_tokens",
            "total_tokens",
        ):
            value = safe_int(stats.get(key))
            totals[key] += value
            bucket[key] += value

        spending = stats.get("spending_usd")
        if spending is None:
            if safe_int(stats.get("usage_call_count")) > 0:
                totals["unpriced_task_runs"] += 1
                bucket["unpriced_task_runs"] += 1
        else:
            value = safe_float(spending)
            totals["spending_usd"] += value
            totals["priced_task_runs"] += 1
            bucket["spending_usd"] = safe_float(bucket.get("spending_usd")) + value
            bucket["priced_task_runs"] += 1

    totals["task_runs"] = len(agent_records)
    totals["spending_usd"] = round(totals["spending_usd"], 6)
    totals["spending_is_partial"] = (
        totals["unpriced_task_runs"] > 0 or totals["missing_stats_task_runs"] > 0
    )
    for bucket in by_model.values():
        bucket["spending_usd"] = round(safe_float(bucket.get("spending_usd")), 6)
    totals["by_model"] = dict(sorted(by_model.items()))
    return totals


def checkpoint_score(records: list[RunRecord]) -> float:
    checks: dict[str, dict[str, Any]] = {}
    for record in records:
        verify = record.verify or {}
        for check in verify.get("checks") or []:
            if not isinstance(check, dict):
                continue
            label = str(check.get("label") or "")
            if not label:
                continue
            entry = checks.setdefault(
                label,
                {"weight": safe_float(check.get("weight"), 1.0), "passed": False},
            )
            if check.get("passed"):
                entry["passed"] = True

    total = sum(safe_float(check["weight"], 1.0) for check in checks.values())
    if total <= 0:
        return 0.0
    earned = sum(
        safe_float(check["weight"], 1.0)
        for check in checks.values()
        if check["passed"]
    )
    return earned / total


def score_experiment(
    path: Path,
    count_missing_as_zero: bool = False,
    task_ids: list[str] | None = None,
) -> dict[str, Any]:
    verify_by_key: dict[ArtifactKey, dict[str, Any]] = {}
    agent_by_key: dict[ArtifactKey, dict[str, Any]] = {}
    unreadable: list[str] = []
    task_id_filter = set(task_ids or [])

    for child in sorted(path.iterdir()) if path.is_dir() else []:
        if child.suffix != ".json":
            continue
        verify_key = parse_verify_name(child)
        agent_key = parse_agent_name(child)
        if not verify_key and not agent_key:
            continue
        artifact_key = verify_key or agent_key
        if task_id_filter and artifact_key and artifact_key.task_id not in task_id_filter:
            continue

        data = load_json(child)
        if data is None:
            unreadable.append(str(child))
            continue
        if verify_key:
            verify_by_key[verify_key] = data
        elif agent_key:
            agent_by_key[agent_key] = data

    keys = sorted(
        set(verify_by_key) | set(agent_by_key),
        key=lambda key: (key.task_id, key.run_idx),
    )
    records = [
        RunRecord(
            task_id=key.task_id,
            run_idx=key.run_idx,
            verify=verify_by_key.get(key),
            agent=agent_by_key.get(key),
        )
        for key in keys
    ]
    discovered_task_ids = {record.task_id for record in records}

    summary = load_json(path / "summary.json") if path.is_dir() else None
    meta = metadata_from_summary(summary)
    grouped: dict[str, list[RunRecord]] = defaultdict(list)
    for record in records:
        grouped[record.task_id].append(record)

    task_rows: list[dict[str, Any]] = []
    for task_id in sorted(grouped):
        task_records = sorted(grouped[task_id], key=lambda record: record.run_idx)
        scoring_records = (
            task_records if count_missing_as_zero
            else [record for record in task_records if record.has_verify]
        )
        scores = [record.score for record in scoring_records]
        best_score = max(scores) if scores else 0.0
        mean_score = statistics.mean(scores) if scores else 0.0
        pass_any = any(record.passed for record in scoring_records)
        first_pass_idx = next(
            (record.run_idx for record in scoring_records if record.passed),
            None,
        )
        max_run_idx = max((record.run_idx for record in task_records), default=0)
        pass_at_k = {
            str(k): any(record.passed and record.run_idx < k for record in scoring_records)
            for k in range(1, max_run_idx + 2)
        }

        task_meta = (meta.get("task_meta") or {}).get(task_id, {})
        task_rows.append({
            "task_id": task_id,
            "domain": task_meta.get("domain") or infer_domain(task_id),
            "sites": task_meta.get("sites") or [],
            "runs_seen": [record.run_idx for record in task_records],
            "verified_runs": [record.run_idx for record in task_records if record.has_verify],
            "missing_verify_runs": [
                record.run_idx for record in task_records
                if record.agent is not None and not record.has_verify
            ],
            "pass": pass_any,
            "first_pass_idx": first_pass_idx,
            "pass_at_k": pass_at_k,
            "best_score": round(best_score, 4),
            "mean_score": round(mean_score, 4),
            "checkpoint_score": round(checkpoint_score(scoring_records), 4),
            "runs": [run_to_dict(record) for record in task_records],
        })

    scored_tasks = [
        task for task in task_rows
        if task["verified_runs"] or count_missing_as_zero
    ]
    best_scores = [task["best_score"] for task in scored_tasks]
    mean_scores = [task["mean_score"] for task in scored_tasks]
    checkpoint_scores = [task["checkpoint_score"] for task in scored_tasks]
    max_k = max(
        (max((int(k) for k in task["pass_at_k"]), default=0) for task in scored_tasks),
        default=0,
    )
    pass_at_k = {}
    for k in range(1, max_k + 1):
        count = sum(1 for task in scored_tasks if task["pass_at_k"].get(str(k)))
        total = len(scored_tasks)
        pass_at_k[str(k)] = {
            "count": count,
            "total": total,
            "rate": round(count / total, 4) if total else 0.0,
        }

    status_counts = Counter(
        str(record.verify.get("status") if record.verify else "MISSING_VERIFY")
        for record in records
    )
    agent_status_counts = Counter(
        str(record.agent.get("status") if record.agent else "MISSING_AGENT")
        for record in records
    )
    by_domain: dict[str, dict[str, Any]] = {}
    for domain, rows in sorted(group_by(scored_tasks, "domain").items()):
        domain_scores = [row["best_score"] for row in rows]
        domain_pass = sum(1 for row in rows if row["pass"])
        by_domain[domain] = {
            "tasks": len(rows),
            "pass": domain_pass,
            "pass_rate": round(domain_pass / len(rows), 4) if rows else 0.0,
            "avg_best_score": round(statistics.mean(domain_scores), 4)
            if domain_scores else 0.0,
        }

    agent_records = [record.agent for record in records if record.agent is not None]
    overall = {
        "tasks_discovered": len(task_rows),
        "tasks_scored": len(scored_tasks),
        "task_runs_discovered": len(records),
        "verified_task_runs": sum(1 for record in records if record.has_verify),
        "agent_task_runs": sum(1 for record in records if record.agent is not None),
        "missing_verify_task_runs": sum(
            1 for record in records
            if record.agent is not None and not record.has_verify
        ),
        "missing_agent_task_runs": sum(
            1 for record in records
            if record.has_verify and record.agent is None
        ),
        "pass": sum(1 for task in scored_tasks if task["pass"]),
        "pass_rate": round(
            sum(1 for task in scored_tasks if task["pass"]) / len(scored_tasks), 4
        ) if scored_tasks else 0.0,
        "pass_at_k": pass_at_k,
        "avg_best_score": round(statistics.mean(best_scores), 4) if best_scores else 0.0,
        "avg_mean_score": round(statistics.mean(mean_scores), 4) if mean_scores else 0.0,
        "avg_checkpoint_score": round(statistics.mean(checkpoint_scores), 4)
        if checkpoint_scores else 0.0,
        "median_best_score": round(statistics.median(best_scores), 4)
        if best_scores else 0.0,
        "score_buckets": score_buckets(best_scores),
        "status_counts": dict(sorted(status_counts.items())),
        "agent_status_counts": dict(sorted(agent_status_counts.items())),
        "count_missing_as_zero": count_missing_as_zero,
    }

    return {
        "path": str(path),
        "model": meta.get("model") or path.name,
        "tasks_dir": meta.get("tasks_dir"),
        "summary_total": meta.get("summary_total"),
        "overall": overall,
        "by_domain": by_domain,
        "llm_usage": llm_usage(agent_records),
        "tasks": task_rows,
        "selected_task_ids": task_ids if task_id_filter else None,
        "missing_selected_task_ids": [
            task_id for task_id in (task_ids or [])
            if task_id not in discovered_task_ids
        ] if task_id_filter else [],
        "unreadable_json": unreadable,
    }


def run_to_dict(record: RunRecord) -> dict[str, Any]:
    verify = record.verify or {}
    agent = record.agent or {}
    checks = verify.get("checks") if isinstance(verify.get("checks"), list) else []
    failed = [
        {
            "label": str(check.get("label") or ""),
            "weight": safe_float(check.get("weight"), 1.0),
            "detail": str(check.get("detail") or ""),
        }
        for check in checks
        if isinstance(check, dict) and not check.get("passed")
    ]
    passed_checks = sum(
        1 for check in checks
        if isinstance(check, dict) and check.get("passed")
    )
    return {
        "run_idx": record.run_idx,
        "verify_status": verify.get("status") if record.verify else "MISSING_VERIFY",
        "score": round(record.score, 4),
        "earned": verify.get("earned"),
        "total": verify.get("total"),
        "all_pass": record.passed,
        "checks_passed": passed_checks,
        "checks_total": len(checks),
        "failed_checks": failed,
        "agent_status": agent.get("status") if record.agent else "MISSING_AGENT",
        "steps": record.steps,
        "agent_error": agent.get("error"),
        "verify_error": verify.get("error") if record.verify else None,
    }


def score_buckets(scores: list[float]) -> dict[str, int]:
    perfect = sum(1 for score in scores if score >= 1.0)
    zero = sum(1 for score in scores if score <= 0.0)
    partial = len(scores) - perfect - zero
    return {"perfect": perfect, "partial": partial, "zero": zero}


def group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "UNKNOWN")].append(row)
    return grouped


def checks_for_output(run: dict[str, Any], mode: str) -> str:
    if mode == "none":
        return ""
    failed = run["failed_checks"]
    if mode == "failed":
        if not failed:
            return "failed=0"
        return "; ".join(
            format_check(check)
            for check in failed
        )
    checks_passed = run["checks_passed"]
    checks_total = run["checks_total"]
    if not failed:
        return f"{checks_passed}/{checks_total} checks passed"
    return (
        f"{checks_passed}/{checks_total} checks passed; failed: "
        + "; ".join(format_check(check) for check in failed)
    )


def format_check(check: dict[str, Any]) -> str:
    label = str(check.get("label") or "").replace("\n", " ").strip()
    detail = str(check.get("detail") or "").replace("\n", " ").strip()
    if len(detail) > 80:
        detail = detail[:77] + "..."
    if detail:
        return f"{label} ({detail})"
    return label


def format_text(results: list[dict[str, Any]], checks: str) -> str:
    parts: list[str] = []
    for result in results:
        overall = result["overall"]
        usage = result["llm_usage"]
        parts.append(f"=== {result['path']} ===")
        parts.append(f"model: {result['model']}")
        if result.get("tasks_dir"):
            parts.append(f"summary_tasks_dir: {result['tasks_dir']}")
        if result.get("summary_total") is not None:
            current = result["overall"]["tasks_discovered"]
            suffix = " (differs from current artifacts)" if result["summary_total"] != current else ""
            parts.append(f"existing_summary_total: {result['summary_total']}{suffix}")
        if result.get("selected_task_ids"):
            missing = result.get("missing_selected_task_ids") or []
            suffix = f" | missing={len(missing)}" if missing else ""
            parts.append(
                f"selected_tasks: {len(result['selected_task_ids'])} requested{suffix}"
            )

        parts.append(
            "tasks: "
            f"{overall['tasks_scored']} scored / {overall['tasks_discovered']} discovered | "
            f"runs: {overall['verified_task_runs']} verified / "
            f"{overall['agent_task_runs']} agent | "
            f"missing_verify={overall['missing_verify_task_runs']} "
            f"missing_agent={overall['missing_agent_task_runs']}"
        )
        parts.append(
            "score: "
            f"pass={overall['pass']}/{overall['tasks_scored']} "
            f"({overall['pass_rate'] * 100:.1f}%) | "
            f"avg_best={overall['avg_best_score']:.4f} | "
            f"avg_mean={overall['avg_mean_score']:.4f} | "
            f"avg_checkpoint={overall['avg_checkpoint_score']:.4f} | "
            f"median_best={overall['median_best_score']:.4f}"
        )
        if overall["pass_at_k"]:
            pass_parts = []
            for k, value in sorted(overall["pass_at_k"].items(), key=lambda item: int(item[0])):
                pass_parts.append(
                    f"pass@{k}={value['count']}/{value['total']} "
                    f"({value['rate'] * 100:.1f}%)"
                )
            parts.append("pass@k: " + " | ".join(pass_parts))
        buckets = overall["score_buckets"]
        parts.append(
            "buckets: "
            f"perfect={buckets['perfect']} partial={buckets['partial']} zero={buckets['zero']}"
        )
        parts.append(
            "verify_status: "
            + ", ".join(f"{key}={value}" for key, value in overall["status_counts"].items())
        )
        parts.append(
            "agent_status: "
            + ", ".join(f"{key}={value}" for key, value in overall["agent_status_counts"].items())
        )
        spending = f"${usage.get('spending_usd', 0.0):.6f}"
        if usage.get("spending_is_partial"):
            spending += " partial"
        parts.append(
            "llm: "
            f"calls={usage.get('llm_call_count', 0)} | "
            f"total_tokens={usage.get('total_tokens', 0)} | "
            f"spending={spending} | "
            f"missing_stats={usage.get('missing_stats_task_runs', 0)}"
        )

        if result["by_domain"]:
            parts.append("by_domain:")
            for domain, stats in result["by_domain"].items():
                parts.append(
                    f"  {domain}: pass={stats['pass']}/{stats['tasks']} "
                    f"({stats['pass_rate'] * 100:.1f}%) "
                    f"avg_best={stats['avg_best_score']:.4f}"
                )

        if result["unreadable_json"]:
            parts.append("unreadable_json:")
            for path in result["unreadable_json"]:
                parts.append(f"  {path}")

        parts.append("tasks:")
        for task in result["tasks"]:
            missing = ""
            if task["missing_verify_runs"]:
                missing = f" missing_verify={task['missing_verify_runs']}"
            parts.append(
                f"  {task['task_id']} [{task['domain']}]: "
                f"pass={task['pass']} best={task['best_score']:.4f} "
                f"mean={task['mean_score']:.4f} "
                f"ckpt={task['checkpoint_score']:.4f} "
                f"runs={task['runs_seen']}{missing}"
            )
            for run in task["runs"]:
                earned = run["earned"]
                total = run["total"]
                earned_label = f"{earned}/{total}" if earned is not None and total is not None else "-"
                line = (
                    f"    r{run['run_idx']}: "
                    f"{run['verify_status']} score={run['score']:.4f} "
                    f"earned={earned_label} "
                    f"agent={run['agent_status']} steps={run['steps']}"
                )
                check_text = checks_for_output(run, checks)
                if check_text:
                    line += f" | {check_text}"
                parts.append(line)
                if run.get("agent_error"):
                    parts.append(f"      agent_error: {run['agent_error']}")
                if run.get("verify_error"):
                    parts.append(f"      verify_error: {run['verify_error']}")
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate current detailed scores for SaaS-Bench experiment directories."
    )
    parser.add_argument(
        "dirs",
        nargs="*",
        type=Path,
        default=[Path("results"), Path("results-subset"), Path("results-subset-v1")],
        help=(
            "Experiment result directories. If omitted, immediate experiments under "
            "results/ and results-subset/ are scored. If a supplied directory has no "
            "direct artifacts, its immediate child experiment directories are scored."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of text.",
    )
    parser.add_argument(
        "--checks",
        choices=("failed", "all", "none"),
        default="failed",
        help="How much verifier check detail to include in text output.",
    )
    parser.add_argument(
        "--count-missing-as-zero",
        action="store_true",
        help=(
            "Include agent runs without verifier JSON in score averages as zero. "
            "By default, missing verifier runs are reported but excluded from scoring."
        ),
    )
    parser.add_argument(
        "--task-ids",
        dest="task_ids_file",
        type=Path,
        default=Path("task_ids_21.txt"),
        help=(
            "Optional file of task ids to include in scoring. Blank lines and # "
            "comments are ignored; comma or whitespace separated ids are accepted. "
            "If the file has no ids, no task filter is applied."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        task_ids = load_task_ids_file(args.task_ids_file) if args.task_ids_file else []
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    requested = args.dirs
    experiments = expand_experiment_dirs(requested)
    if not experiments:
        print("No experiment directories with result artifacts were found.")
        return 1

    results = [
        score_experiment(
            path,
            count_missing_as_zero=args.count_missing_as_zero,
            task_ids=task_ids,
        )
        for path in experiments
    ]
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print(format_text(results, checks=args.checks), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
