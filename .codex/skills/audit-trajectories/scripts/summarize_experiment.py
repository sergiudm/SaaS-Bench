#!/usr/bin/env python3
"""Generate a first-pass SaaS-Bench trajectory audit report."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any


RUN_RE = re.compile(r"^(?P<task_id>.+)_r(?P<run>\d+)\.json$")
VERIFY_RE = re.compile(r"^(?P<task_id>.+)_r(?P<run>\d+)_verify\.json$")
ERROR_WORDS = (
    "error",
    "exception",
    "failed",
    "failure",
    "timeout",
    "timed out",
    "not found",
    "unable",
    "could not",
    "not visible",
    "not clickable",
    "invalid",
    "stale",
    "intercept",
    "blocked",
)
SUCCESS_CLAIMS = (
    "successfully completed",
    "completed all",
    "all requirements",
    "all the requirements",
    "i have completed",
    "task is complete",
    "done",
)


def load_json(path: Path) -> tuple[Any | None, str | None]:
    try:
        return json.loads(path.read_text()), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def first_key(mapping: Any) -> str:
    if isinstance(mapping, dict) and mapping:
        return next(iter(mapping.keys()))
    return "<unknown>"


def action_items(step: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    items: list[tuple[str, dict[str, Any]]] = []
    for raw in as_list(step.get("actions")):
        if not isinstance(raw, dict) or not raw:
            items.append(("<unknown>", {}))
            continue
        name = first_key(raw)
        payload = raw.get(name)
        items.append((name, payload if isinstance(payload, dict) else {}))
    return items


def result_texts(step: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for raw in as_list(step.get("results")):
        if not isinstance(raw, dict):
            continue
        parts = []
        for key in ("extracted_content", "error", "message"):
            value = raw.get(key)
            if value:
                parts.append(str(value))
        if parts:
            texts.append(" | ".join(parts))
    return texts


def has_error_text(text: str) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in ERROR_WORDS)


def success_claims(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in SUCCESS_CLAIMS)


def short(text: Any, limit: int = 120) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3] + "..."


def percentile(values: list[int], pct: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * pct)
    return ordered[index]


def run_key(task_id: str, run: int) -> str:
    return f"{task_id}_r{run}"


def find_task_metadata(tasks_root: Path | None) -> dict[str, dict[str, Any]]:
    if not tasks_root or not tasks_root.exists():
        return {}
    index: dict[str, dict[str, Any]] = {}
    for meta_path in tasks_root.rglob("meta.json"):
        data, error = load_json(meta_path)
        if error or not isinstance(data, dict):
            continue
        task_id = data.get("task_id") or meta_path.parent.name
        if not isinstance(task_id, str):
            continue
        description = meta_path.parent / "description.md"
        verify_py = meta_path.parent / "verify.py"
        index[task_id] = {
            "domain": data.get("category_id"),
            "sites": data.get("meta_data", {}).get("sites", []),
            "description": str(description) if description.exists() else "",
            "verify_py": str(verify_py) if verify_py.exists() else "",
            "meta": str(meta_path),
        }
    return index


def expected_runs_from_summary(summary: dict[str, Any] | None) -> dict[tuple[str, int], dict[str, Any]]:
    expected: dict[tuple[str, int], dict[str, Any]] = {}
    if not summary:
        return expected
    for task in as_list(summary.get("tasks")):
        if not isinstance(task, dict) or not isinstance(task.get("task_id"), str):
            continue
        task_id = task["task_id"]
        scores = as_list(task.get("scores_per_run"))
        steps = as_list(task.get("steps_per_run"))
        run_count = max(len(scores), len(steps), int(summary.get("overall", {}).get("runs") or 0), 1)
        for run in range(run_count):
            expected[(task_id, run)] = {
                "summary_score": scores[run] if run < len(scores) else None,
                "summary_steps": steps[run] if run < len(steps) else None,
                "summary_task": task,
            }
    return expected


def discover_files(result_dir: Path) -> tuple[dict[tuple[str, int], Path], dict[tuple[str, int], Path]]:
    primary: dict[tuple[str, int], Path] = {}
    verify: dict[tuple[str, int], Path] = {}
    for path in result_dir.glob("*.json"):
        if path.name == "summary.json":
            continue
        verify_match = VERIFY_RE.match(path.name)
        if verify_match:
            verify[(verify_match.group("task_id"), int(verify_match.group("run")))] = path
            continue
        run_match = RUN_RE.match(path.name)
        if run_match:
            primary[(run_match.group("task_id"), int(run_match.group("run")))] = path
    return primary, verify


def action_signature(url: str, action_name: str) -> str:
    return f"{action_name}@{url}"


def analyze_primary(path: Path, max_steps: int | None) -> dict[str, Any]:
    data, error = load_json(path)
    if error or not isinstance(data, dict):
        return {"path": str(path), "load_error": error or "not a JSON object"}

    trajectory = as_list(data.get("trajectory"))
    action_counts: Counter[str] = Counter()
    done_values: list[Any] = []
    error_steps: list[dict[str, Any]] = []
    high_click_steps: list[int] = []
    repeated_spans: list[dict[str, Any]] = []
    step_numbers: list[int] = []
    current_sig = None
    current_start = None
    current_len = 0
    current_url = ""

    for idx, step_raw in enumerate(trajectory, start=1):
        step = step_raw if isinstance(step_raw, dict) else {}
        step_no = step.get("step") if isinstance(step.get("step"), int) else idx
        step_numbers.append(step_no)
        url = str(step.get("url") or "")
        actions = action_items(step) or [("<none>", {})]

        for action_name, payload in actions:
            action_counts[action_name] += 1
            if action_name == "done":
                done_values.append(payload.get("success"))
            if action_name == "click":
                index = payload.get("index")
                if isinstance(index, int) and index >= 10000:
                    high_click_steps.append(step_no)

        texts = result_texts(step)
        if any(has_error_text(text) for text in texts):
            error_steps.append(
                {
                    "step": step_no,
                    "url": url,
                    "actions": [name for name, _payload in actions],
                    "text": short(" ".join(texts), 180),
                }
            )

        sig = action_signature(url, actions[0][0])
        if sig == current_sig:
            current_len += 1
        else:
            if current_sig and current_len >= 5:
                repeated_spans.append(
                    {
                        "start": current_start,
                        "end": step_no - 1,
                        "length": current_len,
                        "signature": current_sig,
                        "url": current_url,
                    }
                )
            current_sig = sig
            current_start = step_no
            current_len = 1
            current_url = url

    if current_sig and current_len >= 5:
        repeated_spans.append(
            {
                "start": current_start,
                "end": step_numbers[-1] if step_numbers else current_start,
                "length": current_len,
                "signature": current_sig,
                "url": current_url,
            }
        )

    final_step = trajectory[-1] if trajectory and isinstance(trajectory[-1], dict) else {}
    output = data.get("agent_output") or ""
    near_max = bool(max_steps and len(trajectory) >= max(max_steps - 5, int(max_steps * 0.95)))

    return {
        "path": str(path),
        "task_id": data.get("task_id"),
        "status": data.get("status"),
        "agent_output": output,
        "agent_output_blank": not str(output).strip(),
        "success_claim": success_claims(str(output)),
        "steps": len(trajectory),
        "max_step_seen": max(step_numbers) if step_numbers else 0,
        "near_max_steps": near_max,
        "final_url": final_step.get("url", ""),
        "final_title": final_step.get("title", ""),
        "action_counts": action_counts,
        "done_count": action_counts.get("done", 0),
        "done_values": done_values,
        "error_steps": error_steps,
        "error_count": len(error_steps),
        "high_click_count": len(high_click_steps),
        "high_click_steps": high_click_steps[:20],
        "repeated_spans": repeated_spans,
    }


def analyze_verify(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"missing": True}
    data, error = load_json(path)
    if error or not isinstance(data, dict):
        return {"path": str(path), "load_error": error or "not a JSON object"}
    checks = as_list(data.get("checks"))
    passed = sum(1 for check in checks if isinstance(check, dict) and check.get("passed") is True)
    failed = sum(1 for check in checks if isinstance(check, dict) and check.get("passed") is False)
    return {
        "path": str(path),
        "status": data.get("status"),
        "score": data.get("score"),
        "earned": data.get("earned"),
        "total": data.get("total"),
        "all_pass": data.get("all_pass"),
        "returncode": data.get("returncode"),
        "error": data.get("error"),
        "check_count": len(checks),
        "checks_passed": passed,
        "checks_failed": failed,
        "failed_checks": [
            {
                "label": check.get("label"),
                "detail": check.get("detail"),
                "weight": check.get("weight"),
            }
            for check in checks
            if isinstance(check, dict) and check.get("passed") is False
        ][:8],
    }


def anomaly_labels(primary: dict[str, Any] | None, verify: dict[str, Any], summary_info: dict[str, Any] | None) -> list[str]:
    labels: list[str] = []
    if primary is None:
        labels.append("missing-primary")
    elif primary.get("load_error"):
        labels.append("malformed-primary")
    else:
        if primary.get("steps", 0) == 0:
            labels.append("zero-step")
        if primary.get("near_max_steps"):
            labels.append("near-max-steps")
        if primary.get("agent_output_blank"):
            labels.append("blank-output")
        if primary.get("error_count", 0) > 0:
            labels.append("error-like-results")
        if primary.get("repeated_spans"):
            labels.append("repeated-action-span")
        if primary.get("high_click_count", 0) >= 10:
            labels.append("many-high-index-clicks")

    if verify.get("missing"):
        labels.append("missing-verify")
    elif verify.get("load_error"):
        labels.append("malformed-verify")
    elif verify.get("status") not in ("PASS", "FAIL"):
        labels.append("verify-error-or-skip")

    failed = verify.get("all_pass") is False or verify.get("status") == "FAIL"
    if primary and not primary.get("load_error") and failed:
        if primary.get("success_claim") or True in primary.get("done_values", []):
            labels.append("overconfident-success-claim")

    if summary_info and summary_info.get("summary_steps") == 0 and primary is None:
        labels.append("summary-zero-step")
    return labels


def render_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def format_score(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.3f}"
    return ""


def generate_report(result_dir: Path, tasks_root: Path | None) -> str:
    summary_path = result_dir / "summary.json"
    summary, summary_error = load_json(summary_path) if summary_path.exists() else (None, None)
    summary = summary if isinstance(summary, dict) else None
    task_index = find_task_metadata(tasks_root)
    primary_files, verify_files = discover_files(result_dir)
    expected = expected_runs_from_summary(summary)
    all_keys = set(expected) | set(primary_files) | set(verify_files)
    raw_max_steps = summary.get("max_steps") if summary else None
    max_steps = raw_max_steps if isinstance(raw_max_steps, int) else None

    records: list[dict[str, Any]] = []
    global_actions: Counter[str] = Counter()
    anomaly_counts: Counter[str] = Counter()
    steps: list[int] = []
    scores: list[float] = []

    for task_id, run in sorted(all_keys):
        primary = analyze_primary(primary_files[(task_id, run)], max_steps) if (task_id, run) in primary_files else None
        verify = analyze_verify(verify_files.get((task_id, run)))
        summary_info = expected.get((task_id, run))
        meta = task_index.get(task_id, {})
        labels = anomaly_labels(primary, verify, summary_info)
        anomaly_counts.update(labels)
        if primary and not primary.get("load_error"):
            global_actions.update(primary.get("action_counts", Counter()))
            steps.append(int(primary.get("steps", 0)))
        score = verify.get("score")
        if isinstance(score, (int, float)):
            scores.append(float(score))
        records.append(
            {
                "task_id": task_id,
                "run": run,
                "key": run_key(task_id, run),
                "primary": primary,
                "verify": verify,
                "summary": summary_info,
                "meta": meta,
                "labels": labels,
            }
        )

    lines: list[str] = []
    lines.append("# Trajectory Quality Inventory")
    lines.append("")
    lines.append(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Experiment: `{result_dir}`")
    if tasks_root:
        lines.append(f"- Tasks root: `{tasks_root}`")
    if summary_error:
        lines.append(f"- Summary load error: `{summary_error}`")
    elif summary_path.exists():
        lines.append(f"- Summary: `{summary_path}`")
    else:
        lines.append("- Summary: missing")
    lines.append("")

    overall = summary.get("overall", {}) if summary else {}
    lines.append("## Dataset Snapshot")
    snapshot_rows = [
        ["Expected task runs from summary", len(expected)],
        ["Primary trajectory files", len(primary_files)],
        ["Verifier files", len(verify_files)],
        ["Task runs in inventory", len(records)],
        ["Summary total tasks", overall.get("total", "")],
        ["Summary pass@1 count", overall.get("pass_at_k", {}).get("1", {}).get("count", "")],
        ["Summary avg best score", overall.get("avg_best_score", "")],
    ]
    lines.extend(render_table(["Metric", "Value"], snapshot_rows))
    lines.append("")

    if scores or steps:
        lines.append("## Quantitative Signals")
        signal_rows: list[list[Any]] = []
        if scores:
            signal_rows.extend(
                [
                    ["Score avg", f"{mean(scores):.3f}"],
                    ["Score median", f"{median(scores):.3f}"],
                    ["Score min/max", f"{min(scores):.3f} / {max(scores):.3f}"],
                ]
            )
        if steps:
            signal_rows.extend(
                [
                    ["Steps avg", f"{mean(steps):.1f}"],
                    ["Steps median", f"{median(steps):.1f}"],
                    ["Steps p90", percentile(steps, 0.9)],
                    ["Steps min/max", f"{min(steps)} / {max(steps)}"],
                ]
            )
        lines.extend(render_table(["Signal", "Value"], signal_rows))
        lines.append("")

    if global_actions:
        lines.append("## Action Distribution")
        total_actions = sum(global_actions.values())
        rows = []
        for action, count in global_actions.most_common(20):
            share = 100 * count / total_actions if total_actions else 0
            rows.append([f"`{action}`", count, f"{share:.1f}%"])
        lines.extend(render_table(["Action", "Count", "Share"], rows))
        lines.append("")

    lines.append("## Anomaly Inventory")
    if anomaly_counts:
        rows = [[f"`{label}`", count] for label, count in anomaly_counts.most_common()]
        lines.extend(render_table(["Label", "Task runs"], rows))
    else:
        lines.append("No first-pass anomalies detected.")
    lines.append("")

    lines.append("## Task Run Inventory")
    task_rows = []
    for record in records:
        primary = record["primary"] or {}
        verify = record["verify"]
        meta = record["meta"]
        summary_info = record["summary"] or {}
        summary_task = summary_info.get("summary_task", {}) if isinstance(summary_info, dict) else {}
        task_rows.append(
            [
                f"`{record['key']}`",
                meta.get("domain") or summary_task.get("domain", ""),
                ",".join(meta.get("sites") or summary_task.get("sites") or []),
                primary.get("steps", "missing") if primary else "missing",
                verify.get("status", "missing"),
                format_score(verify.get("score") if verify.get("score") is not None else summary_info.get("summary_score", "")),
                ", ".join(record["labels"]),
            ]
        )
    lines.extend(render_table(["Run", "Domain", "Sites", "Steps", "Verify", "Score", "Flags"], task_rows))
    lines.append("")

    lines.append("## High-Priority Manual Review Queue")
    priority_order = (
        "missing-primary",
        "malformed-primary",
        "missing-verify",
        "malformed-verify",
        "verify-error-or-skip",
        "summary-zero-step",
        "overconfident-success-claim",
        "near-max-steps",
        "repeated-action-span",
        "many-high-index-clicks",
        "error-like-results",
    )
    priority_records = [record for record in records if any(label in record["labels"] for label in priority_order)]
    if not priority_records:
        lines.append("No high-priority queue items from first-pass flags.")
    else:
        for record in priority_records[:60]:
            primary = record["primary"] or {}
            verify = record["verify"]
            lines.append(f"### `{record['key']}`")
            lines.append(f"- Flags: {', '.join(record['labels'])}")
            lines.append(f"- Primary: `{primary.get('path', 'missing')}`")
            lines.append(f"- Verify: `{verify.get('path', 'missing')}`")
            if record["meta"].get("description"):
                lines.append(f"- Task description: `{record['meta']['description']}`")
            if primary and not primary.get("load_error"):
                lines.append(
                    f"- Steps: {primary.get('steps')} | done: {primary.get('done_count')} {primary.get('done_values')} | "
                    f"errors: {primary.get('error_count')} | high-index clicks: {primary.get('high_click_count')}"
                )
                if primary.get("agent_output"):
                    lines.append(f"- Output preview: {short(primary.get('agent_output'), 180)}")
                for span in primary.get("repeated_spans", [])[:3]:
                    lines.append(
                        f"- Repeated span: steps {span['start']}-{span['end']} "
                        f"({span['length']}x `{span['signature']}`)"
                    )
                for error_step in primary.get("error_steps", [])[:3]:
                    lines.append(
                        f"- Error-like result: step {error_step['step']} `{','.join(error_step['actions'])}` "
                        f"{error_step['text']}"
                    )
            if verify.get("failed_checks"):
                for check in verify["failed_checks"][:5]:
                    lines.append(f"- Failed check: {short(check.get('label'))} -> {short(check.get('detail'), 160)}")
            lines.append("")

    lines.append("## Manual Audit Template")
    lines.append("")
    lines.append("Use this template for each reviewed task in the final report:")
    lines.append("")
    lines.append("```markdown")
    lines.append("### <task_id>_rN")
    lines.append("- Files: <primary>, <verify>, <description>")
    lines.append("- Outcome: verify=<PASS/FAIL/ERROR>, score=<score>, steps=<n>")
    lines.append("- Quality rating: <High/Medium/Low/Artifact Missing>")
    lines.append("- Main finding: <one sentence>")
    lines.append("")
    lines.append("| Requirement/check | Trajectory evidence | Verifier result | Judgment |")
    lines.append("|---|---|---|---|")
    lines.append("| <requirement> | Steps <range>: <observed behavior> | <passed/failed/detail> | <quality note> |")
    lines.append("")
    lines.append("- Failure mode(s): <taxonomy labels>")
    lines.append("- Verifier/harness concerns: <none or evidence>")
    lines.append("- Recommendation: <specific change>")
    lines.append("```")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- This file is an automated inventory and review scaffold. Manual trajectory reading is still required for final quality judgments.")
    lines.append("- Treat anomaly labels as leads, not proof.")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path, help="Experiment result directory, e.g. results-subset/gemini-3.5-flash")
    parser.add_argument("--tasks-root", type=Path, default=Path("tasks"), help="SaaS-Bench tasks root. Defaults to ./tasks")
    parser.add_argument("--output", type=Path, default=None, help="Markdown output path. Defaults to <result_dir>/trajectory_quality_report.md")
    args = parser.parse_args()

    result_dir = args.result_dir
    if not result_dir.exists() or not result_dir.is_dir():
        raise SystemExit(f"Result directory not found: {result_dir}")
    output = args.output or (result_dir / "trajectory_quality_report.md")
    report = generate_report(result_dir, args.tasks_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report)
    print(output)


if __name__ == "__main__":
    main()
