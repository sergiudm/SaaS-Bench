#!/usr/bin/env python3
"""Build a static HTML viewer for SaaS-Bench trajectory JSON files.

The viewer is intentionally a visualization/debugging tool, not a browser
replayer. It turns stored trajectory steps into a human-readable timeline with
task selection, filtering, step playback, failure hot spots, URL changes, and
raw JSON inspection.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


IGNORE_JSON_NAMES = {"summary.json", "trajectory_viewer.html"}


@dataclass(frozen=True)
class ArtifactKey:
    task_id: str
    run_idx: int

    @property
    def suffix(self) -> str:
        return f"_r{self.run_idx}"


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def split_run_suffix(stem: str) -> tuple[str, int]:
    head, sep, tail = stem.rpartition("_r")
    if sep and head and tail.isdigit():
        return head, int(tail)
    return stem, 0


def parse_agent_name(path: Path) -> ArtifactKey | None:
    if path.suffix != ".json":
        return None
    if path.name in IGNORE_JSON_NAMES or path.stem.endswith("_verify"):
        return None
    task_id, run_idx = split_run_suffix(path.stem)
    if not task_id:
        return None
    return ArtifactKey(task_id=task_id, run_idx=run_idx)


def parse_verify_name(path: Path) -> ArtifactKey | None:
    if path.suffix != ".json" or not path.stem.endswith("_verify"):
        return None
    task_id, run_idx = split_run_suffix(path.stem[: -len("_verify")])
    if not task_id:
        return None
    return ArtifactKey(task_id=task_id, run_idx=run_idx)


def infer_domain(task_id: str) -> str:
    prefix = task_id.rsplit("_", 1)[0]
    return prefix.replace("-", " ").replace("_", " ").title() if prefix else "UNKNOWN"


def task_metadata(summary: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not summary:
        return {}

    by_task: dict[str, dict[str, Any]] = {}
    for task in summary.get("tasks") or []:
        if not isinstance(task, dict) or not task.get("task_id"):
            continue
        task_id = str(task["task_id"])
        by_task[task_id] = {
            "domain": task.get("domain") or infer_domain(task_id),
            "sites": task.get("sites") or [],
            "best_score": task.get("best_score"),
            "mean_score": task.get("mean_score"),
            "checkpoint_score": task.get("checkpoint_score"),
            "first_pass_run": task.get("first_pass_run"),
        }
    return by_task


def compact_summary(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not summary:
        return {}
    return {
        "tasks_dir": summary.get("tasks_dir"),
        "model": summary.get("model"),
        "workers": summary.get("workers"),
        "hostname": summary.get("hostname"),
        "isolation": summary.get("isolation"),
        "max_steps": summary.get("max_steps"),
        "run": summary.get("run"),
        "overall": summary.get("overall"),
    }


def safe_score(verify: dict[str, Any] | None) -> float | None:
    if not verify:
        return None
    try:
        return float(verify.get("score"))
    except (TypeError, ValueError):
        return None


def action_counts(trajectory: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for step in trajectory:
        if not isinstance(step, dict):
            continue
        for action in step.get("actions") or []:
            if not isinstance(action, dict) or not action:
                continue
            name = next(iter(action))
            counts[name] = counts.get(name, 0) + 1
    return counts


def build_dataset(result_dir: Path, task_ids: set[str] | None, run_indices: set[int] | None) -> dict[str, Any]:
    summary = load_json(result_dir / "summary.json")
    metadata = task_metadata(summary)

    verify_by_key: dict[ArtifactKey, dict[str, Any]] = {}
    for path in sorted(result_dir.glob("*.json")):
        key = parse_verify_name(path)
        if not key:
            continue
        verify = load_json(path)
        if verify is not None:
            verify_by_key[key] = verify

    runs: list[dict[str, Any]] = []
    for path in sorted(result_dir.glob("*.json")):
        key = parse_agent_name(path)
        if not key:
            continue
        if task_ids is not None and key.task_id not in task_ids:
            continue
        if run_indices is not None and key.run_idx not in run_indices:
            continue

        agent = load_json(path)
        if agent is None:
            continue
        trajectory = agent.get("trajectory")
        if not isinstance(trajectory, list):
            trajectory = []

        verify = verify_by_key.get(key)
        meta = metadata.get(key.task_id, {})
        score = safe_score(verify)
        run = {
            "id": f"{key.task_id}{key.suffix}",
            "task_id": key.task_id,
            "run_idx": key.run_idx,
            "domain": meta.get("domain") or infer_domain(key.task_id),
            "sites": meta.get("sites") or [],
            "file": path.name,
            "verify_file": f"{key.task_id}{key.suffix}_verify.json" if verify else None,
            "status": agent.get("status"),
            "agent_output": agent.get("agent_output"),
            "llm_stats": agent.get("llm_stats"),
            "verify_status": verify.get("status") if verify else None,
            "verify_score": score,
            "verify": verify,
            "step_count": len(trajectory),
            "action_counts": action_counts(trajectory),
            "trajectory": trajectory,
        }
        for score_key in ("best_score", "mean_score", "checkpoint_score", "first_pass_run"):
            if score_key in meta:
                run[score_key] = meta[score_key]
        runs.append(run)

    runs.sort(key=lambda r: (str(r["domain"]), str(r["task_id"]), int(r["run_idx"])))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result_dir": str(result_dir.resolve()),
        "summary": compact_summary(summary),
        "runs": runs,
    }


def json_for_html(data: Any) -> str:
    text = json.dumps(data, ensure_ascii=True, separators=(",", ":"))
    return (
        text.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def parse_csv_set(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    items: set[str] = set()
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                items.add(part)
    return items or None


def parse_run_indices(values: list[str] | None) -> set[int] | None:
    raw = parse_csv_set(values)
    if raw is None:
        return None
    indices: set[int] = set()
    for item in raw:
        if not re.fullmatch(r"\d+", item):
            raise SystemExit(f"Invalid run index: {item!r}")
        indices.add(int(item))
    return indices


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a static HTML visualization for SaaS-Bench trajectory files.",
    )
    parser.add_argument(
        "result_dir",
        nargs="?",
        default="results-subset/gemini-3.5-flash",
        help="Directory containing <task_id>_rN.json result files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output HTML path. Defaults to <result_dir>/trajectory_viewer.html.",
    )
    parser.add_argument(
        "--task-id",
        action="append",
        help="Only include a task id. Can be repeated or comma-separated.",
    )
    parser.add_argument(
        "--run-idx",
        action="append",
        help="Only include a run index. Can be repeated or comma-separated.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result_dir = Path(args.result_dir).expanduser()
    if not result_dir.is_dir():
        raise SystemExit(f"Result directory does not exist: {result_dir}")

    output = Path(args.output).expanduser() if args.output else result_dir / "trajectory_viewer.html"
    dataset = build_dataset(
        result_dir=result_dir,
        task_ids=parse_csv_set(args.task_id),
        run_indices=parse_run_indices(args.run_idx),
    )
    if not dataset["runs"]:
        raise SystemExit(f"No trajectory result files found in {result_dir}")

    html = HTML_TEMPLATE.replace("__TRAJECTORY_DATA__", json_for_html(dataset))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    print(f"Wrote {output} with {len(dataset['runs'])} trajectory run(s).")


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SaaS-Bench Trajectory Viewer</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fa;
      --surface: #ffffff;
      --surface-2: #f1f4f8;
      --ink: #18212f;
      --muted: #667085;
      --line: #d9e0ea;
      --strong-line: #bac6d5;
      --accent: #0f766e;
      --accent-ink: #073f3b;
      --danger: #b42318;
      --warn: #b54708;
      --good: #047857;
      --code: #101828;
      --shadow: 0 10px 30px rgba(16, 24, 40, 0.08);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-size: 14px;
    }
    button, input, select {
      font: inherit;
    }
    button {
      border: 1px solid var(--strong-line);
      background: var(--surface);
      color: var(--ink);
      border-radius: 6px;
      min-height: 34px;
      padding: 0 10px;
      cursor: pointer;
    }
    button:hover { border-color: var(--accent); }
    button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: #ffffff;
    }
    button:disabled {
      cursor: not-allowed;
      color: #98a2b3;
      border-color: var(--line);
      background: #f2f4f7;
    }
    input, select {
      width: 100%;
      min-height: 36px;
      border: 1px solid var(--line);
      background: var(--surface);
      color: var(--ink);
      border-radius: 6px;
      padding: 0 10px;
    }
    input[type="range"] {
      padding: 0;
      accent-color: var(--accent);
    }
    .app {
      display: grid;
      grid-template-columns: minmax(290px, 360px) minmax(0, 1fr);
      min-height: 100vh;
    }
    .sidebar {
      border-right: 1px solid var(--line);
      background: var(--surface);
      display: grid;
      grid-template-rows: auto auto minmax(0, 1fr);
      min-height: 100vh;
    }
    .sidebar-header {
      padding: 18px 18px 14px;
      border-bottom: 1px solid var(--line);
    }
    h1 {
      margin: 0;
      font-size: 18px;
      line-height: 1.2;
      font-weight: 720;
      letter-spacing: 0;
    }
    .subtle {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }
    .filters {
      display: grid;
      gap: 10px;
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfe;
    }
    .filter-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .run-list {
      overflow: auto;
      padding: 10px;
    }
    .run-row {
      width: 100%;
      display: grid;
      gap: 7px;
      text-align: left;
      border: 1px solid transparent;
      background: transparent;
      border-radius: 8px;
      padding: 10px;
      min-height: 80px;
    }
    .run-row:hover {
      background: var(--surface-2);
      border-color: var(--line);
    }
    .run-row.selected {
      background: #eef8f6;
      border-color: #84c7bf;
    }
    .run-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      font-weight: 700;
      color: var(--ink);
    }
    .run-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      color: var(--muted);
      font-size: 12px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      min-height: 22px;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 8px;
      color: var(--muted);
      background: var(--surface);
      white-space: nowrap;
    }
    .pill.good { color: var(--good); border-color: #a7f3d0; background: #ecfdf3; }
    .pill.warn { color: var(--warn); border-color: #fedf89; background: #fffaeb; }
    .pill.danger { color: var(--danger); border-color: #fecdca; background: #fef3f2; }
    .main {
      min-width: 0;
      display: grid;
      grid-template-rows: auto auto minmax(0, 1fr);
      min-height: 100vh;
    }
    .topbar {
      padding: 16px 22px 14px;
      background: var(--surface);
      border-bottom: 1px solid var(--line);
      display: grid;
      gap: 10px;
    }
    .topline {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
    }
    .title-block {
      min-width: 0;
    }
    h2 {
      margin: 0 0 6px;
      font-size: 20px;
      line-height: 1.25;
      letter-spacing: 0;
    }
    .header-pills {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      justify-content: flex-end;
    }
    .summary-strip {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfe;
      padding: 9px 10px;
      min-height: 58px;
    }
    .metric-label {
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .metric-value {
      margin-top: 4px;
      font-size: 16px;
      font-weight: 750;
      overflow-wrap: anywhere;
    }
    .player {
      background: #fbfcfe;
      border-bottom: 1px solid var(--line);
      padding: 12px 22px;
      display: grid;
      grid-template-columns: auto minmax(180px, 1fr) auto;
      gap: 12px;
      align-items: center;
    }
    .controls {
      display: flex;
      gap: 8px;
      align-items: center;
    }
    .speed {
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    .content {
      min-height: 0;
      overflow: auto;
      padding: 18px 22px 26px;
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(280px, 380px);
      gap: 18px;
    }
    .panel {
      border: 1px solid var(--line);
      background: var(--surface);
      border-radius: 8px;
      box-shadow: var(--shadow);
      min-width: 0;
    }
    .panel-header {
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
    }
    .panel-title {
      font-weight: 750;
    }
    .panel-body {
      padding: 14px;
      min-width: 0;
    }
    .step-map {
      display: grid;
      grid-template-columns: repeat(var(--step-count), minmax(10px, 1fr));
      gap: 3px;
      overflow-x: auto;
      padding-bottom: 4px;
    }
    .step-marker {
      height: 28px;
      min-width: 10px;
      border: 0;
      border-radius: 3px;
      background: #d0d5dd;
      padding: 0;
    }
    .step-marker:hover { outline: 2px solid #84c7bf; }
    .step-marker.current { background: var(--accent); }
    .step-marker.error { background: #f97066; }
    .step-marker.warn { background: #fdb022; }
    .step-marker.done { background: #32d583; }
    .step-marker.url { box-shadow: inset 0 -5px 0 #175cd3; }
    .narrative-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    .field {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fbfcfe;
      min-width: 0;
    }
    .field.full { grid-column: 1 / -1; }
    .field-label {
      margin-bottom: 7px;
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    .field-value {
      line-height: 1.55;
      overflow-wrap: anywhere;
      white-space: pre-wrap;
    }
    .action-list, .observation-list, .hotspot-list, .url-list, .check-list {
      display: grid;
      gap: 8px;
      margin: 0;
      padding: 0;
      list-style: none;
    }
    .action-item, .observation-item, .hotspot-item, .url-item, .check-item {
      border: 1px solid var(--line);
      background: var(--surface);
      border-radius: 8px;
      padding: 10px;
      min-width: 0;
    }
    .action-item {
      display: grid;
      grid-template-columns: auto minmax(0, 1fr);
      gap: 9px;
      align-items: start;
    }
    .action-type {
      min-width: 64px;
      color: var(--accent-ink);
      font-weight: 750;
    }
    .action-text, .observation-text {
      overflow-wrap: anywhere;
      white-space: pre-wrap;
      line-height: 1.45;
    }
    .observation-item.error, .hotspot-item.error, .check-item.fail {
      border-color: #fecdca;
      background: #fff7f6;
    }
    .hotspot-item.warn {
      border-color: #fedf89;
      background: #fffbf0;
    }
    .hotspot-button {
      display: block;
      width: 100%;
      min-height: 0;
      padding: 0;
      border: 0;
      background: transparent;
      text-align: left;
    }
    .hotspot-button:hover { color: var(--accent); }
    .mini-title {
      font-weight: 720;
      margin-bottom: 3px;
    }
    .tabs {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
    }
    .tab-button.active {
      border-color: var(--accent);
      color: var(--accent-ink);
      background: #eef8f6;
    }
    .tab-panel[hidden] { display: none; }
    details {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfe;
      margin-top: 12px;
    }
    summary {
      cursor: pointer;
      padding: 10px 12px;
      font-weight: 700;
    }
    pre {
      margin: 0;
      padding: 12px;
      overflow: auto;
      max-height: 360px;
      color: var(--code);
      background: #f4f6f8;
      border-top: 1px solid var(--line);
      font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
    }
    .empty {
      color: var(--muted);
      font-style: italic;
    }
    @media (max-width: 1080px) {
      .app { grid-template-columns: 300px minmax(0, 1fr); }
      .content { grid-template-columns: 1fr; }
      .summary-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 780px) {
      .app { grid-template-columns: 1fr; }
      .sidebar { min-height: auto; border-right: 0; border-bottom: 1px solid var(--line); }
      .run-list { max-height: 280px; }
      .topline, .player { grid-template-columns: 1fr; display: grid; }
      .header-pills { justify-content: flex-start; }
      .narrative-grid { grid-template-columns: 1fr; }
      .summary-strip { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <script id="trajectory-data" type="application/json">__TRAJECTORY_DATA__</script>
  <div class="app">
    <aside class="sidebar">
      <div class="sidebar-header">
        <h1>Trajectory Viewer</h1>
        <div class="subtle" id="dataset-meta"></div>
      </div>
      <div class="filters">
        <input id="search" type="search" placeholder="Search task, site, status, action">
        <div class="filter-grid">
          <select id="status-filter" aria-label="Status filter">
            <option value="all">All statuses</option>
            <option value="passed">Verifier passed</option>
            <option value="partial">Partial score</option>
            <option value="zero">Zero score</option>
            <option value="agent-error">Agent errors</option>
          </select>
          <select id="domain-filter" aria-label="Domain filter"></select>
        </div>
        <div class="subtle" id="result-count"></div>
      </div>
      <div class="run-list" id="run-list"></div>
    </aside>

    <main class="main">
      <section class="topbar">
        <div class="topline">
          <div class="title-block">
            <h2 id="run-title">No trajectory selected</h2>
            <div class="subtle" id="run-subtitle"></div>
          </div>
          <div class="header-pills" id="header-pills"></div>
        </div>
        <div class="summary-strip" id="summary-strip"></div>
      </section>

      <section class="player">
        <div class="controls">
          <button id="prev-step" title="Previous step">Prev</button>
          <button id="play-step" class="primary" title="Play or pause">Play</button>
          <button id="next-step" title="Next step">Next</button>
        </div>
        <input id="step-slider" type="range" min="1" max="1" value="1" aria-label="Step slider">
        <div class="speed">
          <span id="step-label">Step 0 / 0</span>
          <select id="speed-select" aria-label="Playback speed">
            <option value="1600">Slow</option>
            <option value="900" selected>Normal</option>
            <option value="450">Fast</option>
          </select>
        </div>
      </section>

      <section class="content">
        <div class="left-col">
          <section class="panel">
            <div class="panel-header">
              <div class="panel-title">Step Map</div>
              <div class="subtle" id="step-map-hint"></div>
            </div>
            <div class="panel-body">
              <div class="step-map" id="step-map"></div>
            </div>
          </section>

          <section class="panel" style="margin-top:18px">
            <div class="panel-header">
              <div class="panel-title" id="step-title">Current Step</div>
              <div class="subtle" id="step-url-change"></div>
            </div>
            <div class="panel-body">
              <div class="narrative-grid">
                <div class="field">
                  <div class="field-label">Page</div>
                  <div class="field-value" id="step-page"></div>
                </div>
                <div class="field">
                  <div class="field-label">Request</div>
                  <div class="field-value" id="step-request"></div>
                </div>
                <div class="field full">
                  <div class="field-label">Evaluation</div>
                  <div class="field-value" id="step-evaluation"></div>
                </div>
                <div class="field full">
                  <div class="field-label">Memory</div>
                  <div class="field-value" id="step-memory"></div>
                </div>
                <div class="field full">
                  <div class="field-label">Next Goal</div>
                  <div class="field-value" id="step-next-goal"></div>
                </div>
                <div class="field full">
                  <div class="field-label">Actions</div>
                  <ul class="action-list" id="action-list"></ul>
                </div>
                <div class="field full">
                  <div class="field-label">Observations</div>
                  <ul class="observation-list" id="observation-list"></ul>
                </div>
              </div>
              <details>
                <summary>Raw step JSON</summary>
                <pre id="raw-step"></pre>
              </details>
            </div>
          </section>
        </div>

        <aside class="right-col">
          <section class="panel">
            <div class="panel-header">
              <div class="panel-title">Analysis</div>
              <div class="tabs">
                <button class="tab-button active" data-tab="hotspots">Hot Spots</button>
                <button class="tab-button" data-tab="urls">URLs</button>
                <button class="tab-button" data-tab="verify">Verifier</button>
              </div>
            </div>
            <div class="panel-body">
              <div class="tab-panel" id="tab-hotspots">
                <ul class="hotspot-list" id="hotspot-list"></ul>
              </div>
              <div class="tab-panel" id="tab-urls" hidden>
                <ul class="url-list" id="url-list"></ul>
              </div>
              <div class="tab-panel" id="tab-verify" hidden>
                <div id="verify-summary"></div>
                <ul class="check-list" id="check-list" style="margin-top:10px"></ul>
              </div>
            </div>
          </section>

          <section class="panel" style="margin-top:18px">
            <div class="panel-header">
              <div class="panel-title">Final Output</div>
            </div>
            <div class="panel-body">
              <div class="field-value" id="agent-output"></div>
            </div>
          </section>
        </aside>
      </section>
    </main>
  </div>

  <script>
    const DATA = JSON.parse(document.getElementById("trajectory-data").textContent);
    const state = {
      selectedRunIndex: 0,
      stepIndex: 0,
      filteredRunIndexes: [],
      timer: null,
      activeTab: "hotspots",
    };

    const $ = (id) => document.getElementById(id);
    const els = {
      datasetMeta: $("dataset-meta"),
      search: $("search"),
      statusFilter: $("status-filter"),
      domainFilter: $("domain-filter"),
      resultCount: $("result-count"),
      runList: $("run-list"),
      runTitle: $("run-title"),
      runSubtitle: $("run-subtitle"),
      headerPills: $("header-pills"),
      summaryStrip: $("summary-strip"),
      prevStep: $("prev-step"),
      playStep: $("play-step"),
      nextStep: $("next-step"),
      stepSlider: $("step-slider"),
      speedSelect: $("speed-select"),
      stepLabel: $("step-label"),
      stepMap: $("step-map"),
      stepMapHint: $("step-map-hint"),
      stepTitle: $("step-title"),
      stepUrlChange: $("step-url-change"),
      stepPage: $("step-page"),
      stepRequest: $("step-request"),
      stepEvaluation: $("step-evaluation"),
      stepMemory: $("step-memory"),
      stepNextGoal: $("step-next-goal"),
      actionList: $("action-list"),
      observationList: $("observation-list"),
      rawStep: $("raw-step"),
      hotspotList: $("hotspot-list"),
      urlList: $("url-list"),
      verifySummary: $("verify-summary"),
      checkList: $("check-list"),
      agentOutput: $("agent-output"),
    };

    function selectedRun() {
      return DATA.runs[state.selectedRunIndex] || null;
    }

    function selectedStep() {
      const run = selectedRun();
      return run ? run.trajectory[state.stepIndex] || null : null;
    }

    function asText(value, fallback = "None recorded") {
      if (value === null || value === undefined || value === "") return fallback;
      if (typeof value === "string") return value;
      return JSON.stringify(value, null, 2);
    }

    function numberText(value, digits = 3) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
      return Number(value).toFixed(digits).replace(/0+$/, "").replace(/\\.$/, "");
    }

    function statusTone(run) {
      const score = Number(run.verify_score || 0);
      const agentStatus = String(run.status || "").toLowerCase();
      const verifyStatus = String(run.verify_status || "").toLowerCase();
      if (verifyStatus === "pass" || score >= 1) return "good";
      if (agentStatus.includes("error") || agentStatus.includes("abort")) return "danger";
      if (score > 0) return "warn";
      return "danger";
    }

    function pill(text, tone = "") {
      const span = document.createElement("span");
      span.className = `pill ${tone}`.trim();
      span.textContent = text;
      return span;
    }

    function firstActionName(step) {
      const actions = step && Array.isArray(step.actions) ? step.actions : [];
      for (const action of actions) {
        if (action && typeof action === "object") return Object.keys(action)[0] || "";
      }
      return "";
    }

    function actionName(action) {
      if (!action || typeof action !== "object") return "action";
      return Object.keys(action)[0] || "action";
    }

    function truncate(text, limit = 140) {
      const value = String(text || "");
      return value.length > limit ? `${value.slice(0, limit - 1)}...` : value;
    }

    function basenameFromUrl(url) {
      if (!url) return "";
      try {
        const parsed = new URL(url);
        return `${parsed.hostname}${parsed.pathname}${parsed.search}`;
      } catch {
        return String(url);
      }
    }

    function describeAction(action) {
      if (!action || typeof action !== "object") return String(action);
      const name = actionName(action);
      const payload = action[name];
      if (!payload || typeof payload !== "object") return JSON.stringify(action);

      if (name === "navigate") {
        return `Open ${payload.url || "unknown URL"}${payload.new_tab ? " in a new tab" : ""}`;
      }
      if (name === "click") {
        const target = payload.index !== undefined ? `element #${payload.index}` : "target";
        return `Click ${target}`;
      }
      if (name === "input") {
        const text = payload.text !== undefined ? JSON.stringify(String(payload.text)) : "text";
        const target = payload.index !== undefined ? `element #${payload.index}` : "field";
        return `Type ${text} into ${target}${payload.clear ? " after clearing it" : ""}`;
      }
      if (name === "wait") {
        return `Wait ${payload.seconds || payload.duration || ""} seconds`.trim();
      }
      if (name === "scroll") {
        const amount = payload.amount !== undefined ? ` by ${payload.amount}` : "";
        return `Scroll${amount}`;
      }
      if (name === "write_file") {
        return `Write ${payload.file_name || "file"}${payload.append ? " by appending" : ""}`;
      }
      if (name === "read_file") {
        return `Read ${payload.file_name || "file"}`;
      }
      if (name === "done") {
        return `Finish task${payload.text ? `: ${payload.text}` : ""}`;
      }
      if (name === "send_keys") {
        return `Send keys ${payload.keys || JSON.stringify(payload)}`;
      }
      if (name === "select_dropdown_option") {
        return `Select ${payload.text || payload.value || "option"} in element #${payload.index}`;
      }
      if (name === "go_back") {
        return "Go back";
      }
      if (name === "open_tab") {
        return `Open tab ${payload.url || ""}`.trim();
      }
      return `${name}: ${truncate(JSON.stringify(payload), 220)}`;
    }

    function observationText(result) {
      if (!result || typeof result !== "object") return String(result);
      const pieces = [];
      if (result.error) pieces.push(`Error: ${result.error}`);
      if (result.extracted_content) pieces.push(result.extracted_content);
      if (result.success !== undefined) pieces.push(`Success: ${result.success}`);
      if (result.is_done !== undefined) pieces.push(`Done: ${result.is_done}`);
      return pieces.join("\\n") || JSON.stringify(result);
    }

    function hasError(step) {
      return (step.results || []).some((result) => result && result.error);
    }

    function hasDone(step) {
      return (step.results || []).some((result) => result && result.is_done);
    }

    function evaluationSignals(step) {
      const thought = step.thought || {};
      const text = [
        thought.evaluation,
        thought.memory,
        thought.next_goal,
        ...(step.results || []).map(observationText),
      ].filter(Boolean).join("\\n").toLowerCase();
      const signals = [];
      if (/error|exception|failed|failure|unable|could not|blocked|timed out|timeout/.test(text)) {
        signals.push("failure language");
      }
      if (/uncertain|not sure|still loading|in progress|retry|try again/.test(text)) {
        signals.push("uncertainty or retry");
      }
      return signals;
    }

    function isUrlChange(run, index) {
      if (!run || index <= 0) return index === 0;
      const prev = run.trajectory[index - 1] || {};
      const curr = run.trajectory[index] || {};
      return (prev.url || "") !== (curr.url || "");
    }

    function repeatedActionSignal(run, index) {
      if (!run || index < 4) return false;
      const current = firstActionName(run.trajectory[index]);
      if (!current) return false;
      let same = 0;
      for (let i = Math.max(0, index - 4); i <= index; i += 1) {
        const step = run.trajectory[i] || {};
        if (firstActionName(step) === current && (step.url || "") === ((run.trajectory[index] || {}).url || "")) {
          same += 1;
        }
      }
      return same >= 4;
    }

    function stepSeverity(run, index) {
      const step = run.trajectory[index] || {};
      if (hasError(step)) return "error";
      if (hasDone(step)) return "done";
      if (evaluationSignals(step).length || repeatedActionSignal(run, index)) return "warn";
      return "";
    }

    function findHotspots(run) {
      const hotspots = [];
      run.trajectory.forEach((step, index) => {
        const signals = [];
        if (hasError(step)) signals.push("action error");
        signals.push(...evaluationSignals(step));
        if (repeatedActionSignal(run, index)) signals.push("possible loop");
        if (!signals.length) return;
        const thought = step.thought || {};
        hotspots.push({
          index,
          severity: hasError(step) ? "error" : "warn",
          title: `Step ${index + 1}: ${signals.join(", ")}`,
          detail: truncate(thought.evaluation || thought.next_goal || observationText((step.results || [])[0]) || "No detail", 180),
        });
      });
      return hotspots;
    }

    function urlTrail(run) {
      const trail = [];
      let last = null;
      run.trajectory.forEach((step, index) => {
        const url = step.url || "";
        if (!url || url === last) return;
        trail.push({
          index,
          title: step.title || "",
          url,
        });
        last = url;
      });
      return trail;
    }

    function runMatchesFilter(run) {
      const query = els.search.value.trim().toLowerCase();
      const status = els.statusFilter.value;
      const domain = els.domainFilter.value;
      if (domain !== "all" && run.domain !== domain) return false;

      const score = Number(run.verify_score || 0);
      const agentStatus = String(run.status || "").toLowerCase();
      const verifyStatus = String(run.verify_status || "").toLowerCase();
      if (status === "passed" && !(verifyStatus === "pass" || score >= 1)) return false;
      if (status === "partial" && !(score > 0 && score < 1)) return false;
      if (status === "zero" && score !== 0) return false;
      if (status === "agent-error" && !(agentStatus.includes("error") || agentStatus.includes("abort"))) return false;

      if (!query) return true;
      const actionNames = Object.keys(run.action_counts || {}).join(" ");
      const haystack = [
        run.task_id,
        run.domain,
        (run.sites || []).join(" "),
        run.status,
        run.verify_status,
        run.file,
        actionNames,
        run.agent_output,
      ].join(" ").toLowerCase();
      return haystack.includes(query);
    }

    function renderDatasetMeta() {
      const summary = DATA.summary || {};
      const model = summary.model || "unknown model";
      const total = DATA.runs.length;
      const maxSteps = summary.max_steps ? `, max ${summary.max_steps} steps` : "";
      els.datasetMeta.textContent = `${model} - ${total} trajectory run(s)${maxSteps}`;

      const domains = Array.from(new Set(DATA.runs.map((run) => run.domain))).sort();
      els.domainFilter.innerHTML = "";
      const all = document.createElement("option");
      all.value = "all";
      all.textContent = "All domains";
      els.domainFilter.appendChild(all);
      domains.forEach((domain) => {
        const option = document.createElement("option");
        option.value = domain;
        option.textContent = domain;
        els.domainFilter.appendChild(option);
      });
    }

    function renderRunList() {
      state.filteredRunIndexes = DATA.runs
        .map((run, index) => ({ run, index }))
        .filter(({ run }) => runMatchesFilter(run))
        .map(({ index }) => index);

      els.resultCount.textContent = `${state.filteredRunIndexes.length} of ${DATA.runs.length} shown`;
      els.runList.innerHTML = "";

      if (!state.filteredRunIndexes.includes(state.selectedRunIndex) && state.filteredRunIndexes.length) {
        state.selectedRunIndex = state.filteredRunIndexes[0];
        state.stepIndex = 0;
      }

      if (!state.filteredRunIndexes.length) {
        const empty = document.createElement("div");
        empty.className = "empty";
        empty.textContent = "No trajectories match the current filters.";
        els.runList.appendChild(empty);
        renderRun();
        return;
      }

      state.filteredRunIndexes.forEach((runIndex) => {
        const run = DATA.runs[runIndex];
        const row = document.createElement("button");
        row.className = `run-row ${runIndex === state.selectedRunIndex ? "selected" : ""}`;
        row.type = "button";
        row.addEventListener("click", () => {
          stopPlayback();
          state.selectedRunIndex = runIndex;
          state.stepIndex = 0;
          renderAll();
        });

        const title = document.createElement("div");
        title.className = "run-title";
        const name = document.createElement("span");
        name.textContent = run.id;
        title.appendChild(name);
        title.appendChild(pill(numberText(run.verify_score), statusTone(run)));

        const meta = document.createElement("div");
        meta.className = "run-meta";
        meta.appendChild(pill(run.domain));
        meta.appendChild(pill(`${run.step_count} steps`));
        meta.appendChild(pill(String(run.status || "unknown")));
        if (run.verify_status) meta.appendChild(pill(`verify ${run.verify_status}`, statusTone(run)));

        row.appendChild(title);
        row.appendChild(meta);
        els.runList.appendChild(row);
      });
    }

    function metric(label, value) {
      const item = document.createElement("div");
      item.className = "metric";
      const l = document.createElement("div");
      l.className = "metric-label";
      l.textContent = label;
      const v = document.createElement("div");
      v.className = "metric-value";
      v.textContent = value;
      item.appendChild(l);
      item.appendChild(v);
      return item;
    }

    function renderHeader(run) {
      if (!run) {
        els.runTitle.textContent = "No trajectory selected";
        els.runSubtitle.textContent = "";
        els.headerPills.innerHTML = "";
        els.summaryStrip.innerHTML = "";
        return;
      }

      els.runTitle.textContent = run.id;
      const sites = (run.sites || []).length ? `Sites: ${(run.sites || []).join(", ")}` : "Sites unavailable";
      els.runSubtitle.textContent = `${run.domain} - ${sites} - ${run.file}`;
      els.headerPills.innerHTML = "";
      els.headerPills.appendChild(pill(String(run.status || "unknown"), statusTone(run)));
      if (run.verify_status) els.headerPills.appendChild(pill(`verify ${run.verify_status}`, statusTone(run)));
      els.headerPills.appendChild(pill(`${run.step_count} steps`));
      if (run.llm_stats && run.llm_stats.total_tokens) {
        els.headerPills.appendChild(pill(`${Number(run.llm_stats.total_tokens).toLocaleString()} tokens`));
      }

      els.summaryStrip.innerHTML = "";
      els.summaryStrip.appendChild(metric("Verify Score", numberText(run.verify_score)));
      els.summaryStrip.appendChild(metric("Agent Status", String(run.status || "unknown")));
      els.summaryStrip.appendChild(metric("Step", `${Math.min(state.stepIndex + 1, run.step_count)} / ${run.step_count}`));
      els.summaryStrip.appendChild(metric("Top Actions", topActionsText(run)));
    }

    function topActionsText(run) {
      const entries = Object.entries(run.action_counts || {}).sort((a, b) => b[1] - a[1]).slice(0, 3);
      if (!entries.length) return "none";
      return entries.map(([name, count]) => `${name} ${count}`).join(", ");
    }

    function renderPlayer(run) {
      const count = run ? run.step_count : 0;
      const hasSteps = count > 0;
      els.prevStep.disabled = !hasSteps || state.stepIndex <= 0;
      els.nextStep.disabled = !hasSteps || state.stepIndex >= count - 1;
      els.playStep.disabled = !hasSteps;
      els.stepSlider.disabled = !hasSteps;
      els.stepSlider.min = hasSteps ? "1" : "0";
      els.stepSlider.max = String(Math.max(1, count));
      els.stepSlider.value = String(hasSteps ? state.stepIndex + 1 : 0);
      els.stepLabel.textContent = hasSteps ? `Step ${state.stepIndex + 1} / ${count}` : "Step 0 / 0";
    }

    function renderStepMap(run) {
      els.stepMap.innerHTML = "";
      if (!run || !run.step_count) {
        els.stepMapHint.textContent = "No steps recorded";
        return;
      }
      els.stepMap.style.setProperty("--step-count", String(run.step_count));
      els.stepMapHint.textContent = "Orange marks warnings, red marks errors, blue underline marks URL changes";
      run.trajectory.forEach((step, index) => {
        const marker = document.createElement("button");
        const severity = stepSeverity(run, index);
        const urlClass = isUrlChange(run, index) ? " url" : "";
        marker.className = `step-marker ${severity} ${index === state.stepIndex ? "current" : ""}${urlClass}`;
        marker.type = "button";
        marker.title = `Step ${index + 1}: ${firstActionName(step) || "no action"}${step.url ? ` - ${basenameFromUrl(step.url)}` : ""}`;
        marker.addEventListener("click", () => {
          stopPlayback();
          state.stepIndex = index;
          renderRun();
        });
        els.stepMap.appendChild(marker);
      });
    }

    function renderStep(run) {
      const step = selectedStep();
      if (!run || !step) {
        els.stepTitle.textContent = "Current Step";
        els.stepUrlChange.textContent = "";
        els.stepPage.textContent = "No step selected";
        els.stepRequest.textContent = "";
        els.stepEvaluation.textContent = "";
        els.stepMemory.textContent = "";
        els.stepNextGoal.textContent = "";
        els.actionList.innerHTML = "";
        els.observationList.innerHTML = "";
        els.rawStep.textContent = "";
        return;
      }

      const thought = step.thought || {};
      els.stepTitle.textContent = `Step ${state.stepIndex + 1}`;
      els.stepUrlChange.textContent = isUrlChange(run, state.stepIndex) ? "URL changed" : "same URL";
      els.stepPage.textContent = `${step.title || "Untitled"}\\n${step.url || "No URL recorded"}`;
      els.stepRequest.textContent = step.request_id || "No request id recorded";
      els.stepEvaluation.textContent = asText(thought.evaluation);
      els.stepMemory.textContent = asText(thought.memory);
      els.stepNextGoal.textContent = asText(thought.next_goal);

      els.actionList.innerHTML = "";
      const actions = Array.isArray(step.actions) ? step.actions : [];
      if (!actions.length) {
        const item = document.createElement("li");
        item.className = "empty";
        item.textContent = "No action recorded for this step.";
        els.actionList.appendChild(item);
      } else {
        actions.forEach((action) => {
          const item = document.createElement("li");
          item.className = "action-item";
          const type = document.createElement("div");
          type.className = "action-type";
          type.textContent = actionName(action);
          const text = document.createElement("div");
          text.className = "action-text";
          text.textContent = describeAction(action);
          item.appendChild(type);
          item.appendChild(text);
          els.actionList.appendChild(item);
        });
      }

      els.observationList.innerHTML = "";
      const results = Array.isArray(step.results) ? step.results : [];
      if (!results.length) {
        const item = document.createElement("li");
        item.className = "empty";
        item.textContent = "No observation recorded for this step.";
        els.observationList.appendChild(item);
      } else {
        results.forEach((result) => {
          const item = document.createElement("li");
          item.className = `observation-item ${result && result.error ? "error" : ""}`;
          const text = document.createElement("div");
          text.className = "observation-text";
          text.textContent = observationText(result);
          item.appendChild(text);
          els.observationList.appendChild(item);
        });
      }

      els.rawStep.textContent = JSON.stringify(step, null, 2);
    }

    function renderHotspots(run) {
      els.hotspotList.innerHTML = "";
      if (!run) return;
      const hotspots = findHotspots(run);
      if (!hotspots.length) {
        const item = document.createElement("li");
        item.className = "empty";
        item.textContent = "No obvious error, uncertainty, or loop signals were detected.";
        els.hotspotList.appendChild(item);
        return;
      }
      hotspots.slice(0, 80).forEach((hotspot) => {
        const item = document.createElement("li");
        item.className = `hotspot-item ${hotspot.severity}`;
        const button = document.createElement("button");
        button.className = "hotspot-button";
        button.type = "button";
        button.addEventListener("click", () => {
          stopPlayback();
          state.stepIndex = hotspot.index;
          renderRun();
        });
        const title = document.createElement("div");
        title.className = "mini-title";
        title.textContent = hotspot.title;
        const detail = document.createElement("div");
        detail.className = "subtle";
        detail.textContent = hotspot.detail;
        button.appendChild(title);
        button.appendChild(detail);
        item.appendChild(button);
        els.hotspotList.appendChild(item);
      });
    }

    function renderUrls(run) {
      els.urlList.innerHTML = "";
      if (!run) return;
      const trail = urlTrail(run);
      if (!trail.length) {
        const item = document.createElement("li");
        item.className = "empty";
        item.textContent = "No URLs recorded.";
        els.urlList.appendChild(item);
        return;
      }
      trail.forEach((entry) => {
        const item = document.createElement("li");
        item.className = "url-item";
        const button = document.createElement("button");
        button.className = "hotspot-button";
        button.type = "button";
        button.addEventListener("click", () => {
          stopPlayback();
          state.stepIndex = entry.index;
          renderRun();
        });
        const title = document.createElement("div");
        title.className = "mini-title";
        title.textContent = `Step ${entry.index + 1}: ${entry.title || "Untitled"}`;
        const detail = document.createElement("div");
        detail.className = "subtle";
        detail.textContent = entry.url;
        button.appendChild(title);
        button.appendChild(detail);
        item.appendChild(button);
        els.urlList.appendChild(item);
      });
    }

    function renderVerify(run) {
      els.verifySummary.innerHTML = "";
      els.checkList.innerHTML = "";
      if (!run || !run.verify) {
        els.verifySummary.textContent = "No verifier file was found for this trajectory.";
        return;
      }
      const verify = run.verify;
      const summary = document.createElement("div");
      summary.className = "field-value";
      summary.textContent = [
        `Status: ${verify.status || "unknown"}`,
        `Score: ${numberText(verify.score)}`,
        verify.error ? `Error: ${verify.error}` : "",
      ].filter(Boolean).join("\\n");
      els.verifySummary.appendChild(summary);

      const checks = Array.isArray(verify.checks) ? verify.checks : [];
      if (!checks.length) {
        const item = document.createElement("li");
        item.className = "empty";
        item.textContent = "No check details recorded.";
        els.checkList.appendChild(item);
        return;
      }
      checks.forEach((check, index) => {
        const passed = Boolean(check.passed || check.success || check.status === "PASS");
        const item = document.createElement("li");
        item.className = `check-item ${passed ? "" : "fail"}`;
        const title = document.createElement("div");
        title.className = "mini-title";
        title.textContent = check.name || check.id || `Check ${index + 1}`;
        const detail = document.createElement("div");
        detail.className = "subtle";
        detail.textContent = asText(check.message || check.error || check.detail || check, "");
        item.appendChild(title);
        item.appendChild(detail);
        els.checkList.appendChild(item);
      });
    }

    function renderAnalysis(run) {
      renderHotspots(run);
      renderUrls(run);
      renderVerify(run);
      els.agentOutput.textContent = run && run.agent_output ? run.agent_output : "No final agent output recorded.";
    }

    function renderRun() {
      const run = selectedRun();
      if (run && state.stepIndex >= run.step_count) state.stepIndex = Math.max(0, run.step_count - 1);
      renderHeader(run);
      renderPlayer(run);
      renderStepMap(run);
      renderStep(run);
      renderAnalysis(run);
    }

    function renderAll() {
      renderRunList();
      renderRun();
    }

    function nextStep() {
      const run = selectedRun();
      if (!run || state.stepIndex >= run.step_count - 1) {
        stopPlayback();
        return;
      }
      state.stepIndex += 1;
      renderRun();
    }

    function prevStep() {
      if (state.stepIndex <= 0) return;
      state.stepIndex -= 1;
      renderRun();
    }

    function stopPlayback() {
      if (state.timer) {
        clearInterval(state.timer);
        state.timer = null;
      }
      els.playStep.textContent = "Play";
    }

    function togglePlayback() {
      if (state.timer) {
        stopPlayback();
        return;
      }
      const run = selectedRun();
      if (!run || !run.step_count) return;
      if (state.stepIndex >= run.step_count - 1) state.stepIndex = 0;
      els.playStep.textContent = "Pause";
      state.timer = setInterval(nextStep, Number(els.speedSelect.value));
    }

    function activateTab(name) {
      state.activeTab = name;
      document.querySelectorAll(".tab-button").forEach((button) => {
        button.classList.toggle("active", button.dataset.tab === name);
      });
      document.querySelectorAll(".tab-panel").forEach((panel) => {
        panel.hidden = panel.id !== `tab-${name}`;
      });
    }

    function bindEvents() {
      els.search.addEventListener("input", renderAll);
      els.statusFilter.addEventListener("change", renderAll);
      els.domainFilter.addEventListener("change", renderAll);
      els.prevStep.addEventListener("click", () => {
        stopPlayback();
        prevStep();
      });
      els.nextStep.addEventListener("click", () => {
        stopPlayback();
        nextStep();
      });
      els.playStep.addEventListener("click", togglePlayback);
      els.speedSelect.addEventListener("change", () => {
        if (state.timer) {
          stopPlayback();
          togglePlayback();
        }
      });
      els.stepSlider.addEventListener("input", () => {
        stopPlayback();
        state.stepIndex = Math.max(0, Number(els.stepSlider.value) - 1);
        renderRun();
      });
      document.querySelectorAll(".tab-button").forEach((button) => {
        button.addEventListener("click", () => activateTab(button.dataset.tab));
      });
      window.addEventListener("keydown", (event) => {
        if (event.target && ["INPUT", "SELECT", "TEXTAREA"].includes(event.target.tagName)) return;
        if (event.key === "ArrowRight") {
          stopPlayback();
          nextStep();
        } else if (event.key === "ArrowLeft") {
          stopPlayback();
          prevStep();
        } else if (event.key === " ") {
          event.preventDefault();
          togglePlayback();
        }
      });
    }

    renderDatasetMeta();
    bindEvents();
    renderAll();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
