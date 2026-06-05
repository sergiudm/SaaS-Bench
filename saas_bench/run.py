"""SaaS-Bench eval harness — main entry point.

Concurrency model: ProcessPoolExecutor(max_workers) + asyncio.run() per process.
Each worker process has its own memory space, fully isolating bubus/browser-use
module-level state.
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import tempfile
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml

from saas_bench.agent import run_task
from saas_bench.loader import build_prompt, load_tasks
from saas_bench.reporting import generate_outputs, summarize_llm_usage_from_files
from saas_bench.slot import SlotManager
from saas_bench.verify_runner import run_verify


_SLOT_PREFIX = os.environ.get("SAAS_SLOT_PREFIX", "rollout")
_TMP_BASE = os.environ.get(
    "SAAS_BENCH_TMP",
    os.path.join(tempfile.gettempdir(), f"saas_bench_{_SLOT_PREFIX}"),
)


def _load_apps_config(apps_yaml: str) -> dict:
    with open(apps_yaml) as f:
        return yaml.safe_load(f)["apps"]


def _global_cleanup() -> None:
    """Kill stale chrome processes belonging to this instance and purge tmp dirs.

    Uses --user-data-dir path to identify only our own Chrome processes,
    so parallel run.py instances with different SAAS_SLOT_PREFIX don't
    kill each other's Chrome processes.
    """
    # 1. Kill only chrome processes whose --user-data-dir is under our _TMP_BASE
    r = subprocess.run(
        f"pkill -9 -f 'remote-debugging-port.*--user-data-dir={_TMP_BASE}'",
        shell=True, capture_output=True, text=True,
    )
    # Also kill by user-data-dir alone (order of flags may differ)
    subprocess.run(
        f"pkill -9 -f -- '--user-data-dir={_TMP_BASE}'",
        shell=True, capture_output=True, text=True,
    )
    killed = "ok" if r.returncode in (0, 1) else f"rc={r.returncode}"

    # 2. Remove leftover user-data dirs
    pattern = f"{_TMP_BASE}/chrome_*"
    leftovers = glob.glob(pattern)
    for d in leftovers:
        shutil.rmtree(d, ignore_errors=True)

    # 3. Remove any orphaned per-task file-system workdirs
    for d in glob.glob(f"{_TMP_BASE}/fs_*"):
        shutil.rmtree(d, ignore_errors=True)

    print(
        f"[startup-cleanup] killed stale chrome ({killed}) | "
        f"removed {len(leftovers)} chrome dirs (prefix={_SLOT_PREFIX}, tmp={_TMP_BASE})",
        flush=True,
    )


def _log_error(result_dir: str, slot_id: int, task_id: str, phase: str, exc: Exception) -> None:
    """Append an error entry to {result_dir}/errors.log."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    entry = (
        f"[{ts}] [slot {slot_id}] [{task_id}] [{phase}]\n"
        f"  {type(exc).__name__}: {exc}\n"
        f"  {''.join(tb[-3:]).rstrip()}\n\n"
    )
    log_path = Path(result_dir) / "errors.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(entry)


def _safe_json_exists(path: Path) -> bool:
    """True when path exists and contains parseable JSON."""
    if not path.exists():
        return False
    try:
        json.loads(path.read_text())
        return True
    except (json.JSONDecodeError, OSError):
        return False


def _agent_result_exists(result_dir: Path, task_id: str, run_idx: int) -> bool:
    """Return whether an agent result already exists for this task/run."""
    paths = [result_dir / f"{task_id}_r{run_idx}.json"]
    if run_idx == 0:
        # Backwards compatibility with older suffix-free result files.
        paths.append(result_dir / f"{task_id}.json")
    return any(_safe_json_exists(path) for path in paths)


def _print_llm_usage(llm_usage: dict) -> None:
    spending = llm_usage.get("spending_usd", 0.0)
    spending_label = f"${spending:.6f}"
    if llm_usage.get("spending_is_partial"):
        spending_label += " (partial; some task runs are missing pricing/stats)"

    print("\n=== LLM usage ===", flush=True)
    print(
        f"  Calls : {llm_usage.get('llm_call_count', 0)} "
        f"({llm_usage.get('usage_call_count', 0)} with token usage)",
        flush=True,
    )
    print(
        f"  Tokens: prompt={llm_usage.get('prompt_tokens', 0)} | "
        f"completion={llm_usage.get('completion_tokens', 0)} | "
        f"total={llm_usage.get('total_tokens', 0)}",
        flush=True,
    )
    print(f"  Spending: {spending_label}", flush=True)


def _load_task_ids_file(task_ids_file: str) -> list[str]:
    """Load task ids from a file, ignoring blanks and # comments."""
    path = Path(task_ids_file)
    if not path.is_file():
        raise FileNotFoundError(f"Task ids file does not exist: {task_ids_file}")

    task_ids: list[str] = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        task_ids.extend(line.replace(",", " ").split())

    if not task_ids:
        raise ValueError(f"Task ids file is empty: {task_ids_file}")
    return task_ids


def _run_one(
    task: dict,
    slot_id: int,
    apps_config: dict,
    model: str,
    result_dir: str,
    max_steps: int,
    hostname: str,
    use_isolation: bool,
    run_idx: int = 0,
    tasks_dir: str = "",
) -> dict:
    """Full execution of a single task (runs in a thread pool, owns its own asyncio event loop)."""
    import asyncio

    run_suffix = f"_r{run_idx}"
    sites: list[str] = task.get("meta", {}).get("meta_data", {}).get("sites", [])
    slot = SlotManager(apps_config, slot_id) if use_isolation else None
    port_map: dict[str, int] = {}
    known: list[str] = []

    if use_isolation and slot and sites:
        known = [a for a in sites if a in apps_config]
        unknown = [a for a in sites if a not in apps_config]
        if unknown:
            print(f"  [slot {slot_id}][{task['task_id']}] unknown apps {unknown}, skipping isolation", flush=True)
        if known:
            slot.start_apps(known, hostname=hostname)
            port_map = slot.get_port_map(known)
    elif not use_isolation and apps_config:
        port_map = {
            app: apps_config[app]["fixed_port"]
            for app in sites
            if app in apps_config and "fixed_port" in apps_config[app]
        }

    agent_result: dict = {"task_id": task["task_id"], "status": "error", "trajectory": []}
    verify_result: dict = {
        "task_id": task["task_id"], "status": "SKIP", "score": 0.0,
        "checks": [], "error": "not executed",
    }

    try:
        prompt, todo_md, input_files = build_prompt(task, port_map, hostname, tasks_root=tasks_dir)

        agent_result = asyncio.run(
            run_task(
                task, model, prompt, result_dir,
                max_steps=max_steps, slot_id=slot_id, todo_md=todo_md,
                run_idx=run_idx, input_files=input_files,
            )
        )

        if use_isolation and task.get("verify_py_path"):
            verify_result = run_verify(task, slot_id, port_map, hostname, result_dir,
                                       run_suffix=run_suffix)
        else:
            verify_result = {
                "task_id": task["task_id"],
                "status": "SKIP",
                "score": 0.0,
                "checks": [],
                "error": "verification skipped in no-isolation mode",
            }

    except Exception as exc:
        _log_error(result_dir, slot_id, task["task_id"], "agent", exc)
        agent_result["status"] = "error"
        agent_result["error"] = f"{type(exc).__name__}: {exc}"
        print(
            f"  [slot {slot_id}][{task['task_id']}] ERROR in agent: "
            f"{type(exc).__name__}: {str(exc)[:200]}",
            flush=True,
        )
    finally:
        if use_isolation and slot and known:
            slot.stop_apps(known)

    return {
        **agent_result,
        "run_idx": run_idx,
        "verify_score": verify_result.get("score", 0.0),
        "verify_status": verify_result.get("status", "SKIP"),
    }


def _run_task_runs(
    task: dict,
    slot_id: int,
    apps_config: dict,
    model: str,
    result_dir: str,
    max_steps: int,
    hostname: str,
    use_isolation: bool,
    run_indices: list[int],
    tasks_dir: str = "",
) -> list[tuple[int, dict]]:
    """Selected runs of a single task execute serially on the same slot."""
    results = []
    for run_idx in run_indices:
        result = _run_one(
            task, slot_id, apps_config, model, result_dir,
            max_steps, hostname, use_isolation, run_idx, tasks_dir,
        )
        results.append((run_idx, result))
    return results


def main(
    tasks_dir: str,
    model: str,
    workers: int,
    result_dir: str,
    max_steps: int,
    hostname: str,
    task_ids_file: str | None,
    apps_yaml: str,
    use_isolation: bool,
    runs: int = 1,
    run_start: int = 0,
    write_report: bool = True,
    rerun_existing: bool = False,
) -> None:
    _global_cleanup()
    started_at = datetime.now()
    t0 = time.perf_counter()

    # Nest results under a model-named subdirectory so different model runs
    # never clobber each other: results/<model_name>/...
    model_slug = model.replace("/", "_").replace(":", "_")
    result_dir = str(Path(result_dir) / model_slug)

    tasks = load_tasks(tasks_dir)
    if task_ids_file:
        task_ids = _load_task_ids_file(task_ids_file)
        task_id_set = set(task_ids)
        tasks = [t for t in tasks if str(t.get("task_id")) in task_id_set]
        found_ids = {str(t.get("task_id")) for t in tasks}
        missing_ids = [task_id for task_id in task_ids if task_id not in found_ids]
        if missing_ids:
            print(
                f"[task-ids] warning: {len(missing_ids)} ids from {task_ids_file} were not found: "
                f"{', '.join(missing_ids[:20])}{' ...' if len(missing_ids) > 20 else ''}",
                flush=True,
            )
        if not tasks:
            raise ValueError(f"No tasks matched ids from {task_ids_file}")

    apps_config = _load_apps_config(apps_yaml)

    run_indices = list(range(run_start, run_start + runs))
    result_path = Path(result_dir)
    pending_jobs: list[tuple[dict, list[int]]] = []
    skipped_jobs = 0
    for task in tasks:
        task_run_indices = [
            run_idx
            for run_idx in run_indices
            if rerun_existing or not _agent_result_exists(result_path, str(task["task_id"]), run_idx)
        ]
        skipped_jobs += len(run_indices) - len(task_run_indices)
        if task_run_indices:
            pending_jobs.append((task, task_run_indices))

    requested_jobs = len(tasks) * runs
    total_jobs = sum(len(run_indices_for_task) for _, run_indices_for_task in pending_jobs)
    print(
        f"Loaded {len(tasks)} tasks × runs={runs} (r{run_start}..r{run_start+runs-1}) = {requested_jobs} requested jobs | "
        f"pending={total_jobs} skipped_existing={skipped_jobs} | "
        f"workers={workers} | model={model} | isolation={use_isolation}",
        flush=True,
    )

    # Each task's r0→r1→...→rN runs serially on the same slot.
    # Tasks themselves run in parallel across workers (one slot per task).
    # This guarantees no two concurrent jobs ever share a slot.
    task_results: list[tuple[dict, int, dict | Exception]] = []

    if pending_jobs:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _run_task_runs, task,
                    i % workers,
                    apps_config, model,
                    result_dir, max_steps, hostname, use_isolation,
                    task_run_indices, tasks_dir,
                ): (task, task_run_indices)
                for i, (task, task_run_indices) in enumerate(pending_jobs)
            }
            for fut in as_completed(futures):
                task, task_run_indices = futures[fut]
                try:
                    for run_idx, result in fut.result():
                        task_results.append((task, run_idx, result))
                except Exception as e:
                    _log_error(result_dir, -1, task.get("task_id", "?"), "worker", e)
                    print(
                        f"  [{task.get('task_id','?')}] WORKER EXCEPTION: "
                        f"{type(e).__name__}: {str(e)[:200]}",
                        flush=True,
                    )
                    for run_idx in task_run_indices:
                        task_results.append((task, run_idx, e))
    else:
        print("No pending task results; all selected task JSON files already exist.", flush=True)

    ended_at = datetime.now()
    duration_s = round(time.perf_counter() - t0, 1)

    # Group statistics by domain (per task×run)
    domain_stats: dict[str, dict] = defaultdict(
        lambda: {"total": 0, "completed": 0, "error": 0, "exception": 0,
                 "verify_pass": 0, "verify_scores": []}
    )
    for task, run_idx, result in task_results:
        cat = task.get("category_id", "UNKNOWN")
        ds = domain_stats[cat]
        ds["total"] += 1
        if isinstance(result, Exception):
            ds["exception"] += 1
        elif isinstance(result, dict):
            if result.get("status") == "completed":
                ds["completed"] += 1
            else:
                ds["error"] += 1
            vs = result.get("verify_status", "SKIP")
            if vs == "PASS":
                ds["verify_pass"] += 1
            ds["verify_scores"].append(result.get("verify_score", 0.0))

    print("\n=== Eval done ===", flush=True)
    total_completed = sum(v["completed"] for v in domain_stats.values())
    total_jobs_done = sum(v["total"] for v in domain_stats.values())
    for cat, ds in sorted(domain_stats.items()):
        avg_score = (
            sum(ds["verify_scores"]) / len(ds["verify_scores"])
            if ds["verify_scores"] else 0.0
        )
        print(
            f"  {cat:6s}: {ds['completed']}/{ds['total']} completed | "
            f"verify_pass={ds['verify_pass']} avg_score={avg_score:.3f} | "
            f"error={ds['error']} exception={ds['exception']}",
            flush=True,
        )
    print(f"  Total : {total_completed}/{total_jobs_done} (tasks={len(tasks)}, runs={runs})", flush=True)

    Path(result_dir).mkdir(parents=True, exist_ok=True)
    llm_usage = summarize_llm_usage_from_files(
        result_path,
        [str(task["task_id"]) for task in tasks],
        run_indices,
    )
    _print_llm_usage(llm_usage)

    run_meta = {
        "tasks_dir":  tasks_dir,
        "task_ids_file": task_ids_file,
        "model":      model,
        "workers":    workers,
        "hostname":   hostname,
        "isolation":  use_isolation,
        "runs":       runs,
        "started_at": started_at.isoformat(timespec="seconds"),
        "ended_at":   ended_at.isoformat(timespec="seconds"),
        "duration_s": duration_s,
    }
    summary_path, report_path = generate_outputs(
        Path(result_dir), tasks, run_meta, max_steps,
    ) if write_report else (None, None)
    if write_report:
        print(f"summary → {summary_path}", flush=True)
        print(f"report → {report_path}", flush=True)
    else:
        print("[--no-report] skipping summary.json/report.md generation", flush=True)

    # Print error log summary if any errors occurred
    error_log = Path(result_dir) / "errors.log"
    if error_log.exists():
        error_count = sum(1 for line in open(error_log) if line.startswith("["))
        print(f"\n⚠️  {error_count} errors recorded → {error_log}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SaaS-Bench eval harness")
    p.add_argument("--tasks-dir", required=True,
                   help="Task directory root (containing Software/ Business/ Healthcare/ Teamwork/ subdirectories)")
    p.add_argument("--model", default="qwen/qwen3.6-plus")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--result-dir", default="results")
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--hostname", default="localhost",
                   help="Hostname the agent uses to access apps")
    p.add_argument("--task-ids", dest="task_ids_file", metavar="PATH",
                   help="Path to a file containing task ids to run")
    p.add_argument("--apps-yaml", default="saas_bench/apps.yaml")
    p.add_argument("--no-isolation", action="store_true",
                   help="Do not start Docker container isolation (share already-running apps)")
    p.add_argument("--runs", type=int, default=1,
                   help="Number of independent repetitions per task (used for pass@k, k=runs)")
    p.add_argument("--run-start", type=int, default=0,
                   help="Starting run_idx; output file suffix is _r{run_start}.._r{run_start+runs-1} (used when re-running specific runs)")
    p.add_argument("--no-report", action="store_true",
                   help="Skip re-generating summary.json/report.md (avoids overwriting existing aggregates when re-running partial runs)")
    p.add_argument("--rerun-existing", action="store_true",
                   help="Run selected tasks even when their result JSON files already exist")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        tasks_dir    = args.tasks_dir,
        model        = args.model,
        workers      = args.workers,
        result_dir   = args.result_dir,
        max_steps    = args.max_steps,
        hostname     = args.hostname,
        task_ids_file= args.task_ids_file,
        apps_yaml    = args.apps_yaml,
        use_isolation= not args.no_isolation,
        runs         = args.runs,
        run_start    = args.run_start,
        write_report = not args.no_report,
        rerun_existing = args.rerun_existing,
    )
