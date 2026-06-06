# AGENTS.md

## Project Overview

SaaS-Bench is a Python benchmark harness for evaluating browser-driving agents on self-hosted SaaS workflows. The core package lives in `saas_bench/`; task definitions and verifiers live under `tasks/`; Docker assets and compose templates live under `docker/`; operational scripts live under `scripts/`.

The reference agent is implemented in `saas_bench/agent.py` and uses `browser-use`. The harness entry point is `saas_bench/run.py`, exposed as the `saas-bench` console script.

## Environment

- Use `uv` for the Python environment and dependency management.
- Python must be `>=3.10`.
- Prefer running Python commands through `uv run`.
- On this machine, ripgrep (`rg`) is not installed. Use alternatives such as `grep`, `find`, `sed`, or Python-based tooling when searching files.
- Bubblewrap (`bwrap`) is also unavailable. If sandboxed shell commands fail with a bubblewrap/bwrap error, retry with the appropriate approved or escalated command path rather than treating it as a project failure.
- Install this package in editable mode with:

```bash
uv pip install -e .
```

- Install browser dependencies with:

```bash
uv run playwright install chromium
```

- This workspace uses a local `browser-use` checkout at:

```text
/data-2u-2/qijun/browser-use
```

When debugging or reinstalling `browser-use`, prefer that local checkout over the PyPI package:

```bash
uv pip install -e /data-2u-2/qijun/browser-use
```

Do not vendor or replace the local `browser-use` checkout unless the user explicitly asks. If edits are needed there, treat it as a separate local project outside this repository.

## Common Commands

- Run the benchmark through the project script:

```bash
bash scripts/run.sh
```

- Run selected tasks:

```bash
bash scripts/run.sh --tasks-dir tasks/uni-m/Business
bash scripts/run.sh --task-ids task_ids.txt
```

- Clean up stale containers from a crashed or interrupted run:

```bash
bash scripts/stop_all.sh
```

- Load downloaded Docker image archives:

```bash
bash scripts/load_images.sh
```

- Lightweight Python validation:

```bash
uv run python -m compileall saas_bench
```

There is no dedicated test suite in the current repository snapshot, so validate changes with targeted compile checks and, when practical, a focused benchmark run.

## Development Notes

- Keep changes scoped to the harness, task, or script being worked on.
- Preserve the `run_task(task, model_name, prompt, result_dir, max_steps, slot_id, todo_md) -> dict` contract for agent implementations.
- Result dictionaries should include `status`, `agent_output`, and `trajectory`.
- Verifiers inspect live Docker application state; do not make harness changes that bypass task `verify.py` behavior.
- Be careful with generated outputs under `results/` and runtime container state. Do not delete run artifacts unless the user asks.
- The repository may have local user edits. Check `git status --short` before modifying files and do not revert unrelated changes.
