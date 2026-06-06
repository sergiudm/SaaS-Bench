---
name: audit-trajectories
description: Systematically audit SaaS-Bench experiment trajectories and write detailed trajectory quality reports. Use when the user asks to inspect, review, grade, diagnose, compare, or summarize the quality of agent trajectories in experiment result directories such as results-subset/gemini-3.5-flash or results/model-name, especially when trajectory JSON files, *_verify.json files, summary.json, task descriptions, or verifier checks must be analyzed together.
---

# Audit Trajectories

## Overview

Audit SaaS-Bench runs by combining automated inventory with manual evidence-based trajectory review. Treat verifier JSON, task descriptions, and raw step-by-step trajectories as separate evidence sources; never treat the agent's final output or an existing aggregate report as ground truth.

## Workflow

1. Confirm the experiment directory, for example `results-subset/gemini-3.5-flash`.
2. Run the helper inventory from the repository root:

```bash
uv run python .codex/skills/audit-trajectories/scripts/summarize_experiment.py <experiment-dir> --tasks-root tasks --output <experiment-dir>/trajectory_quality_report.md
```

If `uv` is unavailable or the user only needs a quick local probe, use `python3` with the same arguments; the script uses only the standard library.

3. Read `references/trajectory_quality_rubric.md` before doing the manual audit.
4. Inspect the generated report to identify missing files, zero-step runs, verifier errors, max-step runs, overconfident completions, loops, high error rates, and unusual action distributions.
5. Manually review all tasks when the experiment has 25 or fewer task runs. For larger experiments, review every critical anomaly plus a stratified sample covering each domain, app, score bucket, and completion state.
6. For each reviewed task, load:
   - `<task_id>_rN.json`
   - `<task_id>_rN_verify.json`
   - matching `tasks/**/<task_id>/description.md`
   - matching `tasks/**/<task_id>/meta.json`
   - `tasks/**/<task_id>/verify.py` only when verifier behavior or scoring details need clarification
7. Write the final report in the output path requested by the user, or use `<experiment-dir>/trajectory_quality_report.md` by default.

## Manual Audit Procedure

For each selected task:

1. Reconstruct the task checklist from `description.md` and the verifier checks. Include exact entity names, dates, amounts, files, app names, and required final states.
2. Build an evidence table mapping each requirement to: trajectory steps that attempted it, verifier check result, and your quality judgment.
3. Read the trajectory in focused windows:
   - first 10 steps for setup, login, and initial plan
   - last 20 steps for termination and final state
   - every `done` action and the 5 steps before it
   - every step with an error-like result and the 3 steps before/after it
   - every app/site transition
   - repeated-action spans and high-click-index spans flagged by the helper
   - steps mentioning task-specific entities from the description or verifier detail
4. Judge behavior quality, not just pass/fail. Note whether the agent understood the objective, made correct state changes, verified intermediate state, recovered from UI friction, avoided loops, preserved data fidelity, and calibrated its final answer to evidence.
5. Separate trajectory failures from possible harness/verifier issues. Mark verifier concerns only when the trajectory contains strong evidence of a correct state that the verifier appears to reject, or when verifier output is missing/erroring.

## Report Requirements

Include these sections unless the user asks for a narrower report:

- Executive summary: overall trajectory quality, confidence, and the most important failure modes.
- Scope and methodology: experiment path, task/run counts, files inspected, sampling policy if not all tasks were reviewed.
- Dataset inventory: missing primary trajectories, missing verifier files, zero-step runs, verifier errors, and incomplete artifacts.
- Quantitative signals: pass rate, score distribution, step statistics, action distribution, done/self-report calibration, loop/error indicators.
- Failure taxonomy: three required sections named `Infra-Level Errors`, `LLM API Errors`, and `Agent Capabilities Issues`, each with root causes, task examples, evidence, impact, and recommended fixes.
- Per-task audits: concise but detailed notes for every reviewed task, including requirement coverage and key trajectory evidence.
- Verifier and harness concerns: suspected false positives/negatives, missing artifacts, crashed runs, or malformed outputs.
- Recommendations: concrete agent, prompt, harness, verifier, or benchmark-process changes, prioritized by expected impact.
- Appendix: paths to inspected files and any commands run.

## Evidence Rules

- Cite file paths and step numbers for important claims.
- Quote only short trajectory snippets; summarize long traces.
- Distinguish observed facts from interpretations.
- Do not inflate a task's quality because the final `agent_output` claims success.
- Do not mark a task as low quality solely because it failed verification; explain the behavioral reason.
- Check whether partial credit reflects meaningful progress or accidental state.
- When a trajectory is absent, report artifact quality separately from agent behavior.
- Classify failures by root cause. Use infra-level errors for harness, artifact, container, app, browser, verifier, or input failures; LLM API errors for provider, quota, model, judge, or token/response service failures; and agent capabilities issues for observable planning, navigation, tool-use, data-entry, verification, recovery, or self-calibration weaknesses.

## Helper Script

Use `scripts/summarize_experiment.py` to create a first-pass markdown report. The script finds primary result JSON files, matching verifier JSON files, task metadata, summary entries, action counts, likely loops, error-like results, done actions, missing artifacts, and self-report mismatches. Treat its anomaly flags as leads for manual review.
