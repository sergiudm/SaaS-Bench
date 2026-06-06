# Trajectory Quality Rubric

Use this rubric after running the helper inventory. It is a guide for careful manual review, not a replacement for reading the raw trajectory.

## Quality Dimensions

Score each reviewed task qualitatively as High, Medium, Low, or Artifact Missing.

| Dimension | High quality | Medium quality | Low quality |
|---|---|---|---|
| Task understanding | Correctly identifies all apps, entities, constraints, dates, amounts, files, and final states. | Captures the main goal but misses secondary validations or exact constraints. | Pursues the wrong workflow, wrong entity, wrong app, or fabricated goal. |
| Planning and decomposition | Maintains a useful checklist and works in the dependency order implied by the task. | Uses a rough plan but loses track of dependencies or skipped requirements. | Wanders, repeats vague goals, or lacks a usable task decomposition. |
| Navigation and app handling | Logs into required apps, uses stable routes/search, and switches apps deliberately. | Reaches most apps but wastes many steps or confuses app context. | Gets stuck before core work, repeatedly targets wrong UI, or fails basic login/navigation. |
| State-changing actions | Performs required creates/edits/approvals/uploads/payments/messages with correct data. | Performs some state changes but with incomplete or uncertain data. | Mostly clicks, searches, or reads without completing required state changes. |
| Verification behavior | Checks important intermediate/final states against the task before moving on or finishing. | Performs occasional checks but accepts weak evidence. | Claims completion without inspecting resulting state or ignores failed checks. |
| Recovery from friction | Changes strategy after failures, uses search/filter/direct URLs reasonably, and avoids repeated ineffective actions. | Recovers from some failures but repeats several ineffective attempts. | Loops on the same action, ignores errors, or consumes the budget without a strategy shift. |
| Efficiency | Makes steady progress with action counts appropriate to task complexity. | Some wasted steps but progress remains clear. | Max-step or near-max-step behavior dominated by loops, waits, or repeated clicks. |
| Data fidelity | Preserves exact names, amounts, dates, email addresses, titles, filenames, statuses, and body text. | Minor formatting issues or one uncertain detail. | Wrong entity, wrong field values, missing required text, or hallucinated values. |
| Self-calibration | Final output accurately reports completion status, blockers, and uncertainty. | Partially accurate but overstates or understates some progress. | Claims success despite failed/missing requirements, or gives no useful final status. |

## Severity Labels

- Critical: missing/corrupt artifacts, zero-step/crashed run, verifier error, no trajectory, wrong task/app, or self-reported success with verifier failure and no evidence.
- High: max-step loop, major required state never attempted, repeated UI targeting failure, or wrong data written to live app.
- Medium: partial progress with clear omissions, weak verification, inefficient navigation, or minor but consequential data mismatch.
- Low: small inefficiencies, harmless retries, noisy thoughts, or cosmetic report inconsistencies.

## Required Evidence Table

For each manually reviewed task, include a compact table like:

| Requirement/check | Trajectory evidence | Verifier result | Judgment |
|---|---|---|---|
| Approve HRMS claim | Steps 7-35 opened claim; no submitted/approved state observed | FAIL: status=Draft | Low: attempted but no state change |

Use step ranges, not only task-level summaries.

## Sampling Larger Experiments

If the run has more than 25 task runs, audit:

- all missing-primary, missing-verifier, zero-step, verifier-error, and malformed JSON cases
- all `done(success=true)` or success-claim runs that failed verification
- all max-step or near-max-step runs
- top repeated-loop and error-rate outliers
- all perfect passes, if any, to confirm whether success is clean or accidental
- at least two tasks per domain and one task per app when available
- at least three mid-score partial-credit tasks, because they often reveal verifier/task ambiguity

State the sampling rule in the report. Do not imply unreviewed tasks were manually inspected.

## Failure Taxonomy

The final report must organize root causes into exactly these three top-level sections. A task can appear in more than one section when multiple causes contributed, but identify one primary cause when possible.

### 1. Infra-Level Errors

Use this section for failures in the benchmark harness, runtime environment, application services, browser/session layer, input assets, artifacts, or verifier infrastructure. These failures make the trajectory missing, incomplete, unreliable, or impossible to fairly judge independent of the agent's decisions.

Common subcategories:

- Missing artifacts: no primary trajectory JSON, no verifier JSON, missing `summary.json`, absent screenshots/files, or missing multimodal inputs.
- Malformed artifacts: invalid JSON, truncated trajectories, inconsistent task IDs, missing `trajectory`, missing `checks`, or run status inconsistent with files.
- Zero-step or crashed runs: summary reports zero steps, process exits before browser work, worker crash, interrupted run, or empty output caused by harness failure.
- Container/app availability: Docker service not started, app health failure, wrong port, hostname unreachable, database unavailable, app setup/reset failure, or stale container state.
- Browser/session failures: browser launch crash, tab/session loss, persistent timeout unrelated to app complexity, automation protocol failure, or unavailable browser dependencies.
- Credential/config problems: wrong seeded credentials, missing environment variables, login disabled by setup drift, or task points to an unavailable app instance.
- Input/data setup problems: missing task fixture, missing uploaded file, wrong seeded records, contaminated state from previous tasks, or fixture mismatch with description/verifier.
- Verifier infrastructure errors: verifier cannot connect to app/database, SQL exception, verifier dependency missing, verifier crashes before checking state, or verifier returns ERROR/SKIP for infrastructure reasons.
- Report generation/accounting errors: summary statistics inconsistent with artifacts, missing per-task usage stats due to run collection failure, or benchmark report omits existing runs.

Evidence to include:

- File paths, run IDs, exception text, return codes, missing filenames, service URLs/ports, verifier status/error fields, and whether any meaningful trajectory steps exist.

### 2. LLM API Errors

Use this section for failures in model-provider calls used by the agent or by verifier LLM/vision judges. Keep these separate from generic infra because the fix is usually quota/model/provider handling, not browser-agent behavior.

Common subcategories:

- Quota or billing limits: HTTP 429, spending cap exceeded, token quota exhausted, or provider-side rate limit.
- Authentication/authorization: missing API key, invalid key, expired credential, 401, 403, or project not authorized for the requested model.
- Model availability: requested model not found, model deprecated, model endpoint unavailable, unsupported region, 404 model/judge errors, or unavailable vision/judge model.
- Provider outages/server errors: 5xx errors, service unavailable, transient provider failure, or repeated provider-side timeouts.
- Context/token failures: prompt exceeds context window, max output/token limit prevents valid response, response truncated before tool/action JSON, or provider rejects image/file payload size.
- Response format failures: malformed provider response, invalid tool-call payload caused by provider output, empty completion, or refusal/safety block unrelated to the benchmark task.
- Usage/pricing telemetry gaps: missing usage records, unpriced calls, or partial cost accounting when this affects evaluating run reliability.
- LLM judge failures: verifier `llm_judge` or `llm_judge_vision` returns API error, rate limit, unavailable model, or transport error.

Evidence to include:

- Provider error text, HTTP status, model name, task step or verifier check where the API failed, whether the failure stopped the agent or only affected verification, and whether retry/backoff was attempted.

### 3. Agent Capabilities Issues

Use this section for observable shortcomings in the agent's reasoning, UI operation, tool use, state mutation, recovery, or calibration. These failures are attributable to the trajectory even when the infrastructure and LLM API are available.

Common subcategories:

- Task comprehension errors: wrong goal, wrong app, wrong entity, missed constraints, skipped required intermediate validation, or hallucinated requirement.
- Planning/decomposition failures: no usable checklist, wrong dependency order, losing track of completed subtasks, or failure to break multi-app workflows into durable milestones.
- Navigation/app-context failures: cannot find target pages, overuses page switching, loses the active app, opens irrelevant routes, or violates expected site navigation discipline.
- DOM/UI targeting failures: repeated clicks on unstable indices, wrong element selection, failure to use search/filter/direct URLs, or inability to operate dynamic forms.
- Form/data-entry failures: wrong field, missing required field, bad dropdown choice, wrong date/amount/status/text, failure to save/submit, or overwritten existing values.
- State-mutation failures: reads existing data but never creates/edits/approves/uploads/pays/sends the required final state.
- Cross-app coordination failures: completes one system but fails to carry required data into another app, misses dependency handoff, or creates inconsistent records across apps.
- Verification failures: assumes success after a click, does not inspect reports/lists/statuses, ignores failed save messages, or fails to compare final state against task requirements.
- Recovery failures: loops on the same action, ignores errors, repeats ineffective strategies, fails to escalate to alternate UI routes, or consumes the budget without changing tactics.
- Multimodal/perception failures: misidentifies image/PDF content, misses visual details needed for the task, or uses unsupported visual assumptions.
- Code/file/tool-use failures: edits `todo.md` instead of app state, misuses file tools, fails to use available browser/search/extract tools, or makes non-task file changes.
- Memory/context failures: forgets task-specific names/amounts/dates, contradicts previous observations, or confuses records between tasks/apps.
- Efficiency/time-budget failures: excessive waits, low-value scrolling/searching, near-max-step runs dominated by repeated actions, or no prioritization of verifier-critical requirements.
- Self-calibration failures: `done(success=true)` or final output claims success despite failed/missing requirements, hides uncertainty, or gives no useful blocker summary.

Evidence to include:

- Step ranges, observed action/result patterns, failed verifier checks, exact data mismatches, repeated spans, final-output claims, and the missed task requirement.

## Report Tone

Be direct and evidence-based. Prefer "The trajectory reached X but never did Y" over broad claims like "bad planning." When uncertain, explain the uncertainty and what evidence would resolve it.
