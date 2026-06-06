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

## Common SaaS-Bench Failure Modes

- Dynamic DOM index loops: repeated click/input attempts at huge or changing indices without changing strategy.
- Navigate-discipline problems: many page opens or app switches that do not correspond to required sites.
- Read-only partial credit: agent inspects existing records that already satisfy some checks but never performs required mutations.
- Premature completion: `done` action or final output claims success before verifier-critical state exists.
- Data drift: correct workflow with wrong date, amount, title, assignee, body text, file, status, or account.
- Cross-app dependency break: completes an earlier app but never carries required data into later apps.
- Weak confirmation: assumes a save/submit worked after clicking but does not inspect status, report, or list state.
- Tool/file distraction: spends many steps editing `todo.md` or reading files without app progress.
- Artifact failure: no primary trajectory, missing verify file, malformed JSON, or run status inconsistent with files.

## Report Tone

Be direct and evidence-based. Prefer "The trajectory reached X but never did Y" over broad claims like "bad planning." When uncertain, explain the uncertainty and what evidence would resolve it.
