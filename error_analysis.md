# Error analysis

Benedetta Gatto (3341868) · Individual AI Project, 20595 Business Analytics · Idea 12, Synthetic customers

This document analyses the errors found during the project: where they were, how they were found and what they imply. The raw, dated log is `mistakes.md` (entries 1–37 plus session 10). Entry numbers below refer to that file. Errors are counted whoever made them: the OpenCode agent (GLM-5.3-Flash Free), the dataset's own documentation, Claude (which I used as a tutor) and me.

## 1. Summary

37 numbered entries in 10 sessions. 34 involve the agent (one of them, entry 36, a rule it proposed and I approved), 1 is an error in the dataset documentation (12), 1 is Claude's (11) and 1 is mine (4); session 10 adds set-up errors of mine. The agent's direct counts (rows, missing columns) were almost always right. Its errors sat where a result has to be *described*, *interpreted* or *validated*.

| Kind of error | Entries | How they were found | What they imply |
|---|---|---|---|
| A. Harness and environment | 1, 2, 5 | Comparing the agent's claims with what I knew of the set-up; testing the permission prompts | Check the harness, not only the answers |
| B. Descriptions that do not match the evidence | 3, 6, 7, 8, 9, 14, 15, 24 | Hand checks against the raw files; re-doing the arithmetic; comparing text with the agent's own tables | Every number and range in a write-up needs a check against its source |
| C. Wrong reading of what the data mean | 12, 13, 16, 17 (and 11, by Claude) | Testing two interpretations against the raw survey file; reading the whole plan after each edit | Assumptions about data must be *tested*, not agreed on |
| D. Code and edits that run but are wrong | 10, 18, 19, 20, 21, 22, 23, 26, 31, 32 | Code review before running; dry run; pilot; assertions and internal consistency checks; independent recomputation | "It ran without errors" is not evidence of correctness |
| E. Validation and fallback rules | 25, 36, 37 | Reading the raw outputs of every invalid record; tracing a log message back to its cause | A validator is code too: test it on real outputs of each model |
| F. Over-interpretation of correct numbers | 27, 28, 33, 34, 35 | Re-doing the arithmetic; refitting models; looking at CI widths and figures | The numbers can be right while the sentence is wrong |
| G. Independence of checks | 29, 30 | Row-by-row comparison of two runs; searching the agent's files for a number it quoted | Agreement between runs or tools is not confirmation unless they are independent |
| H. My own errors | 4, 36 (approval), session 10 set-up | Checking the source before blaming; re-reading my own rules | Human review can fail too |

## 2. Where the errors were and how each was found

| # | Where | What went wrong | How it was found | Fix / consequence |
|---|---|---|---|---|
| 1 | Session 1, set-up | Said uv was not installed and used system python3 | I knew it was installed; `which uv` in Terminal | Rule with the full path added to AGENTS.md |
| 2 | Session 1 | Ran Python while in Plan mode | Its own report mentioned "ad-hoc Python inspection" | Noted; Plan/Build used deliberately afterwards |
| 3 | First data report | Claimed a 135-column overlap when the smaller file has 127 columns | Re-did the arithmetic by hand | Corrected |
| 5 | Harness | Never asked permission before acting (default settings) | Noticed no prompt ever appeared | opencode.json with edit/bash/webfetch = "ask"; verified on 10 Nov |
| 6 | First data report | Described QID287 as columns _1–_12; _8 and _9 do not exist | Hand check C on the CSV | AGENTS.md rule: never describe columns as a range without checking |
| 7 | data/README.md | Broke that rule again ("QID9_1 … QID9_37") | Read the draft | Fixed; instructions alone did not guarantee compliance |
| 8, 9 | data/README.md | Wrong section reference; called pid a real identifier; column count off by one | Checked plan.md and the CSV | Fixed |
| 10 | 01_prepare.py proposal | uuid4 IDs claimed to be "seeded" (they are not) | Read the proposal before approving | IDs C001–C200 from a seeded shuffle |
| 12 | Dataset documentation | Catalog lists respondent 1's products and prices for each column, as if fixed | Two "$0.00" products made no sense next to ~45% No; tested both mappings on the raw file | Products tracked by item number; 79 malformed strings excluded |
| 13 | plan.md | After an edit, sections 4 and 5 contradicted section 3 | Read the whole plan after the edit | Fixed |
| 14, 15 | plan.md | Wrong price example; two numbers from its own table mixed up | Compared with its own tables | Fixed |
| 16, 17 | Synthetic prompt design | Zero price shown as "$0.00", not "$0 (product is free)" as in the survey; a rule dropped silently | Compared the proposal with the raw file | Prompt uses the exact survey text |
| 18 | Verification script | Check with three bugs; it reported only two | I gave the expected count (7,493), checked beforehand | Fixed; result 7,493 |
| 19 | plan.md edit | Briefly deleted the Model block of section 4, then restored it | Its own report; I checked the final file | Block present in the final plan |
| 20 | 02_generate.py | Declared ready after a syntax check; would have crashed on the first call | Code review, then reproduced the TypeError | Fixed; `--dry-run` added |
| 21 | 02_generate.py | Twins processed by income bracket, which would confound daily model drift with income | Code review | Random order (sorted by new ID) |
| 22 | 02_generate.py | A persistent API error would have burned the daily quota | Code review | Stop on 4xx and after 3 consecutive errors |
| 23 | 02_generate.py | Assumed the response field holds the dated model version | Pilot output (false warnings) | Version checked once with models.get |
| 24 | Run planning | "470 − 30" when 470 was already net | Re-did the sum | Corrected |
| 25 | Validation (Gemini) | "$0" rejected; 20 keys excluded for format only | Read the raw outputs of every invalid record | Zero-price aliases; recovered offline, no new calls |
| 26 | Quota counter | Claimed to count the quota correctly; Google counted more | A 429 arrived at 492 counted requests | Harmless (429 stops the run); noted |
| 27, 28 | Real-data summary | "57 pp unexplained"; negative free coefficient read as "free does not matter" | Re-did the arithmetic; refitted the models and looked at the figure | Reported as a limitation of the linear model |
| 29 | Agent-on-its-own run | Not independent: copied every choice from plan.md section 7 | Row-by-row comparison; identical to the last digit | Re-run on 17 Nov in a clean repository (session 10) |
| 30 | Agent-on-its-own run | Presented a number from plan.md as its own result | Searched its scripts and outputs, then plan.md | Recorded |
| 31 | 07_compare_twins.py | Read the real answers for both groups; perfect agreement | Contradiction between identical cell shares and different overall rates | Explicit column; assertion that the two matrices differ |
| 32 | 08_compare_curves.py | Misaligned dummies in the pooled regression | Agent's own cross-check (pooled vs per-item slopes) | Fixed; numbers reconcile exactly |
| 33–35 | Comparison write-up | Narrow synthetic CIs read as more information; LPM slopes below −1 read literally; H2 "rejected" despite low power | Reading the output against the design; looking at CI widths | Reported with these caveats |
| 36 | HPC robustness run | Fallback rule (proposed by the agent, approved by me) switched the schema for ~97% of calls because of a formatting difference | Noticed "format wrapper" in the job log and traced it back | Format recorded per line; results checked by format: no difference |
| 37 | Validation (gpt-oss) | Prices without "$" rejected; 16 keys excluded; the earlier fix listed strings instead of matching values | Read the raw outputs after copying the file | Offline re-validation by value, rule fixed before any answer was analysed |

Errors not made by the agent: **11** (Claude assumed a product mapping without testing it; the agent corrected it), **4** (I blamed the agent for a statement that was in the dataset's README), and the **session 10 set-up** (I placed the clean folder inside the project, left leaks in the copied plan.md and forgot a README).

## 3. How the errors were analysed

For every error I recorded where it was, how it was found, the cause where known, and the fix. Then I asked three questions.

**Would it have changed a result?** The errors that would have changed results were 12 (wrong products), 31 (synthetic = real), 32 (wrong pooled slope), 25 and 37 (silent exclusions concentrated on a few products) and 29 (a false independence check). All were found before any number was used. The errors in category B would have made the write-up wrong without changing the analysis.

**Which check caught it?**

| Check | Errors it caught |
|---|---|
| Hand check against the raw files | 3, 6, 12, 16 |
| Re-doing arithmetic / comparing with the agent's own tables | 3, 14, 15, 24, 27 |
| Reading proposals and plans before approving | 10, 13, 17 |
| Code review before running | 20, 21, 22 |
| Dry run and pilot | 23 (and the step pattern, seen in the pilot) |
| Reading raw outputs of invalid records | 25, 37 |
| Assertions and internal consistency checks | 31, 32 |
| Independent recomputation | 28, 31; confirmed all headline numbers |
| Row-by-row comparison of runs | 29 |
| Independent re-run in a clean repository | 29 (repaired) |

**Why did it happen?** Four causes recur:
1. The agent wrote descriptions without checking them against its own outputs (B, F).
2. It trusted the documentation and the first plausible reading of the data (C).
3. It treated "runs without errors" as "correct" (D).
4. It generalised from the examples it had seen instead of from the requirement (E: listing price strings instead of matching price values).

Rules in AGENTS.md reduced but did not stop repeats (entry 7 after rule 6). What did work was making the check part of the procedure: the dry run, the pilot, assertions, saving every raw output, and recomputing headline numbers outside the agent. The headline number was recomputed by me in Numbers, without any agent.

## 4. What this implies for relying on the agent

- Trust direct counts; check every description, range and interpretation against its source.
- Do not accept "it ran" as evidence: require a dry run, a small pilot and an assertion that would fail if the result were wrong.
- Test validators on real outputs of each model, and match by value, not by exact string.
- When testing independence, make sure the agent cannot read the earlier choices.
- Record who made each error. A second AI tool, the dataset documentation and I were wrong at least once each.
