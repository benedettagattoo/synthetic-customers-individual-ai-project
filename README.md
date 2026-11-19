# Synthetic customers: do LLM twins answer like real respondents?

Individual AI Project, 20595 Business Analytics, Bocconi University (Prof. Abhinav Pandey). Benedetta Gatto, idea 12 ("Synthetic customers").

**Question.** Do LLM-generated synthetic customers, given only the demographics of real survey respondents, reproduce the income-group differences and the price sensitivity seen in the real data?

**Answer in short.** No. Synthetic twins answer in a strict step pattern (Yes up to a threshold price, No above it), are about twice as price sensitive as real people and always take a free product. This holds for two models from different companies (gemini-3.5-flash-lite and gpt-oss:20b), but the twin of the same person changes a lot between models. Real income gaps are close to zero, so the income hypotheses can hardly be tested.

| | Real respondents | Gemini twins | gpt-oss twins |
|---|---|---|---|
| Yes when the product is free | 86.3% (1,495 / 1,732) | 100% | 99.95% |
| Single-step answer ladders | not defined (one price each) | 100% | 90.6% |
| Pooled slope of Yes on relative price | −0.686 | −1.398 | −1.250 |
| Yes rate at the real respondent's price | 47.5% | 51.6% | 57.2% |

Sources: `results/real_slopes.csv`, `results/compare1_hypotheses.csv`, `results/compare2_hypotheses.csv`, `results/robustness_summary.csv`.

## Where to find what

| What | Where |
|---|---|
| Final report (analysis, verification, reflection) | `report.pdf` |
| Record of agent errors and how they were found | `mistakes.md` |
| Question, theory, hypotheses, falsification rules and the log of every analysis choice | `plan.md` (section 7 is the dated choices log) |
| Standing instructions for the agent | `AGENTS.md` |
| Description of every data column | `data/README.md` |
| Scripts, in the order they run | `src/`, `hpc/` |
| Every table and figure | `results/` |
| Agent-on-its-own run of 12 Nov (not independent, see `mistakes.md` entry 29) | `src/agent_*.py`, `results/agent_run/` |
| Independent re-run of 17 Nov (fresh session, clean folder, own choices) | `blind_run/` (its choices: `blind_run/results/choices.md`) |

## Data

Twin-2K-500 (`LLM-Digital-Twin/Twin-2K-500` on Hugging Face, CC BY 4.0): 2,058 US respondents. Income is QID21 (5 brackets); the outcome is Yes/No purchase intention in the wave-4 pricing study, where each respondent saw each of 40 products at one random price from an 11-point ladder. 79 respondents with a malformed price string are excluded from price-based analyses (1,979 valid). Ten products are analysed (items 25, 27, 31, 9, 36, 26, 15, 23, 38, 10). Respondent IDs were replaced by C001–C200 before anything was sent to a model.

## Pipeline

| Script | What it does | Main output |
|---|---|---|
| `src/01_prepare.py` | Stratified sample of 200 respondents (40 per income bracket, seed 20261111), new IDs | `data/derived/sample_200_items.csv` |
| `src/02_generate.py` | Synthetic twins with gemini-3.5-flash-lite: one call per twin and product, all 11 prices in random order (Gabor-Granger; seed 20261112), 2,000 calls over 5 days | `data/derived/synthetic_answers.csv` |
| `src/03_real_curves.py` | Real demand curves by item, income bracket and price | `results/real_demand_curves.csv` |
| `src/04_real_slopes.py` | Real slopes (LPM on relative price + free indicator), bootstrap CIs (B = 2,000) | `results/real_slopes.csv` |
| `src/05_export_prompts.py` | Exports the same 2,000 prompts for the second model | `data/derived/prompts_for_robustness.jsonl` |
| `hpc/06_generate_ollama.py`, `hpc/run_ollama.sbatch` | Robustness run with gpt-oss:20b (Ollama 0.40.16) on the Bocconi HPC | `data/derived/robustness_answers.jsonl` |
| `src/07_compare_twins.py` | Comparison 1: H1–H3 (twin vs real respondent at the same price) | `results/compare1_*.csv` |
| `src/08_compare_curves.py` | Comparison 2: H4–H5 (demand curves and slopes) | `results/compare2_*` |
| `src/09_revalidate_robustness.py`, `src/10a_build_robustness_file.py` | Offline re-validation of gpt-oss outputs (rule in `plan.md`, 17 Nov) | `data/derived/robustness_answers_full.csv` |
| `src/10_robustness_summary.py` | Gemini vs gpt-oss vs real | `results/robustness_summary.csv`, `results/robust_*` |

To reproduce: install [uv](https://docs.astral.sh/uv/), put the raw files in `data/raw/`, and run each script with `uv run python src/<script>.py`. `src/02_generate.py` needs a Google AI Studio key in a local `.env` file (`GEMINI_API_KEY`), which is not in this repository. 07 and 08 re-run on the gpt-oss answers with `--syn-csv data/derived/robustness_answers_full.csv --out-prefix robust_ --model gpt-oss:20b`.

## Models and versions

- Coding agent: OpenCode with GLM-5.3-Flash Free.
- Synthetic answers: `gemini-3.5-flash-lite` (version 3.5-flash-lite-07-2026, temperature 1, Google AI Studio with training off); `gpt-oss:20b` (Ollama 0.40.16, model ID f38aa0c53da5, temperature 1, Bocconi HPC).

## AI assistance

The analysis was carried out with the OpenCode agent; every error it made, and how it was found, is in `mistakes.md`. I also used Claude (Anthropic) as a tutor to plan the steps, write prompts for the agent, review its output and independently recompute several results; these checks are marked "with the help of Claude" in `mistakes.md`. The headline number was recomputed by me by hand, and the Reflection is my own.
