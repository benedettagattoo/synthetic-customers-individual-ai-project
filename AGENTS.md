# AGENTS.md

Standing instructions for the agent working in this project.

## Folder layout
- `data/raw/` – the dataset exactly as given. Read only.
- `data/derived/` – everything the code makes from the raw data.
- `src/` – scripts, numbered in the order they run (`01_clean.py`, `02_describe.py`, ...).
- `results/` – every table and figure, each made by exactly one script.
- `plan.md` – the question, the theory and the planned test.
- `mistakes.md` – where the agent went wrong and how it was found.
- `data/README.md` – description of every column, its meaning and units (read this before touching the data).

## Rules
1. Never modify anything in `data/raw/`.
2. After every filter, merge or drop, print the number of rows and the reason.
3. Every number in a write-up comes from a file in `results/`; name the script that made it.
4. Follow `plan.md`. Ask before changing anything in it, and list every analysis choice you make.
5. Fix random seeds, and log the prompt, output and model name of every model call.
6. No causal claims unless the design supports them. Say "I am not sure" when you are not.
7. Run Python with `uv run python ...` so the project's environment is used.

## Added rules (one per mistake the agent should not repeat)
<!-- Add a new rule here every time the agent makes a mistake -->
- Always run Python with `uv run python ...`. uv is installed at /Users/benedettagatto/.local/bin/uv; if uv is not found, use that full path. Never use the system python3.
- Never describe a set of columns as a range (e.g. QID9_1–QID9_40) without checking every column exists; list any missing ones.
- Never open, print, copy or quote the `.env` file or the API key. Scripts load the key from the environment (e.g. python-dotenv); never write it into code, logs, results or chat.
