#!/bin/bash
#SBATCH --job-name=robust-gptoss20b
#SBATCH --partition=stud
#SBATCH --account=3341868
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=robustness_%j.out
#SBATCH --error=robustness_%j.err

# run_ollama.sbatch — robustness run on the Bocconi HPC (plan.md section 5;
# 15 Nov 2026 choices in section 7).
#
# Starts a local Ollama server, checks the Ollama version and the
# gpt-oss:20b model digest BEFORE any call, runs 06_generate_ollama.py
# (same directory: prompts_for_robustness.jsonl must be here), then stops
# the server.
#
# Smoke test (3 calls, same robustness_answers.jsonl — valid records count,
# the full run skips them):
#   sbatch --export=ALL,SMOKE=3 run_ollama.sbatch
# Full run:
#   sbatch run_ollama.sbatch

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "node: $(hostname), job $SLURM_JOB_ID, start: $(date -Is)"
nvidia-smi -L 2>/dev/null || echo "nvidia-smi not available"

export PATH="$HOME/ollama/bin:$PATH"
export OLLAMA_KEEP_ALIVE=12h

ollama serve > ollama_serve.log 2>&1 &
SERVER_PID=$!
cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Pre-call checks (stdlib python3 only): wait for the server, assert the
# Ollama version, assert the gpt-oss:20b digest from /api/tags starts with
# f38aa0c53da5. Any failure stops the job before a single call is made.
# (The digest is read from /api/tags: /api/show's documented response has
# no digest field.)
# ---------------------------------------------------------------------------
python3 - <<'PYEOF'
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:11434"
EXPECTED_VERSION = "0.40.16"
MODEL = "gpt-oss:20b"
DIGEST_PREFIX = "f38aa0c53da5"


def get(path, timeout=10):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.loads(r.read().decode())


version = None
for _ in range(60):
    try:
        version = get("/api/version")["version"]
        break
    except Exception:
        time.sleep(2)
if version is None:
    sys.exit("FATAL: Ollama server did not come up within 2 minutes")
print("Ollama version:", version, flush=True)
if version != EXPECTED_VERSION:
    sys.exit(f"FATAL: Ollama version {version!r} != {EXPECTED_VERSION!r} "
             "- stopping before any call")

tags = get("/api/tags")
models = tags.get("models", [])
matches = [m for m in models if m.get("name") == MODEL]
if not matches:
    sys.exit(f"FATAL: {MODEL} not found in /api/tags "
             f"(models present: {[m.get('name') for m in models]})")
digest = matches[0].get("digest", "")
print("digest:", digest, flush=True)
if not digest.startswith(DIGEST_PREFIX):
    sys.exit(f"FATAL: {MODEL} digest {digest!r} does not start with "
             f"{DIGEST_PREFIX} - stopping before any call")
print("digest check passed", flush=True)
PYEOF

# ---------------------------------------------------------------------------
# Run the generation script (SMOKE=3 -> first 3 not-done calls only)
# ---------------------------------------------------------------------------
if [ -n "${SMOKE:-}" ]; then
  case "$SMOKE" in
    ''|*[!0-9]*) echo "SMOKE must be an integer"; exit 1 ;;
  esac
  echo "SMOKE mode: first $SMOKE not-done calls only"
  python3 06_generate_ollama.py --limit "$SMOKE"
else
  python3 06_generate_ollama.py
fi

cleanup
trap - EXIT
echo "job complete: $(date -Is)"
