# lm_eval --model local-completions \
#   --model_args model=<MODEL_NAME>,base_url=<BASE_URL> \
#   --tasks <TASK> --include_path <INCLUDE_PATH> \
#   --output_path <OUTPUT_DIR> [--limit <LIMIT>]


# ./run_eval.sh                                  # full run
# LIMIT=20 ./run_eval.sh                         # quick smoke test
# MODEL_NAME=/model/gemma3 BASE_URL=http://localhost:8006/v1 ./run_eval.sh


#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Config (override via env)
# BASE_URL="${BASE_URL:-http://localhost:8006/v1/completions}"
# MODEL_NAME="${MODEL_NAME:-/model/gemma3}"
# TOKENIZER_NAME="${TOKENIZER_NAME:-/home/vasters/titan-ai4trust-deploy/titan-project-private/API/v1/chatbot/model/gemma3}"

# BASE_URL="${BASE_URL:-http://localhost:8096/v1/completions}"
# MODEL_NAME="${MODEL_NAME:-/Mini-Pharos-slerp-merge}"
# TOKENIZER_NAME="${TOKENIZER_NAME:-/home/vasters/pharos/ui/vllm_docker/Mini-Pharos-slerp-merge}"


BASE_URL="${BASE_URL:-http://localhost:8016/v1/completions}"
MODEL_NAME="${MODEL_NAME:-google/gemma-4-31B-it}"
TOKENIZER_NAME="${TOKENIZER_NAME:-google/gemma-4-31B-it}"



TASK="${TASK:-gr_greek_personalities}"
INCLUDE_PATH="${INCLUDE_PATH:-lm_eval_tasks}"
OUTPUT_DIR="${OUTPUT_DIR:-results/gr_personalities}"
LIMIT="${LIMIT:-}"

export OPENAI_API_KEY="${API_KEY:-EMPTY}"
export HF_TOKEN="${HF_TOKEN:-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}"
# venv: recreate if lm_eval can't import (handles copied mac .venv)
# if [ ! -x .venv/bin/python ]; then python3 -m venv .venv; fi
# if ! .venv/bin/python -c "import lm_eval" >/dev/null 2>&1; then
#   rm -rf .venv && python3 -m venv .venv && .venv/bin/pip install lm_eval
# fi

# Server check
echo "Checking server: $BASE_URL"
if ! curl -sS --max-time 5 "$BASE_URL/models" >/dev/null 2>&1; then
  echo "ERROR: server unreachable at $BASE_URL" >&2; exit 1
fi
echo "Server OK ? verify MODEL_NAME ($MODEL_NAME) is listed above."

# Build args
# MODEL_ARGS="model=${MODEL_NAME},base_url=${BASE_URL}"


# MODEL_ARGS="model=${MODEL_NAME},base_url=${BASE_URL},tokenized_requests=False,tokenizer_backend=remote,auth_token=${API_KEY:-EMPTY}"
MODEL_ARGS="model=${MODEL_NAME},base_url=${BASE_URL},tokenizer_backend=huggingface,tokenizer=${TOKENIZER_NAME},auth_token=${API_KEY:-EMPTY}"



CMD=(lm_eval --model local-completions --model_args "$MODEL_ARGS" \
     --tasks "$TASK" --include_path "$INCLUDE_PATH" --output_path "$OUTPUT_DIR" --verbosity "INFO" --log_samples)
[ -n "$LIMIT" ] && CMD=( "${CMD[@]}" --limit "$LIMIT" )

"${CMD[@]}"
echo "Results: $OUTPUT_DIR"   # actual file is results/gr_personalities_<ts>.json
