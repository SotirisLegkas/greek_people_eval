# greek_people_eval

Greek-language biographical QA benchmark. Two halves that are run separately:
a numbered dataset-generation pipeline (`1…11*.py`) and an lm-eval harness.

## Layout
- `1…11*.py`, `check_data.py` — dataset pipeline, run in numeric order (see below)
- `llm_create_qa_dataset.py` — NOT part of the pipeline. An abandoned experiment
  that answers questions in a single LLM call. Ignore it unless asked.
- `start_eval.sh` — the benchmark runner (the only thing run from this checkout)
- `lm_eval_tasks/gr_greek_personalities/` — the lm-eval task; `gr_greek_personalities.jsonl`
  (1650 4-choice items) is the committed eval set, `gr_greek_personalities.yaml` is the config
- `results/gr_personalities/<model>/` — one dir per model, each with `results_<ts>.json` (metrics)
  and `samples_*.jsonl` (~11MB, per-question log). Already committed, ~43MB total.
- `FAMOUS_..._VALIDATION_DATASET_PHAROS.json` (18MB) and `MULTIPLE_CHOICE_DATASET_PHAROS.json`
  (20MB) sit at repo root.

## Running the benchmark (the only locally runnable part)
`start_eval.sh` is NOT executable — use `bash start_eval.sh`, never `./start_eval.sh`.

Override via env: `MODEL_NAME`, `BASE_URL`, `TOKENIZER_NAME`, `TASK`, `INCLUDE_PATH`,
`OUTPUT_DIR`, `LIMIT`, `API_KEY`.
- Smoke test: `LIMIT=20 bash start_eval.sh`
- `BASE_URL` must be the full completions endpoint, e.g. `http://localhost:8016/v1/completions`
  (vLLM or any OpenAI-compatible server must already be running).
- The preflight `curl "$BASE_URL/models"` only detects connection-refused: `curl -sS` without
  `-f` exits 0 on a 404, so it will not catch a wrong path. Verify manually.
- The default `TOKENIZER_NAME` is a gated HF repo and needs the token hardcoded at
  `start_eval.sh:38`. That token is committed to a public remote — treat it as leaked.
  Export `HF_TOKEN=<your own>` to override.
- Output path is really `results/gr_personalities/<model>/results_<timestamp>.json`.
- lm_eval resolves `dataset_kwargs.data_files` relative to CWD, so it must run from the repo
  root. `start_eval.sh` does this itself; invoking `lm_eval` directly needs `cd` here first.

## Environment
No venv, no pinned versions, and the system Python 3.12.4 has none of the deps installed.
`requirements.txt` is incomplete: it omits `pydantic` (imported by `6clean_data.py` and
`7fix_rejected_data.py`) and `lm_eval` (the runner itself). Install all four:
`pip install -r requirements.txt pydantic lm_eval`.

## Dataset pipeline (1…11) — remote Linux box only
Not runnable from this checkout: every input is gitignored or absent, and paths are hardcoded
to `/home/vasters/...` (`10create_multiple_choice_dataset.py:14`). Edit before running anywhere.
Order and file handoffs:

| script | reads | writes |
|---|---|---|
| `1build_gr_people_list.py` | Wikidata SPARQL | `greek_people_metadata_full.json` |
| `2extract_gr_biographies.py` | `elwiki-latest-pages-articles.xml.bz2` (not committed) | `greek_biographies.jsonl` |
| `3llm_qa.py` | `greek_biographies.jsonl` | `llm_qa.json` |
| `4construct_questions_dataset.py` | `dataset/llm_qa.json` | `llm_questions_final.json` |
| `5generate_answers.py` | `llm_questions_final.json` | `QA_FINAL_DATASET_PHAROS.json` |
| `6clean_data.py` | `CLEANED_QA_FINAL_DATASET_PHAROS.json` * | `NEWclean_QA_FINAL_DATASET_PHAROS.json` |
| `check_data.py` | `NEWclean_…` | `ACCEPTED_EXAMPLES_…` |
| `7fix_rejected_data.py` | `REJECTED_EXAMPLES_…` * | `FIXED_REJECTED_…` |
| `8merge_accepted_fixed_data.py` | accepted + fixed | `GR_WIKIPEDIA_BIOS_QA_FINAL_DATASET_PHAROS.json` |
| `9create_val_bio_dataset.py` | `GR_WIKIPEDIA_…` | `FAMOUS_…_VALIDATION_DATASET_PHAROS.json` |
| `10create_multiple_choice_dataset.py` | `FAMOUS_…` | `MULTIPLE_CHOICE_DATASET_PHAROS.json` |
| `11prepare_lm_eval_dataset.py` | `MULTIPLE_CHOICE_…` | `lm_eval_tasks/…/gr_greek_personalities.jsonl` |

\* **The chain is broken here.** No uncommented line in this repo writes
`CLEANED_QA_FINAL_DATASET_PHAROS.json` or `REJECTED_EXAMPLES_QA_FINAL_DATASET_PHAROS.json`
(the producers are commented out in `check_data.py:34-37`). Those two files were produced
by some out-of-repo step; don't assume re-running `1…11` reproduces the dataset.

**Closed-book LLM-as-judge pipeline (post-11).** Scripts `12generate_answers.py` and
`13judge_answers.py` read `FAMOUS_GR_PERSONALITIES_QA_VALIDATION_DATASET_PHAROS.json` directly
(the validation set that produced the MC dataset). They are not part of the 1-11 chain.

Gotchas:
- Run from the repo root — every path is a bare relative filename.
- `3`, `5`, `10` talk to `http://localhost:8006/v1` with model `/model/gemma3`;
  `6`, `7` talk to `http://localhost:8016/v1` with `google/gemma-4-12B-it`.
  Two different servers, hardcoded in-file, not env-driven.
- `mkdir -p checkpoints checkpoints/repaired` before running 5/6/7/10 — they write
  checkpoints into those dirs without creating them, and will `FileNotFoundError`
  on the first checkpoint.
- Checkpoints are write-only. There is no resume logic; on crash, recover manually by
  taking the highest-numbered `checkpoints/*.json`.
- Each stage uses `mp.Pool` + `imap_unordered`, then reassembles by index
  (`new_dataset[i-1] = new`) to preserve input order. Record order is stable.
- `11prepare_lm_eval_dataset.py` is the only stage that filters: it drops any item whose
  `distractor_status != "ok"`, so `retry_ok` items are discarded too. Current
  1650-of-N output.
- Greek text throughout: every file read/write uses `encoding="utf-8"` and
  `ensure_ascii=False`. Never change these.

## Closed-book LLM-as-a-judge (12, 13)
A second benchmark over the **same 1650 questions** as the MC task, but the model under test
sees **only the question** — no biography, no gold answer. It measures parametric knowledge.
Unrelated to `5generate_answers.py`, which answers *with* the biography and is a dataset step,
not an eval. Don't reuse one's prompts for the other.

```
LIMIT=20 python3 12generate_answers.py    # smoke test; needs a server on BASE_URL
python3 13judge_answers.py
```
Env for both: `BASE_URL`, `LIMIT`, `NUM_WORKERS`, `API_KEY`, `OUTPUT_DIR`.
`12` also takes `MODEL_NAME`; `13` takes `JUDGE_MODEL_NAME`. Neither imports `lm_eval`.

Output lands in `results/judge_eval/<model>/` — `candidates_<ts>.jsonl` from `12`,
`judged_<ts>.jsonl` + `summary_<ts>.json` from `13`. Separate from lm-eval's
`results/gr_personalities/`, which has a different schema. Don't conflate the two.

Gotchas specific to these two:
- **59 of the 1650 questions never name their subject**, referring to the person with a bare
  `του`/`της`. Closed-book that pronoun has no referent, so `12` prepends the name for
  exactly those items (detected at runtime by 5-char Greek stem match on `question` vs
  `title`, not a hardcoded list) and tags the row `prompt_style: question_plus_title`.
  Filter on that field to analyse the 1591 clean ones.
- `13` joins the biography back in from the source JSON by `doc_id - 1`; candidates are
  written without it. The judge needs the bio as the authority on what is true.
- The judge is English-prompted but must return **reasons in Greek**. Two prompt rules are
  load-bearing — a prompt edit must not silently drop them: a short candidate is not a wrong
  candidate (gold answers are long, models answer briefly), and the judge scores the
  underlying fact, not the reference phrasing (gold answers carry Wikipedia editor wording).
- `summary.json` reports `abstain_rate` separately from `acc`. An honest "δεν γνωρίζω" is
  `ABSTAIN`, never a `REJECT` — folding refusals into failures misrepresents the model.
  `acc` is comparable to the MC `acc` on the same 1650 items; `abstain_rate` has no MC
  counterpart.
- Judge errors (including `ValidationError` on the pydantic schema) are recorded per row as
  `judge_error` and excluded from the summary denominators, never silently scored.
- Both scripts `mkdir` their own output and `checkpoints/` dirs — unlike 5/6/7/10, which
  crash if theirs are missing.

## Committing
`.gitignore` guards `dataset/*.json`, but that directory does not exist — the pipeline's
multi-hundred-MB intermediates land at the repo root and are unguarded. Check `git status`
for stray `*.json` before committing, and never `git add .` blindly.
