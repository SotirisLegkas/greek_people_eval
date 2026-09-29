########################################################################################################################################################
# LLM-as-a-judge scoring for closed-book candidate answers produced by 12generate_answers.py.
#
# The judge is given the biography (full_text) as the authority on what is true, plus the
# gold answer as the reference for the fact, plus the candidate the model produced WITHOUT
# any of that context. The judge is English-prompted (like 6clean_data.py) but must return
# its reasons in Greek so that failures are inspectable by a human reader.
#
# Two rules below are load-bearing and easy to lose in prompt edits:
#   1. A short candidate is NOT a wrong candidate. Gold answers are long; closed-book
#      models answer briefly. Penalising brevity measures verbosity, not knowledge.
#   2. Score the underlying FACT, not the reference phrasing. Gold answers carry the
#      Wikipedia editor's wording, which no closed-book model will reproduce verbatim.
########################################################################################################################################################

import json
import os
import time
from collections import Counter
from datetime import datetime
from enum import Enum
from pathlib import Path

import tqdm
from openai import OpenAI
import json_repair
import multiprocessing as mp
from pydantic import BaseModel, ConfigDict, Field, ValidationError

########################################################################################################################################################
# Configuration (override via env)
########################################################################################################################################################

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8016/v1")
JUDGE_MODEL_NAME = os.environ.get("JUDGE_MODEL_NAME", "google/gemma-4-31B-it")
INPUT_DIR = os.environ.get("INPUT_DIR", "results/judge_eval")
INPUT_PATH = os.environ.get("INPUT_PATH", "")
# Defaults to the directory holding the resolved input, so judged output lands beside the
# candidates instead of the shared results/judge_eval/ root.
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "")
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
LIMIT = os.environ.get("LIMIT", "")
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
API_KEY = os.environ.get("API_KEY", "EMPTY")
CHECKPOINT_EVERY = int(os.environ.get("CHECKPOINT_EVERY", "100"))
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "300"))
# Recorded in the summary only, since candidates carry no model name. 12generate_answers.py
# prints it and names the output dir, so set this to keep the summary self-describing.
GENERATOR_MODEL = os.environ.get("GENERATOR_MODEL", "")

########################################################################################################################################################
# Prompts
########################################################################################################################################################

SYSTEM_PROMPT = '''You are an expert evaluator of Greek biographical question-answering.

The CANDIDATE was produced by another model that answered the question using ONLY its own
knowledge. It was given no biography, no source text, and no reference answer. Your job is to
judge how good that answer is.

You will receive:
1. TITLE: the name of the person.
2. FULL_TEXT: a Greek Wikipedia article about that person.
3. QUESTION: the question in Greek.
4. GOLD ANSWER: the reference answer, written from FULL_TEXT.
5. CANDIDATE: the answer produced closed-book by the model being evaluated.

Treat FULL_TEXT as the authority on what is true about this person.
Treat GOLD ANSWER as the reference for the specific fact the question asks about.

### IMPORTANT scoring rules

A) Do NOT penalise the CANDIDATE for being shorter than GOLD ANSWER.
   Gold answers are long because they were generated from a full biography. A closed-book
   model that answers concisely has not necessarily failed. Only FAIL the candidate if the
   information it does give is wrong, missing something the question explicitly asked for,
   or contradicts the reference. Concision alone is never a failure.

B) Score the underlying FACT, not the reference phrasing.
   Gold answers are copied from a Wikipedia article and often contain that article's
   editorial wording (for example a school mark described as «μέτριος», or the phrasing of
   a summary sentence). A candidate that states the same fact in different words is correct.
   Do not require verbatim agreement.

C) Do NOT penalise the candidate for facts that FULL_TEXT itself does not mention but that
   are true and consistent. Judge on consistency with FULL_TEXT, not on exhaustiveness of
   agreement with it.

D) An honest refusal is not a wrong answer. If the candidate says it does not know, mark
   answer_correctness as ABSTAIN. ABSTAIN is a legitimate outcome for a closed-book
   evaluation and must not be recorded as a failure.

### Criterion 1: answer_correctness
Does the CANDIDATE correctly answer the question asked?

Return:
- PASS  : factually correct and it answers what was asked.
- ABSTAIN: the candidate honestly says it does not know, or explicitly declines to answer.
- FAIL  : the candidate gives an answer, but it is factually wrong, contradicts the
          reference, or fails to address the specific thing the question asked for.

If not PASS, explain in Greek which part is wrong or missing.

### Criterion 2: answer_grounding
Is the CANDIDATE consistent with FULL_TEXT, or does it fabricate facts?

Return:
- PASS    : everything the candidate asserts about the person is consistent with FULL_TEXT,
            or FULL_TEXT is silent on that point and the candidate is not contradicted.
- FAIL    : the candidate asserts something about the person that FULL_TEXT contradicts, or
            presents an invented fact as established.

Correct-sounding but unsupported specifics are the main thing this criterion must catch.

If FAIL, explain in Greek exactly which claim is unsupported or contradicted.

### Criterion 3: language_quality
Is the CANDIDATE well-written Greek?

Return:
- PASS         : grammatical, natural, fluent Greek. Minor stylistic preferences are fine.
- MINOR_ISSUES : small problems that do not obscure the meaning.
- FAIL         : serious problems that make the answer hard or impossible to understand.

If not PASS, say in Greek what the problems are.

### Overall
Return:
- ACCEPT : answer_correctness is PASS or ABSTAIN, AND answer_grounding is PASS.
           MINOR_ISSUES on language does not prevent acceptance.
- REJECT : answer_correctness is FAIL, OR answer_grounding is FAIL.

An ABSTAIN is never a REJECT on its own. A refusal is not a hallucination.

Return an evaluation that follows the supplied JSON schema.
Write every reason field in Greek, however brief.
Use ONLY FULL_TEXT as evidence of what is true about this person.
Do NOT reward a candidate for agreeing with a well-known person; only FULL_TEXT and
GOLD ANSWER are relevant here.
'''


USER_PROMPT = '''TITLE:
{title}

FULL_TEXT:
{full_text}

QUESTION:
{question}

GOLD ANSWER:
{answer}

CANDIDATE:
{candidate}'''

########################################################################################################################################################
# Schema
########################################################################################################################################################


class CorrectnessLabel(str, Enum):
    PASS = "PASS"
    ABSTAIN = "ABSTAIN"
    FAIL = "FAIL"


class GroundingLabel(str, Enum):
    PASS = "PASS"
    MINOR_ISSUES = "MINOR_ISSUES"
    FAIL = "FAIL"


class LanguageLabel(str, Enum):
    PASS = "PASS"
    MINOR_ISSUES = "MINOR_ISSUES"
    FAIL = "FAIL"


class OverallLabel(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"


class CorrectnessEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: CorrectnessLabel
    reason: str = Field(
        description="Empty when PASS. In Greek otherwise. ABSTAIN when the candidate declines to answer."
    )


class GroundingEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: GroundingLabel
    reason: str = Field(
        description="Empty when PASS. In Greek otherwise, naming the unsupported or contradicted claim."
    )


class LanguageEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: LanguageLabel
    reason: str = Field(
        description="Empty when PASS. In Greek otherwise, naming the language problems."
    )


class AnswerEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer_correctness: CorrectnessEvaluation
    answer_grounding: GroundingEvaluation
    language_quality: LanguageEvaluation
    overall: OverallLabel


RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer_evaluation",
        "strict": True,
        "schema": AnswerEvaluation.model_json_schema(),
    },
}

########################################################################################################################################################
# Model call
########################################################################################################################################################

client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=REQUEST_TIMEOUT)


def ask_model(system, user):
    response = client.chat.completions.create(
        model=JUDGE_MODEL_NAME,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format=RESPONSE_FORMAT,
        temperature=0.0,
    )
    return response.model_dump()["choices"][0]["message"]["content"]


def evaluate(example):
    """Return the parsed evaluation dict, or raise."""
    content = ask_model(
        SYSTEM_PROMPT,
        USER_PROMPT.format(
            title=example["title"],
            full_text=example["full_text"],
            question=example["question"],
            answer=example["answer"],
            candidate=example["candidate"],
        ),
    )
    json_output = json_repair.repair_json(content, return_objects=True)
    return AnswerEvaluation.model_validate(json_output).model_dump(mode="json")


def process_example(args):
    i, example = args

    error = None
    evaluation = None

    if not example["candidate"]:
        # Generation failed. Judge nothing; it is counted as a generation failure upstream.
        error = "empty_candidate"
    else:
        for attempt in range(MAX_RETRIES):
            try:
                evaluation = evaluate(example)
                error = None
                break
            except ValidationError as exc:
                error = f"ValidationError: {exc}"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)

    if evaluation is None:
        print(f"Error judging example {i}: {error}")

    return i, {
        "doc_id": example["doc_id"],
        "title": example["title"],
        "question": example["question"],
        "prompt_style": example.get("prompt_style"),
        "candidate": example["candidate"],
        "gold_answer": example["answer"],
        "evaluation": evaluation,
        "judge_error": error,
    }

########################################################################################################################################################
# Summary
########################################################################################################################################################


def build_summary(records, input_path):
    judged = [r for r in records if r["evaluation"] is not None]
    n_items = len(records)

    def rate(count):
        return count / len(judged) if judged else 0.0

    correctness = Counter(r["evaluation"]["answer_correctness"]["label"] for r in judged)
    grounding = Counter(r["evaluation"]["answer_grounding"]["label"] for r in judged)
    language = Counter(r["evaluation"]["language_quality"]["label"] for r in judged)
    overall = Counter(r["evaluation"]["overall"] for r in judged)

    prompt_styles = Counter(r.get("prompt_style") for r in records if r.get("prompt_style"))

    return {
        "n_items": n_items,
        "n_judged": len(judged),
        "n_judge_errors": n_items - len(judged),
        "acc": rate(overall["ACCEPT"]),
        "acc_stderr": (rate(overall["ACCEPT"]) * (1 - rate(overall["ACCEPT"])) / len(judged)) ** 0.5
        if judged
        else 0.0,
        "abstain_rate": rate(correctness["ABSTAIN"]),
        "criteria": {
            "answer_correctness": {
                "PASS": correctness["PASS"],
                "ABSTAIN": correctness["ABSTAIN"],
                "FAIL": correctness["FAIL"],
            },
            "answer_grounding": {
                "PASS": grounding["PASS"],
                "MINOR_ISSUES": grounding["MINOR_ISSUES"],
                "FAIL": grounding["FAIL"],
            },
            "language_quality": {
                "PASS": language["PASS"],
                "MINOR_ISSUES": language["MINOR_ISSUES"],
                "FAIL": language["FAIL"],
            },
        },
        "overall": {"ACCEPT": overall["ACCEPT"], "REJECT": overall["REJECT"]},
        "prompt_styles": dict(prompt_styles),
        "config": {
            "generator_model": GENERATOR_MODEL or None,
            "judge_model": JUDGE_MODEL_NAME,
            "base_url": BASE_URL,
            "input_path": input_path,
            "limit": LIMIT or None,
            "num_workers": NUM_WORKERS,
        },
    }

########################################################################################################################################################


def load_records(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def resolve_input_path():
    if INPUT_PATH:
        return INPUT_PATH
    candidates = sorted(
        str(p)
        for p in Path(INPUT_DIR).rglob("candidates_*.jsonl")
        if "checkpoints" not in p.parts
    )
    if not candidates:
        raise SystemExit(
            f"No candidates_*.jsonl found under {INPUT_DIR}. Run 12generate_answers.py first, "
            f"or set INPUT_PATH explicitly."
        )
    return candidates[-1]


if __name__ == "__main__":

    input_path = resolve_input_path()
    if not OUTPUT_DIR:
        OUTPUT_DIR = str(Path(input_path).parent)
    print(f"Judging: {input_path}")
    print(f"Output to: {OUTPUT_DIR}")
    print(f"Judge model: {JUDGE_MODEL_NAME} at {BASE_URL}")

    data = load_records(input_path)

    # Candidates are written without the biography, so it is joined in here from the
    # source dataset. The judges need it as the authority on what is true.
    source_path = os.environ.get("SOURCE_FILE", "FAMOUS_GR_PERSONALITIES_QA_VALIDATION_DATASET_PHAROS.json")
    with open(source_path, "r", encoding="utf-8") as f:
        source = json.load(f)
    for record in data:
        original = source[record["doc_id"] - 1]
        record["full_text"] = original["full_text"]
        record["answer"] = original["answer"]

    if LIMIT:
        data = data[: int(LIMIT)]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/checkpoints", exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    judged_path = os.path.join(OUTPUT_DIR, f"judged_{timestamp}.jsonl")
    summary_path = os.path.join(OUTPUT_DIR, f"summary_{timestamp}.json")

    new_dataset = [None] * len(data)

    with mp.Pool(processes=NUM_WORKERS) as pool, open(judged_path, "w", encoding="utf-8") as f:

        for i, new in tqdm.tqdm(
            pool.imap_unordered(process_example, enumerate(data, start=1)),
            total=len(data),
            desc="Judging answers",
        ):
            new_dataset[i - 1] = new
            f.write(json.dumps(new, ensure_ascii=False) + "\n")
            f.flush()

            if i % CHECKPOINT_EVERY == 0:
                with open(
                    f"{OUTPUT_DIR}/checkpoints/judged_checkpoint_{i}.jsonl",
                    "w",
                    encoding="utf-8",
                ) as cf:
                    for record in new_dataset:
                        if record is not None:
                            cf.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = build_summary(new_dataset, input_path)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Records written: {len(new_dataset)}")
    print(f"Judged: {summary['n_judged']}  judge errors: {summary['n_judge_errors']}")
    print(f"acc: {summary['acc']:.4f}  abstain_rate: {summary['abstain_rate']:.4f}")
    print(f"correctness: {summary['criteria']['answer_correctness']}")
    print(f"grounding:   {summary['criteria']['answer_grounding']}")
    print(f"Output: {judged_path}")
    print(f"Summary: {summary_path}")
