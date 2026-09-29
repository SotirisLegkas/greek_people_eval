########################################################################################################################################################
# Closed-book candidate generation for LLM-as-a-judge evaluation.
#
# The model under test sees ONLY the question. The biography (full_text) and the gold
# answer are never sent. This measures parametric knowledge, not reading comprehension.
#
# Compare with 5generate_answers.py, which answers WITH the biography in context. That
# script is a dataset-construction step; this one is an evaluation step. Do not confuse
# the two or reuse one's prompts for the other.
########################################################################################################################################################

import json
import os
import re
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import tqdm
from openai import OpenAI
import multiprocessing as mp

########################################################################################################################################################
# Configuration (override via env)
########################################################################################################################################################

INPUT_FILE = os.environ.get(
    "INPUT_FILE", "FAMOUS_GR_PERSONALITIES_QA_VALIDATION_DATASET_PHAROS.json"
)
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8016/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "google/gemma-4-31B-it")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", f"results/judge_eval/{MODEL_NAME.replace('/', '__')}")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "")
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
LIMIT = os.environ.get("LIMIT", "")
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
API_KEY = os.environ.get("API_KEY", "EMPTY")
CHECKPOINT_EVERY = int(os.environ.get("CHECKPOINT_EVERY", "100"))
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "300"))

########################################################################################################################################################
# Prompts
########################################################################################################################################################

SYSTEM_PROMPT = """
You answer questions about people using only your own knowledge.

You will receive one question in Greek. Answer it directly in Greek.

Rules:
1. Answer from your own knowledge. You are not given any source text.
2. If you do not know the answer, say so plainly in Greek. Do not guess and do not invent facts.
3. If you only know part of the answer, give the part you know and say what you are unsure about.
4. Answer the question that was actually asked. Do not rewrite it as a different question.
5. Write a direct, complete, natural answer in Greek.
6. Do not mention that you were given a question without context, and do not ask for clarification.
7. Do not use Markdown, bullet points, or code fences.
8. Return only the answer text.

Be accurate. A correct short answer is better than a long one containing invented details.
""".strip()


USER_PROMPT = """
{question}
""".strip()

# Used only for the questions that refer to the person with a bare pronoun (e.g. "του",
# "της") and never name them, which leaves the referent unresolvable.
USER_PROMPT_WITH_TITLE = """
{question}

(Το ερώτημα αφορά τον/την: {title})
""".strip()

########################################################################################################################################################
# Pronoun-reference detection
#
# 59 of the 1650 questions never name their subject, so the pronoun has no referent
# and a closed-book model cannot resolve it. For those we prepend the person's name.
# This is a runtime heuristic, not a hardcoded list, so it survives dataset changes.
########################################################################################################################################################

GREEK_WORD = re.compile(r"[Ͱ-Ͽἀ-῿]{4,}")
STEM_LEN = 5


def strip_accents(text):
    return "".join(
        char for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    ).lower()


def stems(text):
    return {word[:STEM_LEN] for word in GREEK_WORD.findall(strip_accents(text))}


def needs_title(question, title):
    """True when the question shares no Greek word stem with the person's name."""
    return not (stems(question) & stems(title))


def build_user_prompt(example):
    if needs_title(example["question"], example["title"]):
        return (
            USER_PROMPT_WITH_TITLE.format(
                question=example["question"], title=example["title"]
            ),
            "question_plus_title",
        )
    return USER_PROMPT.format(question=example["question"]), "question_only"

########################################################################################################################################################
# Model call
########################################################################################################################################################

client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=REQUEST_TIMEOUT)


def ask_model(user_prompt):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )
    return response.model_dump()["choices"][0]["message"]["content"]


def process_example(args):
    i, example = args

    user_prompt, prompt_style = build_user_prompt(example)

    candidate = ""
    raw_response = ""
    error = None

    for attempt in range(MAX_RETRIES):
        try:
            raw_response = ask_model(user_prompt)
            candidate = (raw_response or "").strip()
            if candidate:
                error = None
                break
            error = "empty_response"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        if attempt < MAX_RETRIES - 1:
            time.sleep(2 ** attempt)

    if not candidate:
        print(f"Error processing example {i}: {error}")

    return i, {
        "doc_id": i,
        "title": example["title"],
        "question": example["question"],
        "prompt_style": prompt_style,
        "candidate": candidate,
        "raw_response": raw_response,
        "error": error,
    }

########################################################################################################################################################


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":

    data = load_data(INPUT_FILE)

    if LIMIT:
        data = data[: int(LIMIT)]

    style_counts = {"question_only": 0, "question_plus_title": 0}
    for example in data:
        style_counts[build_user_prompt(example)[1]] += 1
    print(f"Loaded {len(data)} records from {INPUT_FILE}")
    print(f"  question_only        : {style_counts['question_only']}")
    print(f"  question_plus_title  : {style_counts['question_plus_title']}")
    print(f"Model: {MODEL_NAME} at {BASE_URL}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/checkpoints", exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    output_path = OUTPUT_PATH or os.path.join(OUTPUT_DIR, f"candidates_{timestamp}.jsonl")

    new_dataset = [None] * len(data)
    failures = 0

    with mp.Pool(processes=NUM_WORKERS) as pool, open(output_path, "w", encoding="utf-8") as f:

        for i, new in tqdm.tqdm(
            pool.imap_unordered(process_example, enumerate(data, start=1)),
            total=len(data),
            desc="Generating answers",
        ):
            new_dataset[i - 1] = new
            if new["error"]:
                failures += 1
            f.write(json.dumps(new, ensure_ascii=False) + "\n")
            f.flush()

            if i % CHECKPOINT_EVERY == 0:
                with open(
                    f"{OUTPUT_DIR}/checkpoints/candidates_checkpoint_{i}.jsonl",
                    "w",
                    encoding="utf-8",
                ) as cf:
                    for record in new_dataset:
                        if record is not None:
                            cf.write(json.dumps(record, ensure_ascii=False) + "\n")

    answered = sum(1 for record in new_dataset if record["candidate"])
    print(f"Records written: {len(new_dataset)}")
    print(f"Non-empty answers: {answered}")
    print(f"Failed or empty: {failures}")
    print(f"Output: {output_path}")
