from openai import OpenAI
import json
import json_repair
import tqdm
import multiprocessing as mp


#####################################################################
# Configuration
#####################################################################

INPUT_FILE = "llm_qa.json"

ALL_OUTPUT_FILE = "qa_dataset_all.json"
ANSWERED_OUTPUT_FILE = "qa_dataset.json"
UNANSWERED_OUTPUT_FILE = "unanswered_questions.json"
CHECKPOINT_FILE = "qa_dataset_checkpoint.json"

BASE_URL = "http://localhost:8006/v1"
MODEL_NAME = "/model/gemma3"

NUM_WORKERS = 4
MAX_ATTEMPTS = 3
CHECKPOINT_EVERY = 100


#####################################################################
# Load the question dataset
#####################################################################

def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


#####################################################################
# Prompts
#####################################################################

SYSTEM_PROMPT = """
You are creating a question-answering dataset from biographies.

You will receive:
1. A person's name.
2. The person's biography.
3. A list of questions about the person.

Answer every question using only information explicitly stated in the biography.

Rules:
1. Write every answer in Greek.
2. Every answer must be a complete and natural sentence.
3. Every answer must directly answer the corresponding question.
4. Use only information explicitly stated in the biography.
5. Do not use external knowledge.
6. Do not invent, assume, or infer unsupported information.
7. Do not add facts that are not supported by the biography.
8. For every answer, also return an evidence segment.
9. The evidence must be copied exactly from the biography.
10. The evidence must directly support the complete answer.
11. The evidence must be one continuous text segment.
12. Preserve names, dates, spelling, accents, and punctuation in the evidence.
13. Preserve the question ID exactly.
14. Return the same number of answer objects as questions.
15. If the biography does not contain enough information to answer a question, return an empty answer and an empty evidence string.
16. Return only valid JSON.
17. Do not use Markdown or JSON code fences.

Return exactly this structure:

{
  "answers": [
    {
      "id": 0,
      "answer": "A complete answer written in Greek.",
      "evidence": "Exact supporting text copied from the biography."
    }
  ]
}
""".strip()


USER_PROMPT = """
Person's name:
{title}

Biography:
{biography}

Questions:
{questions}

Answer every question using only the biography.

Return one answer object for every question.
""".strip()


#####################################################################
# Create one OpenAI client per worker process
#####################################################################

client = OpenAI(
        api_key="EMPTY",
        base_url=BASE_URL
    )

def ask_model(system_prompt, user_prompt):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
    )

    content = response.model_dump()['choices'][0]['message']['content']
    return content


#####################################################################
# Get questions from either supported structure
#####################################################################

def get_questions(example):
    questions_field = example.get("questions", [])

    # Structure from your current dataset:
    # "questions": {"questions": [...]}
    if isinstance(questions_field, dict):
        questions = questions_field.get("questions", [])

    # Also support:
    # "questions": [...]
    elif isinstance(questions_field, list):
        questions = questions_field

    else:
        questions = []

    return questions


#####################################################################
# Process one biography
#####################################################################

def process_example(args):
    person_index, example = args

    title = example["title"]
    biography = example["full_text"]
    questions = get_questions(example)

    if not questions:
        return person_index, []

    # Give each question a stable ID
    indexed_questions = []

    for question_id, question_data in enumerate(questions):
        indexed_questions.append({
            "id": question_id,
            "question": question_data["question"],
            "question_type": question_data.get(
                "question_type",
                "other"
            )
        })

    # Questions that still require a valid answer
    pending = {
        question["id"]: question
        for question in indexed_questions
    }

    valid_answers = {}

    failure_reasons = {
        question["id"]: "The model did not return an answer."
        for question in indexed_questions
    }

    # First request plus retries
    for attempt in range(1, MAX_ATTEMPTS + 1):

        # Stop when every question has a valid answer
        if not pending:
            break

        try:
            content = ask_model(
                SYSTEM_PROMPT,
                USER_PROMPT.format(
                    title=title,
                    biography=biography,
                    questions=json.dumps(
                        list(pending.values()),
                        ensure_ascii=False,
                        indent=2
                    )
                )
            )

            json_output = json_repair.repair_json(
                content,
                return_objects=True
            )

            # Expected output:
            # {"answers": [...]}
            if isinstance(json_output, dict):
                returned_answers = json_output.get(
                    "answers",
                    []
                )

            # Also tolerate a direct list
            elif isinstance(json_output, list):
                returned_answers = json_output

            else:
                returned_answers = []

            for item in returned_answers:

                if not isinstance(item, dict):
                    continue

                try:
                    question_id = int(item.get("id"))
                except (TypeError, ValueError):
                    continue

                # Ignore unknown IDs or questions already answered
                if question_id not in pending:
                    continue

                answer = item.get("answer", "")
                evidence = item.get("evidence", "")

                if not isinstance(answer, str):
                    failure_reasons[question_id] = (
                        "The returned answer was not text."
                    )
                    continue

                if not isinstance(evidence, str):
                    failure_reasons[question_id] = (
                        "The returned evidence was not text."
                    )
                    continue

                answer = answer.strip()
                evidence = evidence.strip()

                if not answer:
                    failure_reasons[question_id] = (
                        "The model returned an empty answer."
                    )
                    continue

                if not evidence:
                    failure_reasons[question_id] = (
                        "The model returned empty evidence."
                    )
                    continue

                # Validate that the evidence occurs exactly
                # in the person's biography
                answer_start = biography.find(evidence)

                if answer_start == -1:
                    failure_reasons[question_id] = (
                        "The evidence was not copied exactly "
                        "from the biography."
                    )
                    continue

                # The answer passed validation
                valid_answers[question_id] = {
                    "answer": answer,
                    "evidence": evidence,
                    "answer_start": answer_start,
                    "attempt": attempt
                }

                del pending[question_id]

        except Exception as e:
            error_message = (
                f"Request error on attempt {attempt}: {e}"
            )

            for question_id in pending:
                failure_reasons[question_id] = error_message

    #################################################################
    # Create exactly one output item for every input question
    #################################################################

    qa_examples = []

    for question_data in indexed_questions:
        question_id = question_data["id"]

        if question_id in valid_answers:
            result = valid_answers[question_id]

            qa_examples.append({
                "title": title,
                "context": biography,
                "question_id": question_id,
                "question": question_data["question"],
                "question_type": question_data["question_type"],
                "answer": result["answer"],
                "evidence": result["evidence"],
                "answer_start": result["answer_start"],
                "attempt": result["attempt"],
                "status": "answered"
            })

        else:
            qa_examples.append({
                "title": title,
                "context": biography,
                "question_id": question_id,
                "question": question_data["question"],
                "question_type": question_data["question_type"],
                "answer": "",
                "evidence": "",
                "answer_start": -1,
                "attempt": MAX_ATTEMPTS,
                "status": "unanswered",
                "failure_reason": failure_reasons.get(
                    question_id,
                    "Unknown validation error."
                )
            })

    # Verify that no question was lost
    assert len(qa_examples) == len(indexed_questions)

    return person_index, qa_examples


#####################################################################
# Save JSON helper
#####################################################################

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )


#####################################################################
# Main
#####################################################################

if __name__ == "__main__":

    mp.freeze_support()

    data = load_data(INPUT_FILE)

    # Preserve the original biography order despite imap_unordered
    results_by_person = [None] * len(data)

    completed_biographies = 0

    with mp.Pool(processes=NUM_WORKERS) as pool:

        results = pool.imap_unordered(
            process_example,
            enumerate(data),
            chunksize=1
        )

        for person_index, person_qas in tqdm.tqdm(
            results,
            total=len(data),
            desc="Generating answers"
        ):
            results_by_person[person_index] = person_qas
            completed_biographies += 1

            #########################################################
            # Save a checkpoint every 100 processed biographies
            #########################################################

            if completed_biographies % CHECKPOINT_EVERY == 0:

                checkpoint_data = [
                    qa
                    for person_results in results_by_person
                    if person_results is not None
                    for qa in person_results
                ]

                save_json(checkpoint_data,CHECKPOINT_FILE)

                checkpoint_answered = sum(item["status"] == "answered"for item in checkpoint_data)

                checkpoint_unanswered = (len(checkpoint_data) - checkpoint_answered)

                tqdm.tqdm.write(
                    f"Checkpoint after "
                    f"{completed_biographies} biographies: "
                    f"{checkpoint_answered} answered, "
                    f"{checkpoint_unanswered} unanswered"
                )

    #################################################################
    # Flatten results while preserving biography order
    #################################################################

    qa_dataset_all = [
        qa
        for person_results in results_by_person
        if person_results is not None
        for qa in person_results
    ]

    answered_questions = [
        item
        for item in qa_dataset_all
        if item["status"] == "answered"
    ]

    unanswered_questions = [
        item
        for item in qa_dataset_all
        if item["status"] == "unanswered"
    ]

    biographies_without_questions = sum(
        person_results == []
        for person_results in results_by_person
    )

    #################################################################
    # Save final files
    #################################################################

    # Contains every question, including unanswered questions
    save_json(
        qa_dataset_all,
        ALL_OUTPUT_FILE
    )

    # Final usable QA dataset containing only validated answers
    save_json(
        answered_questions,
        ANSWERED_OUTPUT_FILE
    )

    # Questions that still failed after all retries
    save_json(
        unanswered_questions,
        UNANSWERED_OUTPUT_FILE
    )

    #################################################################
    # Print coverage report
    #################################################################

    total_questions = len(qa_dataset_all)
    total_answered = len(answered_questions)
    total_unanswered = len(unanswered_questions)

    print("\nFinished")
    print(f"Biographies: {len(data)}")
    print(
        f"Biographies without questions: "
        f"{biographies_without_questions}"
    )
    print(f"Total questions: {total_questions}")
    print(f"Answered questions: {total_answered}")
    print(f"Unanswered questions: {total_unanswered}")

    if total_questions > 0:
        coverage = (
            100.0 *
            total_answered /
            total_questions
        )

        print(f"Answer coverage: {coverage:.2f}%")

    print(f"\nAll results: {ALL_OUTPUT_FILE}")
    print(f"Validated QA dataset: {ANSWERED_OUTPUT_FILE}")
    print(f"Unanswered questions: {UNANSWERED_OUTPUT_FILE}")