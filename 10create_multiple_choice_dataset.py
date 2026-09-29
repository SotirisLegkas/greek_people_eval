from openai import OpenAI
import json
import json_repair
import tqdm
import random
import multiprocessing as mp

########################################################################################################################################################
########################################################################################################################################################
def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
data = load_data("/home/vasters/pharos/people_dataset_gr_wiki/dataset/FAMOUS_GR_PERSONALITIES_QA_VALIDATION_DATASET_PHAROS.json")

########################################################################################################################################################
########################################################################################################################################################
SYSTEM_PROMPT = """
You are creating a Greek multiple-choice question-answering dataset from biographies.

You will receive:
1. The name of a person.
2. A biography.
3. One question about the person.
4. The correct answer to the question.

Generate exactly 3 wrong answers (distractors) for the question.

Rules:
1. Write every distractor in Greek.
2. Every distractor must be a complete, natural, and plausible sentence that answers the question but is factually WRONG.
3. Use only the biography to generate the distractors.
4. Distractors must be clearly distinct from the correct answer.
5. Distractors must be distinct from each other.
6. Do not use external knowledge.
7. Do not invent, assume, or infer unsupported information.
8. The distractors must be wrong but not absurd, so a reader who knows the biography can tell them apart from the correct answer.
9. Return exactly 3 distractors.
10. Return only valid JSON.
11. Do not use Markdown or JSON code fences.

Return exactly this structure:
{
  "distractors": [
    "First wrong answer in Greek.",
    "Second wrong answer in Greek.",
    "Third wrong answer in Greek."
  ]
}
""".strip()


USER_PROMPT = """
Person's name:
{title}

Biography:
{biography}

Question:
{question}

Correct answer:
{answer}

Generate exactly 3 plausible but wrong answers (distractors) for the question.
""".strip()


CORRECTION_HINT = """

The previous distractors were invalid. They MUST satisfy all of the following:
- be exactly 3 non-empty strings,
- be written in Greek,
- be different from the correct answer,
- be different from each other.
Return only valid JSON in the exact structure:
{{
  "distractors": [
    "First wrong answer in Greek.",
    "Second wrong answer in Greek.",
    "Third wrong answer in Greek."
  ]
}}
""".strip()
########################################################################################################################################################
########################################################################################################################################################
client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8006/v1")

def ask_model(system, user):
    response = client.chat.completions.create(
        model="/model/gemma3", #/model/gemma3 #google/gemma-4-26B-A4B-it
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
    )
    
    content = response.model_dump()['choices'][0]['message']['content']
    return content
########################################################################################################################################################
########################################################################################################################################################
def normalize(s):
    return " ".join(s.strip().lower().split())

def distractors_valid(distractors, correct_answer):
    if not isinstance(distractors, list) or len(distractors) != 3:
        return False

    if not all(isinstance(d, str) for d in distractors):
        return False

    distractors = [d.strip() for d in distractors]

    if any(not d for d in distractors):
        return False

    correct_norm = normalize(correct_answer)
    norms = [normalize(d) for d in distractors]

    if any(n == correct_norm for n in norms):
        return False

    if len(set(norms)) != len(norms):
        return False

    return True

def generate_distractors(example, correction=False):
    user_prompt = USER_PROMPT.format(
        title=example['title'],
        biography=example['full_text'],
        question=example['question'],
        answer=example['answer']
    )

    if correction:
        user_prompt += CORRECTION_HINT

    content = ask_model(SYSTEM_PROMPT, user_prompt)

    json_output = json_repair.repair_json(content, return_objects=True)

    if isinstance(json_output, dict):
        return json_output.get('distractors', [])
    elif isinstance(json_output, list):
        return json_output
    else:
        return []

def process_example(args):
    i, example = args

    distractors = []
    status = "failed"

    try:
        distractors = generate_distractors(example)
        if distractors_valid(distractors, example['answer']):
            status = "ok"
        else:
            distractors = generate_distractors(example, correction=True)
            if distractors_valid(distractors, example['answer']):
                status = "retry_ok"

    except Exception as e:
        print(f"Error processing example {i}: {e}")

    if status == "failed":
        distractors = []

    choices = distractors + [example['answer']]
    random.shuffle(choices)
    correct_index = choices.index(example['answer'])

    new = {
        "title": example["title"],
        "full_text": example["full_text"],
        "question": example['question'],
        "answer": example['answer'],
        "choices": choices,
        "correct_index": correct_index,
        "distractor_status": status
    }

    return i, new
########################################################################################################################################################
########################################################################################################################################################
if __name__ == "__main__":

    new_dataset = [None] * len(data)

    with mp.Pool(processes=4) as pool:

        for i, new in tqdm.tqdm(pool.imap_unordered(process_example,enumerate(data, start=1)), total=len(data),desc="Processing examples"):
            new_dataset[i-1] = new
            # checkpoint every 100 completed examples
            completed = sum(x is not None for x in new_dataset)
            if completed % 1000 == 0:
                with open(f"checkpoints/mc_dataset_checkpoint_{completed}.json","w",encoding="utf-8") as f:
                    json.dump([x for x in new_dataset if x is not None],f,ensure_ascii=False,indent=2)


    with open("MULTIPLE_CHOICE_DATASET_PHAROS.json", "w", encoding="utf-8") as f:
            json.dump(new_dataset, f, ensure_ascii=False, indent=2)