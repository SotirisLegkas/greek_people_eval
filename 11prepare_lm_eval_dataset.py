import json
import os

########################################################################################################################################################
########################################################################################################################################################
INPUT_FILE = "MULTIPLE_CHOICE_DATASET_PHAROS.json"
OUTPUT_FILE = "lm_eval_tasks/gr_greek_personalities/gr_greek_personalities.jsonl"
########################################################################################################################################################
########################################################################################################################################################
def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
########################################################################################################################################################
########################################################################################################################################################
data = load_data(INPUT_FILE)

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

skipped = 0
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for i, example in enumerate(data):
        if example.get("distractor_status") != "ok":
            skipped += 1
            continue

        choices = example.get("choices", [])
        correct_index = example.get("correct_index")

        if not isinstance(choices, list) or len(choices) != 4:
            skipped += 1
            continue

        if not isinstance(correct_index, int) or not (0 <= correct_index < len(choices)):
            skipped += 1
            continue

        if choices[correct_index] != example.get("answer"):
            skipped += 1
            continue

        record = {
            "title": example["title"],
            "question": example["question"],
            "choices": choices,
            "correct_index": correct_index,
            "answer": example["answer"],
        }

        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Total records: {len(data)}")
print(f"Records written: {len(data) - skipped}")
print(f"Records skipped: {skipped}")
print(f"Output: {OUTPUT_FILE}")