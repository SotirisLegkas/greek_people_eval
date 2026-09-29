import json

########################################################################################################################################################
########################################################################################################################################################
def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
accepted_data = load_data("ACCEPTED_EXAMPLES_QA_FINAL_DATASET_PHAROS.json")
fixed_rejected_data = load_data("FIXED_REJECTED_QA_FINAL_DATASET_PHAROS.json")
print(f"Accepted data before append: {len(accepted_data)}")
print(f"Fixed rejected data appended: {len(fixed_rejected_data)}")

for example in fixed_rejected_data:
    example["question"] = example["fixed"]["question"]
    example["answer"] = example["fixed"]["answer"]

# print(fixed_rejected_data[0])
# print(accepted_data[0])

accepted_data.extend(fixed_rejected_data)

before_filtering = len(accepted_data)

accepted_data = [example for example in accepted_data if example.get("question", "").strip() and example.get("answer", "").strip()]
print(f"Records deleted: {before_filtering - len(accepted_data)}")

final_dataset = [{"title": example["title"],
        "full_text": example["full_text"],
        "question": example["question"],
        "answer": example["answer"]}
    for example in accepted_data]

with open("GR_WIKIPEDIA_BIOS_QA_FINAL_DATASET_PHAROS.json", "w", encoding="utf-8") as f:
    json.dump(final_dataset, f, ensure_ascii=False, indent=2) 

print(f"Total records saved: {len(final_dataset)}")
# print(final_dataset[0])
# print(final_dataset[142226])

unique_titles = sorted({example["title"].strip() for example in final_dataset if example.get("title", "").strip()})

with open("UNIQUE_PEOPLE_NAMES_GRWIKI.txt", "w", encoding="utf-8") as f:
    for title in unique_titles:
        f.write(title + "\n")

print(f"Unique titles saved: {len(unique_titles)}")

# Accepted data before append: 138142
# Fixed rejected data appended: 4085
# Records deleted: 33
# Total records saved: 142194 => 142200
# Unique titles saved: 16075

