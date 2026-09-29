import json

########################################################################################################################################################
########################################################################################################################################################
def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
data = load_data("dataset/llm_qa.json")

# print(data[0]['questions']['questions'])

new_dataset = []
for example in data:
    for question_dict in example['questions']['questions']:
        new_dataset.append({"title":example['title'],"full_text": example['full_text'],"question": question_dict['question'],"question_type": question_dict['question_type']})
        
with open("llm_questions_final.json", "w", encoding="utf-8") as f:
    json.dump(new_dataset, f, ensure_ascii=False, indent=2)