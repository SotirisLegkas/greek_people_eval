from openai import OpenAI
import json
import json_repair
import tqdm
import multiprocessing as mp

########################################################################################################################################################
########################################################################################################################################################
def load_data(path):
    data=[]
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)
            data.append({"title":example["title"],"full_text":example["full_text"]})
    return data
    
data = load_data("greek_biographies.jsonl")

########################################################################################################################################################
########################################################################################################################################################
SYSTEM_PROMPT = """
You are an expert dataset creator specializing in biographies.

You will receive:
1. The name of a person.
2. A biography of that person.

Your task is to create multiple questions about the person.

All questions must be written in Greek.

Rules:
1. Use only information explicitly stated in the provided biography.
2. Do not use external knowledge.
3. Do not invent names, dates, places, relatives, events, works, awards, or other facts.
4. Do not create a question when its answer is not available in the biography.
5. Create varied questions covering different aspects of the person's life.
6. Do not create several questions that are only minor paraphrases of one another.
7. Every question must include the person's name so that it can be understood independently.
8. Preserve names, dates, locations, titles, and other details exactly as they appear in the biography.
9. Create one general question asking who the person was or is.
10. When the biography contains the relevant information, create questions about:
    - birth date and place of birth
    - origin and childhood
    - parents and siblings
    - spouse, partner, and children
    - education
    - profession and career
    - political activity
    - important events and achievements
    - books, works, films, music, research, or publications
    - awards and distinctions
    - death and place of burial
    - legacy and influence
11. For a short biography, create only the questions supported by the available information.
12. For a long biography, create up to 10 varied questions.
13. Do not create questions about references, bibliography, external links, or Wikipedia formatting.
14. Return only valid JSON.
15. Do not use Markdown or JSON code fences.

Return exactly this structure:

{{
  "questions": [
    {{
      "question": "A question written in Greek",
      "question_type": "overview"
    }}
  ]
}}

The question_type must be one of:

overview
birth
origin
childhood
family
education
career
politics
works
achievements
awards
death
legacy
timeline
other
""".strip()


USER_PROMPT = """
Person's name:
{title}

Biography:
{biography}

Create questions about this person.

Remember:
- Write every question in Greek.
- Use only facts contained in the biography.
- Return only the requested JSON object.
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
########################################################################################################################################################
def process_example(args):
    i, example = args

    try:
        content = ask_model(
            SYSTEM_PROMPT,
            USER_PROMPT.format(
                title=example["title"],
                biography=example["full_text"]
            )
        )

        json_output = json_repair.repair_json(
            content,
            return_objects=True
        )

        new = {
            "title": example["title"],
            "full_text": example["full_text"],
            "questions": json_output
        }

    except Exception as e:
        print(f"Error processing example {i}: {e}")

        new = {
            "title": example["title"],
            "full_text": example["full_text"],
            "questions": {}
        }

    return i, new

if __name__ == "__main__":

    new_dataset = [None] * len(data)

    with mp.Pool(processes=4) as pool:

        for i, new in tqdm.tqdm(pool.imap_unordered(process_example,enumerate(data, start=1)), total=len(data),desc="Processing examples"):
            
            new_dataset[i-1] = new
            # checkpoint every 100 completed examples
            completed = sum(x is not None for x in new_dataset)
            if completed % 100 == 0:
                with open(f"llm_qa_checkpoint_{completed}.json","w",encoding="utf-8") as f:
                    json.dump([x for x in new_dataset if x is not None],f,ensure_ascii=False,indent=2)


    with open("llm_qa.json", "w", encoding="utf-8") as f:
            json.dump(new_dataset, f, ensure_ascii=False, indent=2)
            
            
            
            
            
            
            
            
            
            
            
            
            
# new_dataset=[]

# for i, example in enumerate(tqdm.tqdm(data, desc="Processing examples"), start=1):
#     content = ask_model(SYSTEM_PROMPT,USER_PROMPT.format(title=example["title"], biography=example["full_text"]))
#     try:
#         json_output = json_repair.repair_json(content, return_objects=True)
#         print(json_output)
#         print("*"*50)
#         new = {"title":example['title'], "full_text":example["full_text"], "questions":json_output}
#     except:
#         new = {"title":example['title'], "full_text":example["full_text"], "questions": {}}
#     new_dataset.append(new)
#     if i % 100 == 0:
#         with open(f"llm_qa_checkpoint_{i}.json", "w", encoding="utf-8") as f:
#             json.dump(new_dataset, f, ensure_ascii=False, indent=2)
########################################################################################################################################################
########################################################################################################################################################
