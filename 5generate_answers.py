from openai import OpenAI
import json
import json_repair
import tqdm
import multiprocessing as mp

########################################################################################################################################################
########################################################################################################################################################
def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
data = load_data("llm_questions_final.json")

########################################################################################################################################################
########################################################################################################################################################
SYSTEM_PROMPT = """
You are creating a Greek question-answering dataset from biographies.

You will receive:
1. The name of a person.
2. A biography.
3. One question about the person.

Answer the question using only information explicitly stated in the biography.

Rules:
1. Write the answer in Greek.
2. Write a complete, natural, and factually correct answer.
3. Include all information from the biography that is directly relevant to the question.
4. Do not use external knowledge.
5. Do not invent, assume, or infer unsupported information.
6. If the question cannot be answered from the biography, return an empty answer.
7. Return only valid JSON.
8. Do not use Markdown or JSON code fences.

Return exactly this structure:
{{
  "answer": "A complete answer in Greek containing all relevant information."
}}
""".strip()


USER_PROMPT = """
Person's name:
{title}

Biography:
{biography}

Question:
{question}

Answer the question using all directly relevant information from the biography.
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
def process_example(args):
    i, example = args

    try:
        content = ask_model(
            SYSTEM_PROMPT,
            USER_PROMPT.format(
                title=example['title'],
                biography=example['full_text'],
                question=example['question']
            )
        )

        json_output = json_repair.repair_json(
            content,
            return_objects=True
        )

        new = {
            "title": example["title"],
            "full_text": example["full_text"],
            "question": example['question'],
            "answer": json_output['answer']
        }

    except Exception as e:
        print(f"Error processing example {i}: {e}")

        new = {
            "title": example["title"],
            "full_text": example["full_text"],
            "question": example['question'],
            "answer": ""
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
                with open(f"checkpoints/QA_dataset_checkpoint_{completed}.json","w",encoding="utf-8") as f:
                    json.dump([x for x in new_dataset if x is not None],f,ensure_ascii=False,indent=2)


    with open("QA_FINAL_DATASET_PHAROS.json", "w", encoding="utf-8") as f:
            json.dump(new_dataset, f, ensure_ascii=False, indent=2)