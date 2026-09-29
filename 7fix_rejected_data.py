from openai import OpenAI
import json
import json_repair
import tqdm
import multiprocessing as mp
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, ValidationError

########################################################################################################################################################
########################################################################################################################################################
def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
data = load_data("REJECTED_EXAMPLES_QA_FINAL_DATASET_PHAROS.json")

########################################################################################################################################################
########################################################################################################################################################
SYSTEM_PROMPT = '''You are an expert editor of Greek question-answer datasets.

Your task is to correct a previously rejected question-answer pair using its validation results.

You will receive:
1. FULL_TEXT
2. QUESTION
3. ANSWER
4. VALIDATION

The VALIDATION contains labels and reasons for:
- question_validity
- answer_grounding
- answer_correctness
- language_quality
- overall

The FULL_TEXT is the only authoritative factual source. Do not use external knowledge.

Your goal is to return a corrected QUESTION and ANSWER that would pass all validation criteria.

### Correction procedure

#### 1. Question validity
- If question_validity is PASS, preserve the original question unless changing it is necessary to make the final question-answer pair coherent.
- If question_validity is FAIL, use its reason to repair the question.
- The corrected question must be meaningful, clear, unambiguous, and answerable solely from the FULL_TEXT.
- Preserve the original intent and subject whenever possible.
- If the original question cannot be repaired into a valid question that is supported by the article, return empty strings for BOTH question and answer.

#### 2. Answer grounding
- If answer_grounding is PASS, preserve the supported content of the answer.
- If answer_grounding is FAIL, identify the unsupported, invented, or contradictory content described in the validation reason.
- Remove or correct all unsupported claims.
- Every factual statement in the corrected answer must be supported by the FULL_TEXT.
- If no fully grounded answer can be produced, return empty strings for BOTH fields.

#### 3. Answer correctness
- If answer_correctness is FAIL, rewrite the answer so that it directly and correctly answers the final question.
- Do not include irrelevant information.
- If the question is changed, update the answer so that it answers the corrected question.
- A short, fully supported answer is preferable to a long answer containing unnecessary details.

#### 4. Greek language quality
- If language_quality is MINOR_ISSUES or FAIL, correct the affected QUESTION, ANSWER, or both.
- Correct grammar, spelling, syntax, punctuation, word choice, fluency, and naturalness.
- If language_quality is PASS, avoid unnecessary stylistic rewriting.

### Editing principles
- Make the smallest changes necessary to resolve all failed criteria.
- Preserve parts that have passed validation.
- Keep as much of the original intent as possible.
- Treat validation reasons as diagnostic guidance, not as an additional factual source.
- If a validation reason conflicts with the FULL_TEXT, follow the FULL_TEXT.
- Never add facts that are absent from the FULL_TEXT.
- Do not mention the validation process in the corrected question or answer.
- Both the final question and final answer must be written in natural Greek.
- Before returning the result, verify that:
  1. the question can be answered from the FULL_TEXT;
  2. the answer is fully grounded in the FULL_TEXT;
  3. the answer directly answers the final question;
  4. both fields have acceptable Greek language quality.

Return only the final corrected question and answer according to the supplied JSON schema.
'''

USER_PROMPT = '''FULL_TEXT:
{full_text}

ORIGINAL QUESTION:
{question}

ORIGINAL ANSWER:
{answer}

VALIDATION:
{validation}
'''


########################################################################################################################################################
########################################################################################################################################################

class QACorrection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        description="The final corrected Greek question."
    )
    answer: str = Field(
        description="The final corrected Greek answer."
    )

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "qa_correction",
        "strict": True,
        "schema": QACorrection.model_json_schema(),
    },
}

########################################################################################################################################################
########################################################################################################################################################
client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8016/v1")  # "http://localhost:8096/v1"

def ask_model(system, user):
    response = client.chat.completions.create(
        model= "google/gemma-4-12B-it", #google/gemma-4-26B-A4B-it   "google/gemma-4-12B-it", #/model/gemma3
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        response_format=RESPONSE_FORMAT,
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
                        full_text=example["full_text"],
                        question=example["question"],
                        answer=example["answer"],
                        validation=json.dumps(
                            example["validation"],
                            ensure_ascii=False,
                            indent=2
                        )
                    )
        )
        
        # print("CONTENT: ",content)

        json_output = json_repair.repair_json(
            content,
            return_objects=True
        )

        json_output = QACorrection.model_validate(json_output).model_dump(mode="json")
        
        # print("JSON OUTPUT: ",json_output)
 
        new = {
            "title": example["title"],
            "full_text": example["full_text"],
            "question": example["question"],
            "answer": example["answer"],
            "validation": example["validation"],
            "fixed": json_output
        }

    except Exception as e:
        print(f"Error processing example {i}: {e}")

        new = {
            "title": example["title"],
            "full_text": example["full_text"],
            "question": example["question"],
            "answer": example["answer"],
            "validation": example["validation"],
            "fixed": {}
        }

    return i, new
########################################################################################################################################################
########################################################################################################################################################
if __name__ == "__main__":

    new_dataset = [None] * len(data)

    with mp.Pool(processes=8) as pool:

        for i, new in tqdm.tqdm(pool.imap_unordered(process_example,enumerate(data, start=1)), total=len(data),desc="Processing examples"):
            new_dataset[i-1] = new
            # checkpoint every 100 completed examples
            completed = sum(x is not None for x in new_dataset)
            if completed % 500 == 0:
                with open(f"checkpoints/repaired/fixed_QA_dataset_checkpoint_{completed}.json","w",encoding="utf-8") as f:
                    json.dump([x for x in new_dataset if x is not None],f,ensure_ascii=False,indent=2)


    with open("FIXED_REJECTED_QA_FINAL_DATASET_PHAROS.json", "w", encoding="utf-8") as f:
            json.dump(new_dataset, f, ensure_ascii=False, indent=2)