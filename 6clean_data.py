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
    
data = load_data("CLEANED_QA_FINAL_DATASET_PHAROS.json")

########################################################################################################################################################
########################################################################################################################################################
# SYSTEM_PROMPT ='''You are an expert evaluator of Greek question-answer datasets.

# Your task is to evaluate the quality of a question-answer pair based ONLY on the provided Greek text.

# You will receive:
# 1. TITLE: The name of a person.
# 2. FULL_TEXT: a Wikipedia article in Greek.
# 3. QUESTION: a question in Greek.
# 4. ANSWER: the proposed answer in Greek.

# Evaluate the example according to the following criteria:

# ### Criterion 1: Question validity
# Determine whether the question:
# - is meaningful and grammatically correct Greek.
# - is clear and unambiguous.
# - can be answered solely from the FULL_TEXT.
# - is not misleading or self-contradictory.
# - does not require outside knowledge.

# Return:
# - PASS
# - FAIL
# If FAIL, briefly explain why.

# ### Criterion 2: Answer grounding
# Determine whether the answer:
# - is completely supported by the FULL_TEXT.
# - does not contain any information that is absent from the FULL_TEXT.
# - does not hallucinate facts.
# - does not contradict the FULL_TEXT.

# Small paraphrases are allowed, but no new factual information.

# Return:
# - PASS
# - FAIL
# If FAIL, explain exactly which part is unsupported or incorrect.

# ### Criterion 3: Answer correctness
# Determine whether the answer correctly answers the question.

# Return:
# - PASS
# - FAIL
# If FAIL, explain why.

# ### Criterion 4: Greek language quality
# Evaluate both the QUESTION and the ANSWER. 

# Check:
# - grammar
# - spelling
# - syntax
# - punctuation
# - naturalness
# - word choice
# - fluency

# Minor stylistic preferences should not be penalized.

# Return:
# - PASS
# - MINOR_ISSUES
# - FAIL
# If not PASS, explain the issues.

# ### Criterion 5: Overall quality

# Return one of:
# - ACCEPT
# - REJECT

# Reject the example if any of the following is true:
# - the question is invalid
# - the answer is unsupported
# - the answer does not answer the question
# - the Greek contains serious language errors

# Return ONLY valid JSON with the following schema:
# {{
#   "question_validity": {{
#     "label": "PASS | FAIL",
#     "reason": ""
#   }},
#   "answer_grounding": {{
#     "label": "PASS | FAIL",
#     "reason": ""
#   }},
#   "answer_correctness": {{
#     "label": "PASS | FAIL",
#     "reason": ""
#   }},
#   "language_quality": {{
#     "label": "PASS | MINOR_ISSUES | FAIL",
#     "reason": ""
#   }},
#   "overall": "ACCEPT | REJECT"
# }}

# Evaluate strictly.

# Use ONLY the FULL_TEXT as evidence.

# Do NOT use any external knowledge, even if you know the subject.
# '''

SYSTEM_PROMPT = '''You are an expert evaluator of Greek question-answer datasets.

Your task is to evaluate the quality of a question-answer pair based ONLY on the provided Greek text.

You will receive:
1. TITLE: The name of a person.
2. FULL_TEXT: a Wikipedia article in Greek.
3. QUESTION: a question in Greek.
4. ANSWER: the proposed answer in Greek.

Evaluate the example according to the following criteria:

### Criterion 1: Question validity
Determine whether the question:
- is meaningful, clear, and unambiguous.
- can be answered solely from the FULL_TEXT.
- is not misleading or self-contradictory.
- does not require outside knowledge.

Minor language issues do not make the question invalid if its meaning remains clear.

Return:
- PASS
- FAIL

If FAIL, briefly explain why. Otherwise, return an empty reason.

### Criterion 2: Answer grounding
Determine whether the answer:
- is completely supported by the FULL_TEXT.
- does not introduce factual information absent from the FULL_TEXT.
- does not hallucinate facts.
- does not contradict the FULL_TEXT.

Small paraphrases are allowed, but no new factual information.

Return:
- PASS
- FAIL

If FAIL, explain exactly which part is unsupported or incorrect. Otherwise, return an empty reason.

### Criterion 3: Answer correctness
Determine whether the answer directly and correctly answers the question.

Return:
- PASS
- FAIL

If FAIL, explain why. Otherwise, return an empty reason.

### Criterion 4: Greek language quality
Evaluate both the QUESTION and the ANSWER.

Check:
- grammar
- spelling
- syntax
- punctuation
- naturalness
- word choice
- fluency

Minor stylistic preferences should not be penalized.

Return:
- PASS
- MINOR_ISSUES
- FAIL

Use MINOR_ISSUES for small language problems that do not affect meaning.
Use FAIL only for serious language problems that make the question or answer unclear or unnatural.

If not PASS, identify whether the issue is in the QUESTION, the ANSWER, or both.
Otherwise, return an empty reason.

### Criterion 5: Overall quality
Return:
- ACCEPT
- REJECT

Return REJECT if:
- question_validity is FAIL, or
- answer_grounding is FAIL, or
- answer_correctness is FAIL, or
- language_quality is FAIL.

Otherwise, return ACCEPT.

An example with language_quality MINOR_ISSUES may still be ACCEPTED.

Return an evaluation that follows the supplied JSON schema.

Evaluate strictly.
Use ONLY the FULL_TEXT as evidence.
Do NOT use external knowledge, even if you know the subject.
'''


USER_PROMPT = '''TITLE:
{title}

FULL_TEXT:
{full_text}

QUESTION:
{question}

ANSWER:
{answer}'''

########################################################################################################################################################
########################################################################################################################################################
class PassFail(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


class LanguageLabel(str, Enum):
    PASS = "PASS"
    MINOR_ISSUES = "MINOR_ISSUES"
    FAIL = "FAIL"


class OverallLabel(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"


class PassFailEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: PassFail
    reason: str = Field(description="Empty when the label is PASS; otherwise a brief explanation.")


class LanguageEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: LanguageLabel
    reason: str = Field(description="Empty when the label is PASS; otherwise an explanation.")


class QAEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question_validity: PassFailEvaluation
    answer_grounding: PassFailEvaluation
    answer_correctness: PassFailEvaluation
    language_quality: LanguageEvaluation
    overall: OverallLabel


RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "qa_evaluation",
        "strict": True,
        "schema": QAEvaluation.model_json_schema(),},
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
        response_format=RESPONSE_FORMAT, # NEW
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
                full_text=example['full_text'],
                question=example['question'],
                answer=example['answer']
            )
        )

        json_output = json_repair.repair_json(
            content,
            return_objects=True
        )
        
        json_output = QAEvaluation.model_validate(json_output).model_dump(mode="json") # NEW
        # print(json_output)
 
        new = {
            "title": example["title"],
            "full_text": example["full_text"],
            "question": example['question'],
            "answer": example['answer'],
            "validation": json_output
        }

    except Exception as e:
        print(f"Error processing example {i}: {e}")

        new = {
            "title": example["title"],
            "full_text": example["full_text"],
            "question": example['question'],
            "answer": example['answer'],
            "validation": {}
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
            if completed % 1000 == 0:
                with open(f"checkpoints/clean_QA_dataset_checkpoint_{completed}.json","w",encoding="utf-8") as f:
                    json.dump([x for x in new_dataset if x is not None],f,ensure_ascii=False,indent=2)


    with open("NEWclean_QA_FINAL_DATASET_PHAROS.json", "w", encoding="utf-8") as f:
            json.dump(new_dataset, f, ensure_ascii=False, indent=2)