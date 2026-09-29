import json

########################################################################################################################################################
########################################################################################################################################################
def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
data = load_data("NEWclean_QA_FINAL_DATASET_PHAROS.json")
########################################################################################################################################################
########################################################################################################################################################
final_dataset=[]
n=0
k=0
for example in data:
    if example['validation']=={}:
        n=n+1
        # print("n:",n)
    elif example['validation']['overall']=="ACCEPT":
    # elif example['validation']['overall']!="ACCEPT":
        k=k+1
        print("k:",k)
        # print(example["title"])
        final_dataset.append(example)
# print(len(final_dataset))        
    #     continue
    # else:
    #     final_dataset.append(example)


with open("ACCEPTED_EXAMPLES_QA_FINAL_DATASET_PHAROS.json", "w", encoding="utf-8") as f:
    json.dump(final_dataset, f, ensure_ascii=False, indent=2) 
    
# with open("REJECTED_EXAMPLES_QA_FINAL_DATASET_PHAROS.json", "w", encoding="utf-8") as f:
#     json.dump(final_dataset, f, ensure_ascii=False, indent=2)        
        
# with open("CLEANED_QA_FINAL_DATASET_PHAROS.json", "w", encoding="utf-8") as f:
#     json.dump(final_dataset, f, ensure_ascii=False, indent=2)


#142241 examples

# 34584
# 42309
# 61226
# 65986
# 66963
# 99594
# 128012