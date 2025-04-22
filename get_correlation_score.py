"""
Get correlation scores with human judgements
Input: a QA model name and a judgement model name 
Output: a correlation score 
"""
import os
import json
from utils import pearson_correlation, extract_answer_mistral_judge


model_mapping = {
    "llama-3.1-8b": 1,
    "llama-3.1-70b": 2,
    "qwen2-7b": 3,
    "qwen2-72b": 4,
    "gemma-2-9b": 5,
    "gemma-2-27b": 6,
    "mistral-7b": 7,
    "mixtral-8x7b": 8
}

def get_ids_use(data_human):
    return [item["_id"] for item in data_human]


human_file = "data/human_result.json"
with open(human_file, "r", encoding="utf-8") as f:
    data_human = json.load(f)

ids_use_human = get_ids_use(data_human)

DATASET_NAMES = ["quoref", "drop", "hotpotqa", "2wiki"]

# Directories
INPUT_DIR = "outputs/judge"


def cal_correlation_score(qa_model, judge_model):
    """
    qa_model = llama-3.1-8b (must be one in the list of 8 QA models)
    judge_model = mistral-v0.3 or qwen-2.5-72b or llama-3.3-70b
    """
    qa_model_id = model_mapping[qa_model]

    input_subdir = os.path.join(INPUT_DIR, judge_model)

    all_pred_data = []
    for dataset_name in DATASET_NAMES:
        input_filename = f"{qa_model}_{dataset_name}.json"
        input_path = os.path.join(input_subdir, input_filename)
        # print(input_path)
        if not os.path.exists(input_path):
            print(f"[!] File not found: {input_path}")
            continue

        with open(input_path, "r", encoding="utf-8") as f:
            all_pred_data += json.load(f)

    # 
    id_to_item = {item["_id"]: item for item in all_pred_data}
    pred_use = [id_to_item[i] for i in ids_use_human]
    #
    assert len(pred_use) == len(data_human), "Lists are not the same length."
    human_label = []
    judge_label = []
    for idx, (item1, item2) in enumerate(zip(pred_use, data_human)):
        id1 = item1.get("_id")
        id2 = item2.get("_id")
        assert id1 == id2, f"Mismatch at index {idx}: {id1} != {id2}"
        
        human_label.append(int(item2[f"human_label_{qa_model_id}"]))
        if "LLM-judge" not in item1:
            item1["LLM-judge"] = extract_answer_mistral_judge(item1["LLM-judge-generated"]).strip()
        judge_label.append(1 if item1["LLM-judge"] == "CORRECT" else 0)

    score = round(pearson_correlation(human_label, judge_label), 3)
    return score 


def cal_correlation_score_em_f1(qa_model, judge_model):
    """
    qa_model = llama-3.1-8b (must be one in the list of 8 QA models)
    judge_model = any model, just for getting em and f1 scores
    """
    qa_model_id = model_mapping[qa_model]

    input_subdir = os.path.join(INPUT_DIR, judge_model)

    all_pred_data = []
    for dataset_name in DATASET_NAMES:
        input_filename = f"{qa_model}_{dataset_name}.json"
        input_path = os.path.join(input_subdir, input_filename)
        # print(input_path)
        if not os.path.exists(input_path):
            print(f"[!] File not found: {input_path}")
            continue

        with open(input_path, "r", encoding="utf-8") as f:
            all_pred_data += json.load(f)

    # 
    id_to_item = {item["_id"]: item for item in all_pred_data}
    pred_use = [id_to_item[i] for i in ids_use_human]
    #
    assert len(pred_use) == len(data_human), "Lists are not the same length."
    human_label = []
    em_label = []
    f1_label = []
    for idx, (item1, item2) in enumerate(zip(pred_use, data_human)):
        id1 = item1.get("_id")
        id2 = item2.get("_id")
        assert id1 == id2, f"Mismatch at index {idx}: {id1} != {id2}"
        
        human_label.append(int(item2[f"human_label_{qa_model_id}"]))
        #
        em_label.append(1 if item1["EM"] == True else 0)
        # print(em_label)
        f1_label.append(1 if item1["f1"] >=0.5 else 0)

    score_em = round(pearson_correlation(human_label, em_label), 3)
    score_f1 = round(pearson_correlation(human_label, f1_label), 3)
    return score_em, score_f1 


# QA models
MODEL_NAMES = [
    "Mistral-7B",
    "Mixtral-8x7B",
    "Qwen2-7B",
    "Qwen2-72B",
    "gemma-2-9b",
    "gemma-2-27b",
    "Llama-3.1-8B",
    "Llama-3.1-70B"
]

if __name__ == "__main__":
    # ====== correlation scores for em and f1
    print("=======This is correlation scores for EM and F1 ==============")
    judge_model = "qwen-2.5-72b" # mistral-v0.3 or llama-3.3-70b or qwen-2.5-72b 
    average_em, average_f1 = 0, 0
    for qa_model in MODEL_NAMES:
        print(f"QA model: {qa_model}")
        em, f1 = cal_correlation_score_em_f1(qa_model.lower(), judge_model)
        average_em += em
        average_f1 += f1
        print(f"EM: {em} ==== F1: {f1}")
    print(f"Average EM: {round(average_em/len(MODEL_NAMES), 3)}")
    print(f"Average F1: {round(average_f1/len(MODEL_NAMES), 3)}")

    # 
    # ====== correlation scores for llm-as-a-judge
    JUDGE_NAMES = ["mistral-v0.3", "llama-3.3-70b", "qwen-2.5-72b"]

    # judge_model = "llama-3.3-70b" # mistral-v0.3 or llama-3.3-70b or qwen-2.5-72b 
    for judge_model in JUDGE_NAMES: 
        print(f"=======This is correlation scores for Judge model: {judge_model}==============")
        average = 0
        for qa_model in MODEL_NAMES:
            print(f"QA model: {qa_model}")
            average += cal_correlation_score(qa_model.lower(), judge_model)
            print(cal_correlation_score(qa_model.lower(), judge_model))
        print(f"Average: {round(average/len(MODEL_NAMES), 3)}")