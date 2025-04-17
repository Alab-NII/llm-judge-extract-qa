from tqdm import tqdm
import re
from collections import defaultdict, deque
import os
import csv
import json
import numpy as np
from evaluation.hotpot_eval import update_answer
from evaluation.drop_eval import evaluate_json_custom_one_sample
from evaluation.quoref_eval import evaluate_one_sample
from utils import convert_list2dict, extract_answer_mistral_judge


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

DATASET_NAMES = ["quoref", "drop", "hotpotqa", "2wiki"]

# Directories
INPUT_DIR = "outputs/judge"

JUDGE_NAMES = ["mistral-v0.3", "llama-3.3-70b", "qwen-2.5-72b"]


with open("data/drop.json", "r", encoding="utf-8") as f:
    data_drop = json.load(f)
data_drop_dict = convert_list2dict(data_drop)


def eval_one_file(data, data_name): 
    count_correct = 0
    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0}
    for item in tqdm(data):
        # for some cases that we don't extract judge response yet
        if "LLM-judge" not in item:
            item["LLM-judge"] = extract_answer_mistral_judge(item["LLM-judge-generated"]).strip()
        #
        item["pred_answer"] = str(item["pred_answer"])
        item["gold_ans"] = str(item["gold_ans"])
        if data_name == "drop":
            predicted = item["pred_answer"]
            annotation = data_drop_dict[item["_id"]]
            em, f1 = evaluate_json_custom_one_sample(annotation, predicted)
            metrics['em'] += float(em)
            metrics['f1'] += f1
        # 
        elif data_name == "quoref":
            candidate_answers = item["gold_ans"]
            predicted_answer = item["pred_answer"]
            em, f1 = evaluate_one_sample(candidate_answers, predicted_answer)
            metrics['em'] += float(em)
            metrics['f1'] += f1
        else:
            em, f1, prec, recall = update_answer(metrics, item["pred_answer"], item["gold_ans"])
        if item["LLM-judge"] == "CORRECT":
            count_correct += 1
    #
    judge_score = round((count_correct/len(data)*100), 1)
    # 
    for k in metrics.keys():
        metrics[k] = round(metrics[k]/len(data)*100, 1)
    #
    return metrics['em'], metrics['f1'], judge_score


csv_output_path = "outputs/judge/eval_summary.csv"

rows = []

# Main
# Loop through judgment LLMs
for judge_name in JUDGE_NAMES:
    for dataset_name in DATASET_NAMES:
        for qa_model in MODEL_NAMES:
            qa_model_id = qa_model.lower()
        
            input_subdir = os.path.join(INPUT_DIR, judge_name)

            input_filename = f"{qa_model_id}_{dataset_name}.json"
            input_path = os.path.join(input_subdir, input_filename)

            if not os.path.exists(input_path):
                print(f"[!] File not found: {input_path}")
                continue

            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # print(f"Length Data: {len(data)}")
            em, f1, judge_score = eval_one_file(data, dataset_name)
            row = {
                "qa_model": qa_model,
                "dataset": dataset_name,
                "judge_model": judge_name,
                "EM": em,
                "F1": f1,
                "LLM-Judge Score": round(judge_score, 4),
            }
            rows.append(row)


os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
with open(csv_output_path, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["qa_model", "dataset", "judge_model", "EM", "F1", "LLM-Judge Score"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(rows)