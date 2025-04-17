# run_judge.py

import os
import json
from tqdm import tqdm

from tasks.llm_judge.judge_runner import run_judge_task
from models.llama_3 import LlamaModel3
from models.mistral_v03 import MistralModel3
from models.qwen import QwenModel
from models.llama import LlamaModel
from utils import extract_answer, extract_answer_mistral_judge
from evaluation.hotpot_eval import get_scores


# QA models
MODEL_NAMES = [
    "Llama-3.1-8B",
    "Llama-3.1-70B",
    "Qwen2-7B",
    "Qwen2-72B",
    "gemma-2-9b",
    "gemma-2-27b",
    "Mistral-7B",
    "Mixtral-8x7B"
]

DATASET_NAMES = ["quoref", "drop", "hotpotqa", "2wiki"]

# Directories
QA_INPUT_DIR = "outputs/qa_postprocess"
OUTPUT_DIR = "outputs/judge"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 
judge_models = {
    # "mistral-v0.3": MistralModel3(model_name="Mistral-7B-Instruct-v0.3"),
    # "llama-3.3-70b": LlamaModel3(model_name="Llama-3.3-70B"),
    # "qwen-2.5-72b": QwenModel(model_name="Qwen2.5-72B"),
    # "qwen-2-72b": QwenModel(model_name="Qwen2-72B"),
    "qwen-2-7b": QwenModel(model_name="Qwen2-7B"),
    # "llama-3.1-70b": LlamaModel(model_name="Llama-3.1-70B"),
    "llama-3.1-8b": LlamaModel(model_name="Llama-3.1-8B")
}

# Loop through judgment LLMs
for judge_key, judge_model in judge_models.items():
    for qa_model in MODEL_NAMES:
        qa_model_id = qa_model.lower()
        for dataset_name in DATASET_NAMES:
            input_filename = f"{qa_model_id}_{dataset_name}.json"
            input_path = os.path.join(QA_INPUT_DIR, input_filename)

            if not os.path.exists(input_path):
                print(f"[!] File not found: {input_path}")
                continue

            output_subdir = os.path.join(OUTPUT_DIR, judge_key)
            os.makedirs(output_subdir, exist_ok=True)
            
            output_path = os.path.join(output_subdir, f"{qa_model_id}_{dataset_name}.json")

            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            #
            data_out = []
            #####
            for item in tqdm(data, desc=f"{judge_key} judging {qa_model_id} on {dataset_name}"):
                pred = str(item.get("pred_answer", "")).strip()
                gold = str(item.get("answer", "")).strip()
                question = item["question"]
                context = item["context"]

                # Basic structure
                new_item = {
                    "_id": item["_id"],
                    "question": question,
                    "gold_ans": gold,
                    "generated_ans": item.get("generated_ans", ""),
                    "pred_answer": pred,
                    "answer_type": item.get("answer_type", "")
                }

                # 
                em, f1, prec, recall = get_scores(pred, gold)
                new_item["EM"] = em
                new_item["f1"] = f1

                # Early decision logic
                if em:
                    new_item["LLM-judge"] = "CORRECT"
                elif pred == "":
                    new_item["LLM-judge"] = "INCORRECT"
                else:
                    # Only call model if needed 
                    _, judge_output = run_judge_task(
                        ques=question,
                        gold_ans=gold,
                        pred_ans=pred,
                        context=context,
                        model=judge_model
                    )
                    if judge_key == "mistral-v0.3":
                        new_item["LLM-judge"] = extract_answer_mistral_judge(judge_output).strip()
                    else:
                        new_item["LLM-judge"] = extract_answer(judge_output).strip()
                    #
                    new_item["LLM-judge-generated"] = judge_output

                data_out.append(new_item)


            #####
            with open(output_path, "w", encoding="utf-8") as f_out:
                json.dump(data_out, f_out, indent=2, ensure_ascii=False)

            print(f"[✓] Saved: {output_path}")

