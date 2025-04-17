import os
import json
from tasks.qa.qa_runner import run_qa_task
# from models.openai import OpenAIModel
from models.llama import LlamaModel
from models.qwen import QwenModel
from models.gemma import GemmaModel
from models.mistral import MistralModel
from models.mixtral import MixtralModel

# List of model names to be used
MODEL_NAMES = [
    "Llama-3.1-8B", "Llama-3.1-70B", "Qwen2-7B", "Qwen2-72B",
    "gemma-2-9b", "gemma-2-27b", "Mistral-7B", "Mixtral-8x7B"
]

# 
# MODEL_NAMES = ["Mistral-7B", "Mixtral-8x7B"]

# 
model_registry = {
    "llama": {
        model_name: lambda model_name=model_name: LlamaModel(model_name=model_name)
        for model_name in MODEL_NAMES if "Llama" in model_name
    },
    "qwen": {
        model_name: lambda model_name=model_name: QwenModel(model_name=model_name)
        for model_name in MODEL_NAMES if "Qwen" in model_name
    },
    "gemma": {
        model_name: lambda model_name=model_name: GemmaModel(model_name=model_name)
        for model_name in MODEL_NAMES if "gemma" in model_name
    },
    "mistral": {
        model_name: lambda model_name=model_name: MistralModel(model_name=model_name)
        for model_name in MODEL_NAMES if "Mistral" in model_name
    },
    "mixtral": {
        model_name: lambda model_name=model_name: MixtralModel(model_name=model_name)
        for model_name in MODEL_NAMES if "Mixtral" in model_name
    }
}

# Dataset and output directory setup
DATASETS = ["quoref.json", "drop.json", "hotpotqa.json", "2wiki.json"]
DATA_DIR = "data"
OUTPUT_DIR = "outputs/qa_inference"
os.makedirs(OUTPUT_DIR, exist_ok=True)


for family, versions in model_registry.items():
    for model_name, model_loader in versions.items():
        model = model_loader(model_name) 
        print(f"Running QA for model: {family} {model_name}")

        for dataset_file in DATASETS:
            with open(os.path.join(DATA_DIR, dataset_file), "r", encoding="utf-8") as f:
                samples = json.load(f)
                samples = samples[0:5]

            results = []
            for item in samples:
                prompt, answer = run_qa_task(item["question"], item["context"], model)
                results.append({
                    "_id": item["_id"],
                    "question": item["question"],
                    "answer": item["answer"],
                    "answer_type": item["answer_type"],
                    "context": item["context"],
                    "generated_ans": answer,
                    "user_prompt": prompt
                })

            output_path = os.path.join(
                OUTPUT_DIR, f"{model_name.lower()}_{dataset_file}"
            )
            print(f"Saving output to: {output_path}")
            with open(output_path, "w", encoding="utf-8") as out_f:
                json.dump(results, out_f, indent=2, ensure_ascii=False)
