# tasks/llm_judge/judge_runner.py

from prompts.judge_prompt_template import build_judge_prompt
from models.base_model import BaseModel

def run_judge_task(ques: str, gold_ans: str, pred_ans: str, context: str, model: BaseModel) -> str:
    prompt = build_judge_prompt(ques, gold_ans, pred_ans, context)
    return model.call(prompt)

