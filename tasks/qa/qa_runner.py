# tasks/qa/qa_runner.py


from prompts.qa_prompt_template import build_qa_prompt
from models.base_model import BaseModel


def run_qa_task(question: str, context: str, model: BaseModel) -> str:
    prompt = build_qa_prompt(question, context)
    return model.call(prompt)
