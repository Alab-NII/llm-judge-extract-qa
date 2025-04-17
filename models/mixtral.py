import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.base_model import BaseModel


class MixtralModel(BaseModel):
    def __init__(self, model_name="Mixtral-8x7B"):
        """
        """
        super().__init__(model_name=model_name)
        self.model_id = f"mistralai/{model_name}-Instruct-v0.1"
        self.device = "cuda"

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

    def call(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(inputs, max_new_tokens=512)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return prompt, answer
