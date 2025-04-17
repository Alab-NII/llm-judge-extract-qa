import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.base_model import BaseModel

class MistralModel(BaseModel):
    def __init__(self, model_name="Mistral-7B"):
        """
        """
        super().__init__(model_name=model_name)  
        self.model_name = model_name
        self.model_id = f"mistralai/{model_name}-Instruct-v0.1"
        self.device = "auto"

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

    def call(self, prompt: str) -> str:
        """
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt"
        ) #.to(self.device)

        # self.model.to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                max_new_tokens=512,
                do_sample=False
            )
        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        answer = decoded[0]

        return prompt, answer
