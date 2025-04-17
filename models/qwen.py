import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.base_model import BaseModel


class QwenModel(BaseModel):
    def __init__(self, model_name="Qwen2-7B"):
        super().__init__(model_name=model_name)  
        self.model_id = f"Qwen/{model_name}-Instruct"
        self.device = "cuda"

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

        # Format with chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        # Generate output
        with torch.no_grad():
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=512
            )

        # Remove input tokens from generated_ids
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        answer = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return prompt, answer
