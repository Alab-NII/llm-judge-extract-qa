import torch
from transformers import pipeline
from models.base_model import BaseModel

class LlamaModel(BaseModel):
    def __init__(self, model_name="Llama-3.1-8B"):
        """
        """
        super().__init__(model_name=model_name)  
        model_id = f"meta-llama/Meta-{model_name}-Instruct"
        self.model_id = model_id
        
        # Set up the pipeline for text generation
        self.pipeline = pipeline(
            "text-generation", 
            model=self.model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto", 
        )

    def call(self, prompt: str) -> str:
        """
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        # Apply chat template
        input_ids = self.pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        # Define terminators
        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>") 
        ]

        # Generate output from the model
        outputs = self.pipeline(
            input_ids,
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=False
        )

        # 
        answer = outputs[0]["generated_text"][len(input_ids):].strip()
        return prompt, answer
