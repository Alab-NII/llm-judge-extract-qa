import torch
from transformers import pipeline
from models.base_model import BaseModel

class GemmaModel(BaseModel):
    def __init__(self, model_name="gemma-2-9b"):
        """
        """
        super().__init__(model_name=model_name)  

        model_id = f"google/{model_name}-it"
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
            # {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>") 
        ]

        # Generate output from the model
        outputs = self.pipeline(
            messages,
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=False
        )
        answer = outputs[0]["generated_text"][-1]["content"].strip()
        # 
        return prompt, answer
