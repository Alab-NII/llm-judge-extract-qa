# models/base_model.py

class BaseModel:
    def __init__(self, model_name, system_prompt="You are an expert in question answering systems."):
        """
        """
        self.model_name = model_name
        self.system_prompt = system_prompt

    def call(self, *args, **kwargs):
        """
        """
        raise NotImplementedError("Each model must implement the call method.")
