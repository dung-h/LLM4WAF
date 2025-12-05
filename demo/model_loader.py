# demo/model_loader.py
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import gc
import logging

logger = logging.getLogger(__name__)

class LLMModelLoader:
    _instance = None # Singleton instance
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMModelLoader, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.tokenizer = None
            cls._instance.loaded_model_id = None
            cls._instance.loaded_adapter_path = None
        return cls._instance

    def load_model(self, base_model_id: str, adapter_path: str):
        if self.model and self.loaded_model_id == base_model_id and self.loaded_adapter_path == adapter_path:
            logger.info(f"Model {base_model_id} with adapter {adapter_path} already loaded.")
            return self.model, self.tokenizer

        self.unload_model() # Unload any previously loaded model

        logger.info(f"Loading base model: {base_model_id} with adapter: {adapter_path}")
        
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                quantization_config=bnb_config,
                device_map="auto",
                token=os.environ.get("HF_TOKEN"),
                trust_remote_code=True
            )
            
            tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=os.environ.get("HF_TOKEN"), trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left" # Important for batch generation

            # Load LoRA adapter
            model = PeftModel.from_pretrained(model, adapter_path)
            model.eval() # Set to eval mode for inference

            self.model = model
            self.tokenizer = tokenizer
            self.loaded_model_id = base_model_id
            self.loaded_adapter_path = adapter_path
            logger.info("Model loaded successfully!")
            return self.model, self.tokenizer
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.unload_model() # Ensure clean state on failure
            raise

    def unload_model(self):
        if self.model:
            logger.info(f"Unloading model: {self.loaded_model_id} with adapter: {self.loaded_adapter_path}")
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            self.loaded_model_id = None
            self.loaded_adapter_path = None
            
            # Aggressive garbage collection and VRAM clear
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("Model unloaded.")

    def get_loaded_model(self):
        return self.model, self.tokenizer, self.loaded_model_id, self.loaded_adapter_path

# Global instance for easy access across Gradio app
model_loader = LLMModelLoader()
