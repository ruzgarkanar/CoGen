import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from typing import Dict, Tuple
import os
import logging
from pathlib import Path
from ..config.settings import HF_TOKEN, MODEL_NAME

class ModelManager:
    def __init__(self, model_dir: str):
        self.logger = logging.getLogger(__name__)
        self.model_dir = Path(model_dir)
        
        self.model_path = self.model_dir / "model"
        self.tokenizer_path = self.model_dir / "tokenizer"
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.tokenizer_path.mkdir(parents=True, exist_ok=True)
        
        self.model_name = MODEL_NAME
        
        self.device = torch.device('cpu')  
        self.logger.info(f"Using device: {self.device}")
        
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        try:
            self.logger.debug("Checking model files...")
            model_files_exist = (
                os.path.exists(os.path.join(self.model_path, "pytorch_model.bin")) and
                os.path.exists(os.path.join(self.model_path, "config.json"))
            )
            
            if model_files_exist:
                self.logger.info("Loading model from local files...")
                self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
                
                if not self.model or not self.tokenizer:
                    raise ValueError("Model or tokenizer loading failed")
                    
                self.logger.info("Model loaded successfully")
            else:
                self.logger.info(f"Downloading model {self.model_name}")
                self.model = AutoModelForQuestionAnswering.from_pretrained(
                    self.model_name,
                    token=HF_TOKEN
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    token=HF_TOKEN
                )
                self.save_model()
                self.logger.info("Model downloaded and saved")
                
            if self.model is not None:
                self.model = self.model.to(self.device)
                
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def save_model(self):
        try:
            self.model = self.model.cpu()
            
            state_dict = self.model.state_dict()
            
            for key in state_dict:
                if torch.is_tensor(state_dict[key]):
                    state_dict[key] = state_dict[key].float()
            
            model_file = os.path.join(self.model_path, "pytorch_model.bin")
            self.logger.info(f"Saving model to {model_file}")
            torch.save(state_dict, model_file)
            
            self.model.config.save_pretrained(self.model_path)
            self.tokenizer.save_pretrained(self.tokenizer_path)
            
            self.logger.info("Model and tokenizer saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
            
    def optimize_model(self):
        try:
            if hasattr(self.model, 'is_quantized'):
                self.logger.info("Model is already quantized")
                return

            self.model = self.model.cpu()
            
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            
            for param in self.model.parameters():
                param.data = param.data.float()

            self.logger.info("Model quantized successfully")
            
        except Exception as e:
            self.logger.error(f"Error quantizing model: {str(e)}")
            self.logger.warning("Keeping original model without quantization")
