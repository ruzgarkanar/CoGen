import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from project root .env file
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

class Config:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.settings = self._load_config()
        
    def _load_config(self):
        with open(self.config_path) as f:
            return yaml.safe_load(f)
            
    @property
    def model_settings(self):
        return self.settings.get('model', {})

# Hugging Face settings
HF_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
