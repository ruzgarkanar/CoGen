from ..utils.advanced_text_processor import AdvancedTextProcessor
from typing import List

class Preprocessor:
    def __init__(self):
        self.text_processor = AdvancedTextProcessor()
        
    def preprocess(self, text: str) -> List[str]:
        cleaned_text = self.text_processor.clean_text(text)
        tokens = self.text_processor.tokenize(cleaned_text)
        tokens_without_stopwords = self.text_processor.remove_stopwords(tokens)
        return tokens_without_stopwords
    
    def prepare_response(self, response: str) -> str:
        return response.strip().capitalize()
