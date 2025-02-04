import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import logging
from src.bot.chatbot import Chatbot
from src.scripts.initialize_system import initialize_system

def main():
    logging.basicConfig(
        level=logging.DEBUG,  
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        if not initialize_system():
            logger.error("System initialization failed!")
            return
            
        base_dir = Path(__file__).parent.parent.parent
        data_dir = base_dir / "processed-data"
        model_dir = base_dir / "models"
        
        model_files_exist = (
            (model_dir / "model" / "config.json").exists() and
            (model_dir / "model" / "pytorch_model.bin").exists() and
            (model_dir / "tokenizer" / "tokenizer.json").exists()
        )
        
        if not model_files_exist:
            logger.warning("Model files not found, running without QA capabilities")
            model_dir = None
        
        chatbot = Chatbot(
            data_dir=str(data_dir),
            model_dir=str(model_dir) if model_files_exist else None
        )
        
        logger.info("Chatbot initialized. Type 'quit' to exit.")
        
        while True:
            question = input("\nQA: ")
            if question.lower() in ['quit', 'exit', 'q']:
                break
                
            response = chatbot.generate_response(question)
            
            print("\nResponse:", response['response'])
            print("Confidence:", response['confidence'])
            print("Source:", response['source'])
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
