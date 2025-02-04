import logging
from pathlib import Path
from bot.chatbot import Chatbot

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('chatbot.log'),
            logging.StreamHandler()
        ]
    )

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        data_dir = Path("data/documents")
        chatbot = Chatbot(str(data_dir))
        
        print("Chatbot is ready! Type 'exit' to quit.")
        while True:
            query = input("You: ").strip()
            if query.lower() == 'exit':
                break
                
            response = chatbot.generate_response(query)
            print(f"Bot: {response['response']}")
            if response['confidence'] < 0.5:
                print(f"(Low confidence: {response['confidence']:.2f})")
                
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise

if __name__ == "__main__":
    main()
