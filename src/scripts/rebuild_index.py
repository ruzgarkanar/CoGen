import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import logging
from src.bot.chatbot import Chatbot

def rebuild_index():
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    try:
        base_dir = Path(__file__).parent.parent.parent
        data_dir = base_dir / "processed-data"
        vector_store_dir = data_dir / "vector_store"
        
        if vector_store_dir.exists():
            for file in vector_store_dir.glob("*"):
                file.unlink()
            
        chatbot = Chatbot(str(data_dir))
        
        logger.info("Vector store rebuilt successfully")
        
    except Exception as e:
        logger.error(f"Failed to rebuild index: {e}")
        raise

if __name__ == "__main__":
    rebuild_index()
