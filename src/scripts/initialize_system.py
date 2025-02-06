import os
import sys
from pathlib import Path
import asyncio
import shutil

project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import logging
from src.bot.chatbot import Chatbot
from src.data.document_loader import DocumentLoader
from src.database.vector_store import VectorStore
import numpy as np

async def initialize_system():
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    try:
        base_dir = Path(__file__).parent.parent.parent
        data_dir = base_dir / "processed-data"
        vector_store_dir = data_dir / "vector_store"
        demo_data_dir = base_dir / "demo-data"
        
        if vector_store_dir.exists():
            backup_dir = vector_store_dir.parent / "vector_store_backup"
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            shutil.copytree(vector_store_dir, backup_dir)
            logger.info("Created backup of existing vector store")
            
        vector_store_dir.mkdir(parents=True, exist_ok=True)
        
        doc_loader = DocumentLoader(str(demo_data_dir))
        demo_files = list(demo_data_dir.glob('**/*'))
        if not demo_files:
            if (vector_store_dir.parent / "vector_store_backup").exists():
                shutil.rmtree(vector_store_dir)
                shutil.copytree(backup_dir, vector_store_dir)
                logger.info("Restored vector store from backup")
                return True
            else:
                logger.error("No documents found in demo-data directory!")
                return False

        tasks = []
        for file_path in demo_files:
            if file_path.is_file():
                if file_path.suffix.lower() in ['.pdf']:
                    tasks.append(doc_loader._process_pdf(file_path))
                elif file_path.suffix.lower() in ['.txt', '.md']:
                    tasks.append(doc_loader._process_text(file_path))
                elif file_path.suffix.lower() in ['.docx']:
                    tasks.append(doc_loader._process_docx(file_path))
                elif file_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    tasks.append(doc_loader._process_image(file_path))

        documents = [doc for doc in await asyncio.gather(*tasks) if doc is not None]
        
        if not documents:
            logger.error("No documents were successfully processed!")
            if (vector_store_dir.parent / "vector_store_backup").exists():
                shutil.rmtree(vector_store_dir)
                shutil.copytree(backup_dir, vector_store_dir)
                logger.info("Restored vector store from backup")
            return False
            
        logger.info(f"Processed {len(documents)} documents")
        
        vector_store = VectorStore(str(vector_store_dir))
        
        embeddings = []
        for doc in documents:
            doc_embedding = vector_store.encoder.encode(doc['content'])
            embeddings.append(doc_embedding)
            
        vector_store.add_documents(documents, np.array(embeddings))
        logger.info("Documents indexed successfully")
        
        if (vector_store_dir.parent / "vector_store_backup").exists():
            shutil.rmtree(backup_dir)
            
        return True
        
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        if (vector_store_dir.parent / "vector_store_backup").exists():
            shutil.rmtree(vector_store_dir)
            shutil.copytree(backup_dir, vector_store_dir)
            logger.info("Restored vector store from backup due to error")
        raise

if __name__ == "__main__":
    asyncio.run(initialize_system())
