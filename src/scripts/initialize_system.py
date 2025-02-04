import os
import sys
from pathlib import Path

project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import logging
from src.bot.chatbot import Chatbot
from src.data.document_loader import DocumentLoader
from src.database.vector_store import VectorStore
import numpy as np
import shutil

def initialize_system():
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    try:
        base_dir = Path(__file__).parent.parent.parent
        data_dir = base_dir / "processed-data"
        vector_store_dir = data_dir / "vector_store"
        demo_data_dir = base_dir / "demo-data"
        
        if vector_store_dir.exists():
            shutil.rmtree(vector_store_dir)
            logger.info("Cleaned existing vector store")
        
        vector_store_dir.mkdir(parents=True, exist_ok=True)
        
        doc_loader = DocumentLoader(str(demo_data_dir))
        documents = doc_loader.load_documents()
        
        if not documents:
            logger.error("No documents were loaded!")
            return False
            
        logger.info(f"Loaded {len(documents)} documents")
        
        for doc in documents:
            if not doc.get('content'):
                logger.error(f"Empty content in document: {doc.get('source', 'unknown')}")
                return False
        
        vector_store = VectorStore(str(vector_store_dir))
        
        embeddings = []
        for doc in documents:
            doc_embedding = vector_store.encoder.encode(doc['content'])
            embeddings.append(doc_embedding)
            logger.debug(f"Created embedding for {doc.get('source', 'unknown')}")
        
        embeddings_array = np.array(embeddings)
        logger.debug(f"Embeddings shape: {embeddings_array.shape}")
        
        if embeddings_array.shape[0] == 0:
            logger.error("No embeddings were created!")
            return False
        
        vector_store.add_documents(documents, embeddings_array)
        logger.info("Documents indexed successfully")
        
        test_query = "What is Mohawk Industries?"
        results = vector_store.search_documents(test_query)
        logger.info(f"Test search found {len(results)} results")
        
        return True
        
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        raise

if __name__ == "__main__":
    initialize_system()
