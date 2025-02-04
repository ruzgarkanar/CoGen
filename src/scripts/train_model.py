import sys
from pathlib import Path

project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import logging
import json
from src.models.model_manager import ModelManager
from src.models.model_trainer import ModelTrainer
from src.data.document_loader import DocumentLoader

def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ) 
    logger = logging.getLogger(__name__)
    
    try:
        base_dir = Path(__file__).parent.parent.parent
        model_dir = base_dir / "models"
        data_dir = base_dir / "processed-data"
        vector_store_dir = data_dir / "vector_store"

        logger.info(f"Checking paths:")
        logger.info(f"Base dir: {base_dir}")
        logger.info(f"Data dir: {data_dir}")
        logger.info(f"Vector store dir: {vector_store_dir}")
        
        if not vector_store_dir.exists():
            logger.error(f"Vector store directory not found. Please run prepare_data.py first")
            return

        metadata_file = vector_store_dir / "metadata.json"
        if not metadata_file.exists():
            logger.error(f"Metadata file not found. Please run prepare_data.py first")
            return
            
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                logger.debug(f"Metadata structure: {json.dumps(list(metadata.keys()), indent=2)}")
        except Exception as e:
            logger.error(f"Failed to read metadata: {e}")
            return

        doc_loader = DocumentLoader(str(data_dir))
        processed_documents = doc_loader.load_documents()
        
        logger.info(f"Loaded {len(processed_documents)} documents")
        if len(processed_documents) == 0:
            logger.error("No documents found in processed-data directory!")
            return
            
        if processed_documents:
            logger.debug(f"First document structure: {json.dumps(processed_documents[0], indent=2)}")
        
        model_manager = ModelManager(str(model_dir))
        model_manager.load_model()
        
        trainer = ModelTrainer(model_manager)
        
        training_data = trainer.prepare_training_data(processed_documents)
        logger.info(f"Generated {len(training_data)} QA pairs")
        
        if len(training_data) == 0:
            logger.error("No training data generated!")
            logger.debug("Document metadata sample:", processed_documents[0].get('metadata', {}))
            return
            
        trainer.train_model(training_data)
        logger.info("Model training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
