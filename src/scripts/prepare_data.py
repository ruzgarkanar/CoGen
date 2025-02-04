import sys
from pathlib import Path

project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import logging
from src.utils.document_extractor import DocumentExtractor
from src.utils.text_standardizer import TextStandardizer
from src.utils.text_chunker import TextChunker
from src.database.vector_store import VectorStore
from src.utils.advanced_text_processor import AdvancedTextProcessor
from sentence_transformers import SentenceTransformer

class DataPreparer:
    def __init__(self, input_dir: str, output_dir: str):
        self.logger = logging.getLogger(__name__)
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.doc_extractor = DocumentExtractor()
        self.text_standardizer = TextStandardizer()
        self.text_chunker = TextChunker(chunk_size=512, overlap=50)
        self.text_processor = AdvancedTextProcessor()
        self.vector_store = VectorStore(str(self.output_dir / "vector_store"))
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
    def process_documents(self):
        try:
            self.logger.info("Extracting text from documents...")
            raw_texts = self.doc_extractor.extract_from_directory(
                self.input_dir,
                file_types=['.pdf', '.docx', '.png', '.jpg']
            )
            
            processed_documents = []
            
            for file_path, raw_text in raw_texts.items():
                try:
                    self.logger.debug(f"Processing text from {file_path}")
                    self.logger.debug(f"Raw text length: {len(raw_text)}")
                    self.logger.debug(f"First 100 chars: {raw_text[:100]}")

                    standardized = self.text_standardizer.standardize(raw_text)
                    self.logger.debug(f"Standardized text length: {len(standardized.text)}")
                    
                    processed = self.text_processor.process_text(standardized.text)
                    self.logger.debug(f"Processed text length: {len(processed['cleaned_text'])}")
                    
                    chunks = self.text_chunker.split_into_chunks(processed['cleaned_text'])
                    self.logger.debug(f"Created {len(chunks)} chunks")

                    for chunk in chunks:
                        doc = {
                            'content': chunk['text'],
                            'metadata': {
                                'source_file': str(file_path),
                                'chunk_id': chunk.get('id'),
                                'neighbors': chunk.get('neighbors', {}),
                                'technical_info': processed.get('technical_information', {}),
                                'entities': processed.get('named_entities', []),
                                'key_phrases': chunk.get('metadata', {}).get('key_terms', [])
                            }
                        }
                        processed_documents.append(doc)
                        
                except Exception as e:
                    self.logger.error(f"Error processing {file_path}: {str(e)}", exc_info=True)
                    continue
            
            if not processed_documents:
                self.logger.warning("No documents were successfully processed!")
                return 0

            self.logger.info("Creating vector embeddings...")
            embeddings = self.encoder.encode(
                [doc['content'] for doc in processed_documents],
                batch_size=32,
                show_progress_bar=True
            )
            
            self.logger.info("Saving to vector store...")
            self.vector_store.add_documents(processed_documents, embeddings)
            
            return len(processed_documents)
            
        except Exception as e:
            self.logger.error(f"Document processing failed: {str(e)}", exc_info=True)
            raise

def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        import nltk
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('maxent_ne_chunker')
        nltk.download('words')
    except Exception as e:
        logger.error(f"Failed to download NLTK resources: {e}")
        raise

    try:
        base_dir = Path(__file__).parent.parent.parent
        input_dir = base_dir / "demo-data"
        output_dir = base_dir / "processed-data"
        
        if not input_dir.exists() or not any(input_dir.iterdir()):
            logger.error(f"No documents found in {input_dir}")
            logger.info("Please add some PDF/text documents to the demo-data directory")
            return
            
        preparer = DataPreparer(
            input_dir=str(input_dir),
            output_dir=str(output_dir)
        )
        
        num_docs = preparer.process_documents()
        logger.info(f"Successfully processed {num_docs} document chunks")
        
    except Exception as e:
        logger.error(f"Data preparation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
