import os
import json
import logging
from typing import List, Dict
from pathlib import Path
import PyPDF2

class DocumentLoader:
    def __init__(self, data_dir: str):
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path(data_dir)
        self.vector_store_dir = self.data_dir / "vector_store"
        
    def load_documents(self) -> List[Dict]:
        try:
            # Check vector store metadata
            if self.vector_store_dir.exists():
                metadata_file = self.vector_store_dir / "metadata.json"
                if metadata_file.exists():
                    self.logger.info(f"Loading documents from metadata: {metadata_file}")
                    with open(metadata_file, 'r') as f:
                        try:
                            metadata = json.load(f)
                            # Directly convert numbered dict to list
                            documents = []
                            if isinstance(metadata, dict):
                                # Handle both formats: numbered dict or documents list
                                if 'documents' in metadata:
                                    documents = metadata['documents']
                                else:
                                    documents = [doc for _, doc in metadata.items() 
                                               if isinstance(doc, dict) and 'content' in doc]
                                
                                if documents:
                                    self.logger.info(f"Loaded {len(documents)} documents from metadata")
                                    return documents
                            
                            self.logger.warning("No valid documents in metadata")
                        except json.JSONDecodeError:
                            self.logger.warning("Invalid metadata file")

            # Fallback to PDF loading
            pdf_files = list(self.data_dir.glob("**/*.pdf"))  # Recursive arama
            self.logger.info(f"Found {len(pdf_files)} PDF documents")
            
            documents = []
            for pdf_path in pdf_files:
                try:
                    with open(pdf_path, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        text = ""
                        for page in pdf_reader.pages:
                            text += page.extract_text()
                        
                        document = {
                            'content': text,
                            'metadata': {
                                'source_file': str(pdf_path),
                                'page_count': len(pdf_reader.pages)
                            },
                            'source': pdf_path.name
                        }
                        documents.append(document)
                        
                except Exception as e:
                    self.logger.error(f"Error processing {pdf_path}: {e}")
                    continue
            
            # Yüklenen dokümanları metadata olarak kaydet
            if documents:
                self._save_metadata({'documents': documents})
                
            return documents
            
        except Exception as e:
            self.logger.error(f"Error loading documents: {e}")
            return []

    def _save_metadata(self, metadata: Dict):
        """Metadata'yı vector store dizinine kaydet"""
        try:
            self.vector_store_dir.mkdir(parents=True, exist_ok=True)
            metadata_file = self.vector_store_dir / "metadata.json"
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            self.logger.info(f"Saved metadata to {metadata_file}")
        except Exception as e:
            self.logger.error(f"Error saving metadata: {e}")

    def save_document(self, document: Dict, is_metadata: bool = True):
        try:
            # Vector store dizinini oluştur
            self.vector_store_dir.mkdir(parents=True, exist_ok=True)
            
            if is_metadata:
                filepath = self.vector_store_dir / "metadata.json"
                mode = 'r+' if filepath.exists() else 'w'
                
                with open(filepath, mode, encoding='utf-8') as f:
                    if mode == 'r+':
                        try:
                            data = json.load(f)
                        except json.JSONDecodeError:
                            data = {}
                    else:
                        data = {}
                    
                    data.update(document)
                    
                    f.seek(0)
                    json.dump(data, f, ensure_ascii=False, indent=2)
                    f.truncate()
                    
            else:
                # Diğer JSON dosyaları için
                filepath = self.data_directory / f"{document.get('id', 'doc')}.json"
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(document, f, ensure_ascii=False, indent=2)
                    
            self.logger.info(f"Successfully saved document to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving document: {str(e)}")
            raise
