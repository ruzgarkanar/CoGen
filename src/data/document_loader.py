import os
import json
import logging
from typing import List, Dict
from pathlib import Path
import PyPDF2
import aiohttp
import aiofiles
import asyncio
from pathlib import Path
from typing import List, Dict, Optional
import logging
from concurrent.futures import ThreadPoolExecutor
import pytesseract
from pdf2image import convert_from_path
from docx import Document
from PIL import Image
import io

class DocumentLoader:
    def __init__(self, data_dir: str):
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path(data_dir)
        self.vector_store_dir = self.data_dir / "vector_store"
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def load_documents(self) -> List[Dict]:
        try:
            metadata_file = self.vector_store_dir / "metadata.json"
            if not metadata_file.exists():
                self.logger.error(f"Metadata file not found at {metadata_file}")
                return []

            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            documents = []
            for doc_id, doc_data in metadata.items():
                if isinstance(doc_data, dict) and 'content' in doc_data:
                    documents.append(doc_data)

            self.logger.info(f"Loaded {len(documents)} documents successfully")
            return documents

        except Exception as e:
            self.logger.error(f"Error loading documents: {e}")
            return []

    async def _process_pdf(self, file_path: Path) -> Optional[Dict]:
        try:
            loop = asyncio.get_event_loop()
            images = await loop.run_in_executor(
                self.executor,
                convert_from_path,
                str(file_path)
            )
            
            text_parts = []
            for image in images:
                text = await loop.run_in_executor(
                    self.executor,
                    pytesseract.image_to_string,
                    image
                )
                text_parts.append(text)

            return {
                'content': '\n'.join(text_parts),
                'source': str(file_path),
                'type': 'pdf'
            }
        except Exception as e:
            self.logger.error(f"PDF processing error for {file_path}: {e}")
            return None

    async def _process_text(self, file_path: Path) -> Optional[Dict]:
        try:
            async with aiofiles.open(file_path, mode='r', encoding='utf-8') as f:
                content = await f.read()
                
            return {
                'content': content,
                'source': str(file_path),
                'type': 'text'
            }
        except Exception as e:
            self.logger.error(f"Text processing error for {file_path}: {e}")
            return None

    async def _process_docx(self, file_path: Path) -> Optional[Dict]:
        try:
            loop = asyncio.get_event_loop()
            
            def read_docx():
                doc = Document(str(file_path))
                return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
            
            content = await loop.run_in_executor(self.executor, read_docx)
            
            return {
                'content': content,
                'source': str(file_path),
                'type': 'docx'
            }
        except Exception as e:
            self.logger.error(f"DOCX processing error for {file_path}: {e}")
            return None

    async def _process_image(self, file_path: Path) -> Optional[Dict]:
        try:
            loop = asyncio.get_event_loop()
            
            def extract_text_from_image():
                with Image.open(file_path) as img:
                    return pytesseract.image_to_string(img)
            
            text = await loop.run_in_executor(self.executor, extract_text_from_image)
            
            return {
                'content': text,
                'source': str(file_path),
                'type': 'image'
            }
        except Exception as e:
            self.logger.error(f"Image processing error for {file_path}: {e}")
            return None

    def _save_metadata(self, metadata: Dict):
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
                filepath = self.data_dir / f"{document.get('id', 'doc')}.json"
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(document, f, ensure_ascii=False, indent=2)
                    
            self.logger.info(f"Successfully saved document to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving document: {str(e)}")
            raise
