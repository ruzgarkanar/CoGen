from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from pathlib import Path
import faiss
import pickle
import json
import logging
from datetime import datetime  
from sentence_transformers import SentenceTransformer 
from ..config.settings import HF_TOKEN

class VectorStore:
    def __init__(self, index_path: str = "data/vector_index"):
        self.logger = logging.getLogger(__name__)
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        self.dimension = 384 
        self.index = None
        
        self.metadata = {}
        self.id_to_metadata = {}
        
        self._initialize_index()
        
        self.cache = {}
        self.batch_size = 1000
        
        self.encoder = SentenceTransformer(
            'sentence-transformers/all-MiniLM-L6-v2',
            token=HF_TOKEN
        )
        
    def _initialize_index(self):
        try:
            index_file = self.index_path / "faiss_index.bin"
            if index_file.exists():
                self.index = faiss.read_index(str(index_file))
                self._load_metadata()
            else:
                self.index = faiss.IndexFlatL2(self.dimension)
                self._save_index()
        except Exception as e:
            self.logger.error(f"Failed to initialize index: {e}")
            raise

    def add_documents(self, documents: List[Dict], embeddings: np.ndarray):
        try:
            if len(embeddings.shape) != 2:
                embeddings = embeddings.reshape(-1, self.dimension)
                
            if len(documents) != embeddings.shape[0]:
                raise ValueError(f"Number of documents ({len(documents)}) and embeddings ({embeddings.shape[0]}) must match")
            
            if embeddings.shape[1] != self.dimension:
                raise ValueError(f"Embedding dimension mismatch. Expected {self.dimension}, got {embeddings.shape[1]}")
            
            start_id = self.index.ntotal
            
            self.index.add(embeddings.astype('float32'))
            
            self.logger.debug(f"Adding embeddings with shape: {embeddings.shape}")
            
            for idx, doc in enumerate(documents):
                doc_id = str(start_id + idx)
                self.id_to_metadata[doc_id] = {
                    'content': doc['content'],
                    'metadata': doc.get('metadata', {}),
                    'source': doc.get('source', 'unknown'),
                    'timestamp': datetime.now().isoformat()
                }
            
            self._save_index()
            self._save_metadata()
            
            self.logger.info(f"Added {len(documents)} documents. New total: {self.index.ntotal}")
            
        except Exception as e:
            self.logger.error(f"Failed to add documents: {str(e)}")
            raise

    def _batch_update_metadata(self, documents: List[Dict], start_id: int):
        metadata_updates = {}
        for idx, doc in enumerate(documents):
            doc_id = start_id + idx
            metadata_updates[str(doc_id)] = { 
                'content': doc['content'],
                'metadata': doc.get('metadata', {}),
                'source': doc.get('source', 'unknown'),
                'timestamp': datetime.now().isoformat()
            }
        
        self.id_to_metadata.update(metadata_updates)

    def search(self, query_vector: np.ndarray, k: int = 5, threshold: float = 0.1) -> List[Tuple[Dict, float]]:
        try:
            if len(query_vector.shape) == 1:
                query_vector = query_vector.reshape(1, -1)
                
            distances, indices = self.index.search(query_vector, k)
            
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx != -1:  
                    doc = self.id_to_metadata.get(str(idx))
                    if doc and isinstance(doc, dict) and doc.get('content'):  
                        similarity = 1 / (1 + distance)
                        self.logger.debug(f"Document {idx} similarity: {similarity}")
                        if similarity >= threshold:
                            self.logger.debug(f"Found document with similarity {similarity}")
                            doc_copy = {
                                'content': doc['content'],
                                'metadata': doc.get('metadata', {}),
                                'source': doc.get('source', 'unknown'),
                                'summary': doc.get('metadata', {}).get('summary', ''),
                                'technical_info': doc.get('metadata', {}).get('technical_info', {})
                            }
                            results.append((doc_copy, similarity))
            
            self.logger.info(f"Found {len(results)} relevant documents")
            if not results:
                self.logger.warning("No valid documents found")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []

    def batch_search(
        self, 
        query_vectors: np.ndarray, 
        k: int = 5
    ) -> List[List[Tuple[Dict, float]]]:
        try:
            distances, indices = self.index.search(query_vectors, k)
            
            results = []
            for batch_idx in range(len(query_vectors)):
                batch_results = []
                for idx, distance in zip(indices[batch_idx], distances[batch_idx]):
                    if idx != -1:
                        similarity = 1 / (1 + distance)
                        metadata = self.id_to_metadata.get(int(idx), {})
                        batch_results.append((metadata, float(similarity)))
                results.append(batch_results)
                
            return results
            
        except Exception as e:
            self.logger.error(f"Batch search failed: {e}")
            raise

    def _save_index(self):
        index_file = self.index_path / "faiss_index.bin"
        faiss.write_index(self.index, str(index_file))

    def _save_metadata(self):
        metadata_file = self.index_path / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:  
            json.dump(self.id_to_metadata, f, ensure_ascii=False, indent=2)

    def _load_metadata(self):
        metadata_file = self.index_path / "metadata.json"
        if (metadata_file.exists()):
            with open(metadata_file, 'r', encoding='utf-8') as f:  
                self.id_to_metadata = json.load(f)

    def get_stats(self) -> Dict[str, Any]:
        return {
            'total_documents': self.index.ntotal,
            'dimension': self.dimension,
            'index_type': type(self.index).__name__,
            'metadata_count': len(self.id_to_metadata)
        }

    def clear(self):
        self.index = faiss.IndexFlatL2(self.dimension)
        self.id_to_metadata = {}
        self._save_index()
        self._save_metadata()

    def search_documents(self, query: str, top_k: int = 3) -> List[Dict]:
        try:
            self.logger.debug(f"Searching for query: {query}")
            self.logger.debug(f"Index size: {self.index.ntotal if self.index else 0}")
            self.logger.debug(f"Metadata count: {len(self.id_to_metadata)}")
            
            if not self.index or self.index.ntotal == 0:
                self.logger.warning("Empty vector store!")
                return []
            
            query_vector = self.encoder.encode([query], show_progress_bar=False)[0]
            
            distances, indices = self.index.search(
                query_vector.reshape(1, -1).astype('float32'), 
                min(top_k, self.index.ntotal) 
            )
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1:  
                    doc_id = str(idx)
                    self.logger.debug(f"Checking document {doc_id}")
                    
                    if doc_id in self.id_to_metadata: 
                        doc = self.id_to_metadata[doc_id].copy()  
                        similarity_score = float(1 / (1 + distances[0][i]))
                        doc['score'] = similarity_score
                        
                        self.logger.debug(f"Found document: score={similarity_score:.4f}, "
                                        f"content_length={len(doc.get('content', ''))}")
                        results.append(doc)
            
            self.logger.info(f"Found {len(results)} relevant documents")
            return results
            
        except Exception as e:
            self.logger.error(f"Search error: {str(e)}")
            return []
