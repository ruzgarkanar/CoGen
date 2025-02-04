from datetime import datetime 
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from .preprocessor import Preprocessor
from ..data.document_loader import DocumentLoader
from ..utils.text_standardizer import TextStandardizer
from ..utils.advanced_text_processor import AdvancedTextProcessor
from ..database.vector_store import VectorStore

class Chatbot:
    def __init__(self, data_dir: str, model_dir: str = None, language: str = 'en'):
        self.logger = logging.getLogger(__name__)
        self.language = language
        self.model_dir = model_dir
        
        self.preprocessor = Preprocessor()
        self.doc_loader = DocumentLoader(data_dir)
        self.text_standardizer = TextStandardizer()
        self.text_processor = AdvancedTextProcessor()
        
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        if model_dir:
            model_path = Path(model_dir) / "model"  
            tokenizer_path = Path(model_dir) / "tokenizer"  
            
            if (model_path / "config.json").exists() and (model_path / "pytorch_model.bin").exists():
                try:
                    self.logger.info(f"Loading QA model from {model_path}")
                    self.qa_model = AutoModelForQuestionAnswering.from_pretrained(str(model_path))
                    
                    self.logger.info(f"Loading tokenizer from {tokenizer_path}")
                    self.qa_tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
                    
                    self.logger.info("Successfully loaded QA model and tokenizer")
                except Exception as e:
                    self.logger.error(f"Error loading QA model: {e}")
                    self.qa_model = None
                    self.qa_tokenizer = None
            else:
                self.logger.warning(f"Required model files not found in {model_path}")
                self.qa_model = None
                self.qa_tokenizer = None
        else:
            self.qa_model = None
            self.qa_tokenizer = None
        
        if model_dir:
            try:
                from ..models.model_manager import ModelManager
                self.model_manager = ModelManager(model_dir)
                self.model_manager.load_model()
                self.logger.info("Model manager initialized successfully")
            except Exception as e:
                self.logger.error(f"Error initializing model manager: {e}")
                self.model_manager = None
        else:
            self.model_manager = None

        self.documents = []
        self.document_embeddings = {}
        self.conversation_history = []
        self.vector_store = VectorStore(str(Path(data_dir) / "vector_store"))
        
        self.load_and_process_documents()

    def load_and_process_documents(self):
        """Load, process and vectorize documents"""
        try:
            raw_documents = self.doc_loader.load_documents()
            if not raw_documents:
                self.logger.error("No documents loaded!")
                return
                
            self.logger.info(f"Loaded {len(raw_documents)} documents")
            self.logger.debug(f"First document sample: {raw_documents[0]}")
            
            processed_documents = []
            embeddings = []
            
            for doc in raw_documents:
                try:
                    if not isinstance(doc, dict):
                        self.logger.warning(f"Invalid document format: {doc}")
                        continue
                        
                    if 'content' not in doc:
                        self.logger.warning(f"No content in document: {doc.keys()}")
                        continue

                    standardized = self.text_standardizer.standardize(doc['content'])
                    processed = self.text_processor.process_text(standardized.text)
                    
                    processed_doc = {
                        'content': processed['cleaned_text'],
                        'original': doc['content'],
                        'metadata': processed['metadata'],
                        'key_phrases': processed['key_phrases'],
                        'summary': processed['summary'],
                        'technical_info': processed['technical_information'],
                        'source': doc.get('source', 'unknown')
                    }
                    
                    embedding = self.encoder.encode(processed['cleaned_text'])
                    
                    processed_documents.append(processed_doc)
                    embeddings.append(embedding)
                    
                except Exception as e:
                    self.logger.error(f"Error processing document: {str(e)}")
                    continue
            
            if not processed_documents:
                self.logger.error("No documents were successfully processed!")
                return

            self.vector_store.add_documents(
                processed_documents,
                np.array(embeddings)
            )
            
            self.documents = processed_documents
            self.logger.info(f"Successfully processed {len(processed_documents)} documents")
            
        except Exception as e:
            self.logger.error(f"Document processing failed: {e}")
            raise

    def generate_response(self, question: str) -> Dict:
        try:
            self.logger.debug(f"Processing question: {question}")
            
            if not self.vector_store:
                self.logger.warning("Vector store not initialized, initializing now...")
                self._initialize_vector_store()
            
            relevant_docs = self.vector_store.search_documents(question, top_k=3)
            self.logger.debug(f"Found {len(relevant_docs)} relevant documents")
            
            if not relevant_docs:
                return {
                    'response': "I apologize, but I couldn't find enough relevant information to answer your question accurately.",
                    'confidence': 0.0,
                    'source': 'system'
                }

            context = " ".join([doc['content'] for doc in relevant_docs])
            
            if self.qa_model and self.qa_tokenizer:  
                inputs = self.qa_tokenizer(
                    question,
                    context,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                )
                
                with torch.no_grad():
                    outputs = self.qa_model(**inputs)
                    
                start_scores = outputs.start_logits
                end_scores = outputs.end_logits
                
                answer_start = torch.argmax(start_scores)
                answer_end = torch.argmax(end_scores) + 1
                
                answer_tokens = inputs["input_ids"][0][answer_start:answer_end]
                answer = self.qa_tokenizer.decode(answer_tokens)
                
                confidence = float(torch.max(torch.softmax(start_scores, dim=-1)) * 
                                torch.max(torch.softmax(end_scores, dim=-1)))
                
                if confidence > 0.5:
                    return {
                        'response': answer,
                        'confidence': confidence,
                        'source': 'model'
                    }
            
            best_doc = relevant_docs[0]
            return {
                'response': self._extract_relevant_sentence(best_doc['content'], question),
                'confidence': float(best_doc.get('score', 0.5)),
                'source': best_doc.get('source', 'unknown')
            }
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return {
                'response': "I apologize, but I encountered an error while processing your question.",
                'confidence': 0.0,
                'source': 'system'
            }

    def _find_relevant_documents(
        self, 
        query_embedding: np.ndarray, 
        query_keyphrases: List[str],
        top_k: int = 3
    ) -> List[Tuple[Dict, float]]:
        """Find most relevant documents using multiple similarity metrics"""
        return self.vector_store.search(
            query_embedding,
            k=top_k,
            threshold=0.3
        )

    def _generate_contextual_response(
        self,
        query: Dict,
        relevant_docs: List[Tuple[Dict, float]],
        conversation_history: List[Dict]
    ) -> Dict:
        """Generate response using context and conversation history"""
        try:
            best_doc, confidence = relevant_docs[0]
            
            relevant_sentences = self._extract_relevant_sentences(
                query['cleaned_text'],
                best_doc['content'],
                num_sentences=2
            )
            
            response = {
                "status": "success",
                "response": " ".join(relevant_sentences),
                "confidence": confidence,
                "source": best_doc['source'],
                "context": {
                    "summary": best_doc['summary'],
                    "technical_info": best_doc.get('technical_info', {}),
                    "similar_docs": [(doc['source'], score) for doc, score in relevant_docs[1:]]
                }
            }
            
            return response

        except Exception as e:
            self.logger.error(f"Contextual response generation failed: {e}")
            return self._create_error_response(str(e))


    def _calculate_embedding_similarity(self, query_embed: np.ndarray, doc_embed: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        return np.dot(query_embed, doc_embed) / (np.linalg.norm(query_embed) * np.linalg.norm(doc_embed))

    def _calculate_keyphrase_similarity(self, query_phrases: List[str], doc_phrases: List[str]) -> float:
        """Calculate similarity based on shared keyphrases"""
        if not query_phrases or not doc_phrases:
            return 0.0
        shared = set(query_phrases) & set(doc_phrases)
        return len(shared) / max(len(query_phrases), len(doc_phrases))

    def _update_conversation_history(self, query: str, response: Dict):
        """Update conversation history"""
        self.conversation_history.append({
            'query': query,
            'response': response,
            'timestamp': datetime.now().isoformat()  
        })
        
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

    def _generate_qa_response(self, question: str, context: str) -> Dict:
        """BERT-QA model ile cevap üret"""
        try:
            if not context or not question:
                self.logger.warning("Empty context or question")
                return self._create_error_response("No valid context found")

            self.qa_model.eval()
            
            inputs = self.qa_tokenizer(
                question,
                context,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=512,
                return_offsets_mapping=True  
            )
            
            offset_mapping = inputs.pop("offset_mapping")
            
            with torch.no_grad():
                outputs = self.qa_model(**inputs)
            
            start_logits = outputs.start_logits[0]
            end_logits = outputs.end_logits[0]
            
            start_idx = torch.argmax(start_logits).item()
            end_idx = torch.argmax(end_logits[start_idx:]) + start_idx
            
            answer = self.qa_tokenizer.convert_tokens_to_string(
                self.qa_tokenizer.convert_ids_to_tokens(
                    inputs["input_ids"][0][start_idx:end_idx + 1]
                )
            )
            
            if not answer or len(answer.split()) < 2:
                sentences = [s.strip() for s in context.split('.') if s.strip()]
                if sentences:
                    relevant_sentence = max(sentences, 
                        key=lambda s: self._calculate_similarity(question.lower(), s.lower())
                    )
                    answer = f"Based on the document, {relevant_sentence}."
                else:
                    return self._create_error_response("Could not generate a valid answer")
            else:
                answer = f"{answer.strip().capitalize()}."
            
            confidence = float(torch.max(torch.softmax(start_logits, dim=0)) * 
                            torch.max(torch.softmax(end_logits, dim=0)))
            
            return {
                "status": "success",
                "response": answer,
                "confidence": confidence,
                "source": "BERT-QA Model"
            }
            
        except Exception as e:
            self.logger.error(f"QA response generation failed: {e}")
            return self._generate_simple_response([({"content": context}, 0.5)])

    def _generate_simple_response(self, relevant_docs: List[Tuple[Dict, float]]) -> Dict:
        """Basit cevap üretme"""
        try:
            best_doc, confidence = relevant_docs[0]
            
            self.logger.debug(f"Document structure: {best_doc}")
            
            if not isinstance(best_doc, dict):
                self.logger.error("Document is not a dictionary")
                return self._create_error_response("Invalid document format")
                
            if 'content' not in best_doc:
                if 'text' in best_doc:
                    content = best_doc['text']
                elif 'original' in best_doc:
                    content = best_doc['original']
                else:
                    self.logger.error(f"No content found in document: {best_doc.keys()}")
                    return self._create_error_response("No content found in document")
            else:
                content = best_doc['content']

            return {
                "status": "success",
                "response": content[:200] + "..." if len(content) > 200 else content,
                "confidence": confidence,
                "source": best_doc.get('source', 'unknown')
            }
            
        except Exception as e:
            self.logger.error(f"Error in _generate_simple_response: {str(e)}")
            return self._create_error_response(str(e))

    def _create_error_response(self, error_message: str) -> Dict:
        """Hata durumunda cevap oluştur"""
        return {
            "status": "error",
            "response": f"Sorry, an error occurred: {error_message}",
            "confidence": 0.0,
            "source": "system"
        }
        
    def _create_no_match_response(self) -> Dict:
        """Eşleşme bulunamadığında cevap oluştur"""
        return {
            "status": "no_match",
            "response": "Sorry, I couldn't find any relevant information for your question.",
            "confidence": 0.0,
            "source": "system"
        }
        
    def _extract_relevant_sentences(self, query: str, context: str, num_sentences: int = 2) -> List[str]:
        """Bağlamdan ilgili cümleleri çıkar"""
        try:
            sentences = context.split('.')
            return sentences[:num_sentences]
        except Exception as e:
            self.logger.error(f"Error extracting sentences: {e}")
            return [context[:200] + "..."]

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """İki metin arasındaki benzerliği hesapla"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0.0

    def _extract_relevant_sentence(self, content: str, question: str) -> str:
        """En alakalı cümleyi seç"""
        try:
            sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 10]
            
            if not sentences:
                return content[:200] + "..."
            
            question_words = set(question.lower().split())
            
            scored_sentences = []
            for sentence in sentences:
                sentence_words = set(sentence.lower().split())
                
                common_words = question_words & sentence_words
                if common_words:  
                    score = len(common_words) / len(question_words)
                    scored_sentences.append((sentence, score))
            
            if scored_sentences:
                scored_sentences.sort(key=lambda x: x[1], reverse=True)
                self.logger.debug(f"Best matching sentence score: {scored_sentences[0][1]}")
                return scored_sentences[0][0]
                
            keyword_matches = {
                'address': lambda s: 'address' in s.lower() or 'location' in s.lower(),
                'product': lambda s: 'product type' in s.lower() or 'product' in s.lower(),
                'company': lambda s: 'industries' in s.lower() or 'inc' in s.lower(),
                'date': lambda s: 'date' in s.lower() or '20' in s,
            }
            
            for keyword, matcher in keyword_matches.items():
                if keyword in question.lower():
                    matching_sentences = [s for s in sentences if matcher(s)]
                    if matching_sentences:
                        return matching_sentences[0]
            
            return self._find_section_with_context(content, question)
            
        except Exception as e:
            self.logger.error(f"Error in _extract_relevant_sentence: {e}")
            return content[:200] + "..."

    def _find_section_with_context(self, content: str, question: str) -> str:
        """Soruyla ilgili bölümü bul ve bağlamıyla birlikte döndür"""
        words = content.split()
        question_words = set(question.lower().split())
        
        window_size = 20
        best_section = ""
        max_score = 0
        
        for i in range(len(words) - window_size):
            section = ' '.join(words[i:i + window_size])
            section_words = set(section.lower().split())
            score = len(question_words & section_words)
            
            if score > max_score:
                max_score = score
                best_section = section
        
        return best_section if best_section else content[:200]
