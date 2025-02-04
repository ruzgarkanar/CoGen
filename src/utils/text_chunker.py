import logging
from typing import List, Dict
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy
import re

class TextChunker:
    def __init__(self, 
                 chunk_size: int = 512,
                 overlap: int = 50,
                 respect_boundaries: bool = True):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.respect_boundaries = respect_boundaries
        self.logger = logging.getLogger(__name__)
        
        try:
            nltk.download('punkt')
            nltk.download('averaged_perceptron_tagger')
            nltk.download('maxent_ne_chunker')
            nltk.download('words')
            
            self._sentence_tokenize = lambda text: re.split(r'(?<=[.!?])\s+', text)
        except Exception as e:
            self.logger.warning(f"NLTK initialization failed: {e}")
            self._sentence_tokenize = lambda text: re.split(r'(?<=[.!?])\s+', text)

        try:
            self.nlp = spacy.load('en_core_web_lg')
        except Exception as e:
            self.logger.warning(f"SpaCy model loading failed: {e}")
            self.nlp = None

    def _create_chunk(self, text: str, start_idx: int, end_idx: int, chunk_id: int) -> Dict:
        return {
            'id': chunk_id,
            'text': text,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'neighbors': {
                'prev': chunk_id - 1 if chunk_id > 0 else None,
                'next': None  
            },
            'metadata': {
                'length': len(text),
                'word_count': len(text.split())
            }
        }

    def split_into_chunks(self, text: str) -> List[Dict]:
        try:
            self.logger.debug(f"Starting to split text of length {len(text)}")
            chunks = []
            
            if not text:
                self.logger.warning("Empty text provided")
                return chunks

            sentences = self._sentence_tokenize(text)
            current_chunk = []
            current_length = 0
            
            for i, sentence in enumerate(sentences):
                if not sentence.strip():
                    continue
                    
                words = sentence.split() 
                sentence_length = len(words)
                
                if current_length + sentence_length <= self.chunk_size:
                    current_chunk.append(sentence)
                    current_length += sentence_length
                else:
                    if current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        chunk = self._create_chunk(
                            chunk_text,
                            start_idx=len(chunks),
                            end_idx=len(chunks) + 1,
                            chunk_id=len(chunks)
                        )
                        chunks.append(chunk)
                    
                    current_chunk = [sentence]
                    current_length = sentence_length

            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunk = self._create_chunk(
                    chunk_text,
                    start_idx=len(chunks),
                    end_idx=len(chunks) + 1,
                    chunk_id=len(chunks)
                )
                chunks.append(chunk)

            for i in range(len(chunks) - 1):
                chunks[i]['neighbors']['next'] = chunks[i + 1]['id']

            self.logger.debug(f"Created {len(chunks)} chunks")
            return chunks

        except Exception as e:
            self.logger.error(f"Error in split_into_chunks: {str(e)}")
            return []

    def _identify_sections(self, doc) -> List[str]:
        sections = []
        current_section = []
        
        for sent in doc.sents:
            current_section.append(sent.text)
            
            if self._is_section_boundary(sent):
                if current_section:
                    sections.append(" ".join(current_section))
                current_section = []
                
        if current_section:
            sections.append(" ".join(current_section))
            
        return sections

    def _chunk_section(self, section: str, doc) -> List[Dict]:
        chunks = []
        words = word_tokenize(section)
        
        start_idx = 0
        while start_idx < len(words):
            end_idx = min(start_idx + self.chunk_size, len(words))
            
            if self.respect_boundaries and end_idx < len(words):
                end_idx = self._find_sentence_boundary(words, end_idx)
            
            chunk_text = " ".join(words[start_idx:end_idx])
            chunks.append({
                'text': chunk_text,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'metadata': self._extract_chunk_metadata(chunk_text)
            })
            
            start_idx = end_idx - self.overlap
            
        semantic_chunks = []
        current_chunk = []
        current_topic = None
        
        for sent in doc.sents:
            sent_topic = self._identify_sentence_topic(sent)
            
            if current_topic and sent_topic != current_topic and len(current_chunk) > 0:
                semantic_chunks.append(self._create_chunk(current_chunk))
                current_chunk = []
            
            current_topic = sent_topic
            current_chunk.append(sent)
            
        if current_chunk:
            semantic_chunks.append(self._create_chunk(current_chunk))
            
        return semantic_chunks

    def _is_section_boundary(self, sent) -> bool:

        return (
            sent.text.isupper() or
            len(sent) < 5 or
            sent.text.endswith(':') or
            any(token.is_title for token in sent)
        )

    def _find_sentence_boundary(self, words: List[str], approx_idx: int) -> int:
        text = " ".join(words[max(0, approx_idx-20):min(approx_idx+20, len(words))])
        sentences = sent_tokenize(text)
        
        boundary = approx_idx
        for sent in sentences:
            sent_words = word_tokenize(sent)
            if len(sent_words) + boundary >= approx_idx:
                return boundary + len(sent_words)
            boundary += len(sent_words)
            
        return approx_idx

    def _extract_chunk_metadata(self, text: str) -> Dict:
        doc = self.nlp(text)
        return {
            'entities': [ent.text for ent in doc.ents],
            'key_terms': [token.text for token in doc if token.pos_ in ('NOUN', 'PROPN')],
            'length': len(doc)
        }

    def _identify_sentence_topic(self, sent) -> str:

        entities = [ent.label_ for ent in sent.ents]
        if entities:
            return entities[0]
        
        important_words = [token.text for token in sent 
                         if token.pos_ in ('NOUN', 'PROPN') 
                         and not token.is_stop]
        if important_words:
            return important_words[0]
            
        return None
