from typing import List, Dict, Optional, Tuple, Set
import spacy
from spacy.tokens import Doc, Span
from textblob import TextBlob
from contractions import fix
import inflect
import re
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
import string
import logging
import syllables  

class AdvancedTextProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        self.nlp = spacy.load('en_core_web_lg') 
        self.lemmatizer = WordNetLemmatizer()
        self.inflect_engine = inflect.engine()
        self.special_chars = set(string.punctuation)
        
        self.initialize_nltk_resources()
        
        self.cache = {}
        self.keywords = self._load_domain_keywords()
        self.technical_patterns = self._compile_technical_patterns()
        self.domain_vectorizer = self._initialize_domain_vectorizer()

    def initialize_nltk_resources(self):
        try:
            import nltk
            required_resources = [
                'punkt',          
                'averaged_perceptron_tagger', 
                'wordnet',        
                'stopwords',     
                'words'          
            ]
            
            for resource in required_resources:
                try:
                    nltk.data.find(f'tokenizers/{resource}')
                except LookupError:
                    nltk.download(resource)
                    
            self.logger.info("Successfully initialized NLTK resources")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize NLTK resources: {e}")
            raise

    def _extract_domain_keywords(self, text: str) -> List[str]:
        keywords = []
        for keyword, weight in self.keywords.items():
            if keyword in text.lower():
                keywords.append(keyword)
        return keywords
    
    

    def process_text(self, text: str, use_cache: bool = True) -> Dict:

        cache_key = hash(text)
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]
            
        text = self._initial_cleanup(text)
        
        doc = self.nlp(text)
        
        processed_data = {
            'original_text': text,
            'cleaned_text': self._advanced_clean(doc),
            'sentences': self._extract_sentences(doc),
            'named_entities': self._extract_entities(doc),
            'key_phrases': self._extract_key_phrases(doc),
            'sentiment': self._analyze_sentiment(text),
            'statistics': self._generate_statistics(doc),
            'metadata': self._extract_metadata(doc),
            'keywords': self._extract_domain_keywords(text),
            'summary': self._generate_summary(text),
            'topics': self._identify_topics(text),
            'relations': self._extract_relations(text),
            'text_structure': self._analyze_structure(text),
            'technical_information': self._extract_technical_information(doc)
        }
        
        if use_cache:
            self.cache[cache_key] = processed_data
            
        return processed_data

    def _initial_cleanup(self, text: str) -> str:

        text = fix(text)
        
        special_chars = {
            '"': '"',  # curly quotes
            '"': '"',
            ''': "'",  # apostrophes
            ''': "'",
            '«': '"',  # angle quotes
            '»': '"',
            '„': '"',  # other quotes
            '‟': '"',
            '‹': "'",  # single angle quotes
            '›': "'",
        }
        
        for special, normal in special_chars.items():
            text = text.replace(special, normal)
        
        text = ' '.join(text.split())
        
        return text
    
    def _normalize_number(self, text: str) -> str:

        try:
            number = float(text.replace(',', ''))
            return f"{number:.2f}".rstrip('0').rstrip('.')
        except ValueError:
            return text

    def _advanced_clean(self, doc: Doc) -> str:
        cleaned_tokens = []
        
        for token in doc:
            if token.is_space or token.is_punct:
                continue
                
            if token.like_num:
                cleaned_tokens.append(self._normalize_number(token.text))
                continue
                
            if token.like_url or token.like_email:
                cleaned_tokens.append(token.text.lower())
                continue
                
            pos = self._get_wordnet_pos(token.pos_)
            if pos:
                lemma = self.lemmatizer.lemmatize(token.text.lower(), pos=pos)
                cleaned_tokens.append(lemma)
            else:
                cleaned_tokens.append(token.text.lower())
                
        return ' '.join(cleaned_tokens)
    
    def _get_root_verb(self, sent) -> Optional[str]:
        for token in sent:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                aux_verbs = [aux.text for aux in token.children if aux.dep_ == "aux"]
                return " ".join(aux_verbs + [token.text]) if aux_verbs else token.text
        return None

    def _get_subjects(self, sent) -> List[str]:
        subjects = []
        for token in sent:
            if "subj" in token.dep_:
                subtree = list(token.subtree)
                start = subtree[0].i
                end = subtree[-1].i + 1
                subjects.append(sent[start:end].text)
        return subjects

    def _get_objects(self, sent) -> List[str]:
        objects = []
        for token in sent:
            if "obj" in token.dep_:
                subtree = list(token.subtree)
                start = subtree[0].i
                end = subtree[-1].i + 1
                objects.append(sent[start:end].text)
        return objects
        


    def _extract_sentences(self, doc: Doc) -> List[Dict]:
        sentences = []
        for sent in doc.sents:
            sent_data = {
                'text': sent.text,
                'root_verb': self._get_root_verb(sent),
                'subjects': self._get_subjects(sent),
                'objects': self._get_objects(sent),
                'sentiment': TextBlob(sent.text).sentiment.polarity
            }
            sentences.append(sent_data)
        return sentences
    
    def _get_entity_context(self, doc: Doc, ent, window: int = 5) -> str:

        start = max(0, ent.start - window)
        end = min(len(doc), ent.end + window)
        
        context_tokens = []
        for i in range(start, end):
            if i < ent.start or i >= ent.end:
                context_tokens.append(doc[i].text)
            else:
                if i == ent.start:
                    context_tokens.append(f"[{ent.text}]")
                    
        return " ".join(context_tokens)

    def _extract_entities(self, doc: Doc) -> Dict[str, List[str]]:
        entities = {}
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append({
                'text': ent.text,
                'start_char': ent.start_char,
                'end_char': ent.end_char,
                'context': self._get_entity_context(doc, ent)
            })
        return entities

    def _extract_key_phrases(self, doc: Doc) -> List[str]:
        key_phrases = []
        seen = set()
        
        for chunk in doc.noun_chunks:
            if chunk.text.lower() not in seen:
                key_phrases.append(chunk.text)
                seen.add(chunk.text.lower())
        
        for token in doc:
            if token.pos_ == "VERB":
                phrase = self._extract_verb_phrase(token)
                if phrase and phrase.lower() not in seen:
                    key_phrases.append(phrase)
                    seen.add(phrase.lower())
                    
        return key_phrases
    
    def _extract_verb_phrase(self, token) -> Optional[str]:
        phrase = [token.text]
        for child in token.children:
            if child.dep_ in ('aux', 'neg', 'advmod'):
                phrase.append(child.text)
        return " ".join(phrase)
    

    def _get_sentiment_assessment(self, polarity: float) -> str:
        if polarity > 0.1:
            return "positive"
        elif polarity < -0.1:
            return "negative"
        else:
            return "neutral"
        
        

    def _analyze_sentiment(self, text: str) -> Dict:
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity,
            'assessment': self._get_sentiment_assessment(blob.sentiment.polarity)
        }
    

    def _get_pos_distribution(self, doc: Doc) -> Dict:
        pos_counts = {token.pos_: 0 for token in doc}
        for token in doc:
            pos_counts[token.pos_] += 1
        return pos_counts
    


    def _calculate_readability(self, doc: Doc) -> Dict:
        try:
            word_syllables = [syllables.estimate(token.text) for token in doc if token.is_alpha]
            total_syllables = sum(word_syllables)
            
            words = len([token for token in doc if token.is_alpha])
            sentences = len(list(doc.sents))
            
            if words == 0 or sentences == 0:
                return {'flesch_kincaid': 0.0}
            
            fk_score = 0.39 * (words / sentences) + 11.8 * (total_syllables / words) - 15.59
            
            return {
                'flesch_kincaid': round(fk_score, 2),
                'total_syllables': total_syllables,
                'avg_syllables_per_word': round(total_syllables / words, 2) if words > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating readability: {e}")
            return {'flesch_kincaid': 0.0}

    def _calculate_vocabulary_complexity(self, doc: Doc) -> float:
        unique_words = len(set(token.text.lower() for token in doc if not token.is_punct))
        return unique_words / len(doc)
    

    def _predict_text_category(self, doc: Doc) -> str:
        noun_count = sum(1 for token in doc if token.pos_ == "NOUN")
        verb_count = sum(1 for token in doc if token.pos_ == "VERB")
        adj_count = sum(1 for token in doc if token.pos_ == "ADJ")
        
        if noun_count > verb_count and noun_count > adj_count:
            return "description"
        elif verb_count > noun_count and verb_count > adj_count:
            return "narrative"
        else:
            return "mixed"
        

    def _generate_statistics(self, doc: Doc) -> Dict:
        return {
            'token_count': len(doc),
            'sentence_count': len(list(doc.sents)),
            'avg_word_length': sum(len(token.text) for token in doc if not token.is_punct) / len(doc),
            'unique_words': len(set(token.text.lower() for token in doc if not token.is_punct)),
            'pos_distribution': self._get_pos_distribution(doc),
            'readability_scores': self._calculate_readability(doc)
        }

    def _extract_metadata(self, doc: Doc) -> Dict:
        return {
            'language': doc.lang_,
            'entities_count': len(doc.ents),
            'vocabulary_complexity': self._calculate_vocabulary_complexity(doc),
            'text_category': self._predict_text_category(doc)
        }

    def _extract_relations(self, text: str) -> List[Dict]:
        relations = []
        doc = self.nlp(text)
        
        for sent in doc.sents:
            for token in sent:
                if token.dep_ in ('nsubj', 'dobj'):
                    relation = {
                        'subject': token.text,
                        'verb': token.head.text,
                        'object': [c.text for c in token.head.children if c.dep_ == 'dobj'],
                        'type': token.dep_
                    }
                    relations.append(relation)
        return relations
    

    def _rank_sentences(self, sentences: List[Span]) -> List[Span]:
        return sorted(sentences, key=lambda x: len(x.text), reverse=True)
    
    def _split_paragraphs(self, text: str) -> List[str]:
        return [para.strip() for para in text.split('\n') if para.strip()]
    

    def _identify_headers(self, text: str) -> List[str]:
        headers = []
        for line in text.split('\n'):
            if line.isupper() and line.strip():
                headers.append(line)
        return headers
    
    def _determine_hierarchy(self, text: str) -> Dict:
        hierarchy = {}
        headers = self._identify_headers(text)
        
        for header in headers:
            level = sum(1 for c in header if c.isupper())
            hierarchy[header] = level
            
        return hierarchy
    

    def _generate_summary(self, text: str) -> str:
        doc = self.nlp(text)
        sentences = list(doc.sents)
        
        important_sentences = self._rank_sentences(sentences)
        return ' '.join([sent.text for sent in important_sentences[:3]])
        
    def _identify_topics(self, text: str) -> List[str]:
        doc = self.nlp(text)
        topics = {}
        
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT', 'TECH']:
                topics[ent.text] = topics.get(ent.text, 0) + 1
                
        return sorted(topics.items(), key=lambda x: x[1], reverse=True)
        
    def _analyze_structure(self, text: str) -> Dict:
        return {
            'paragraphs': self._split_paragraphs(text),
            'section_headers': self._identify_headers(text),
            'hierarchy': self._determine_hierarchy(text)
        }

    def _get_wordnet_pos(self, spacy_pos: str) -> str:
        tag_map = {
            'VERB': wordnet.VERB,
            'NOUN': wordnet.NOUN,
            'ADJ': wordnet.ADJ,
            'ADV': wordnet.ADV
        }
        return tag_map.get(spacy_pos, wordnet.NOUN)

    def _compile_technical_patterns(self) -> Dict[str, re.Pattern]:
        return {
            'measurements': re.compile(r'\b\d+(?:\.\d+)?\s*(?:mm|cm|m|kg|g|MHz|GB|TB)\b'),
            'product_codes': re.compile(r'\b[A-Z]{2,}-\d{3,}\b'),
            'technical_specs': re.compile(r'\b(?:specifications?|specs?|features?)\b', re.IGNORECASE)
        }
        
    def _extract_technical_information(self, doc: Doc) -> Dict:
        technical_info = {
            'measurements': [],
            'specifications': [],
            'product_codes': [],
            'technical_terms': []
        }
        
        for sent in doc.sents:
            if any(pattern.search(sent.text) for pattern in self.technical_patterns.values()):
                self._process_technical_sentence(sent, technical_info)
                
        return technical_info
    

    def _is_technical_term(self, term: str) -> bool:

        return term.lower() in ['system', 'interface', 'processor', 'module', 'protocol', 'algorithm']
    
        
    def _process_technical_sentence(self, sent: Span, info_dict: Dict):

        measurements = self.technical_patterns['measurements'].finditer(sent.text)
        info_dict['measurements'].extend(m.group() for m in measurements)
        
        product_codes = self.technical_patterns['product_codes'].finditer(sent.text)
        info_dict['product_codes'].extend(p.group() for p in product_codes)
        
        for token in sent:
            if token.pos_ == "NOUN" and token.is_alpha and len(token.text) > 2:
                if self._is_technical_term(token.text):
                    info_dict['technical_terms'].append(token.text)

    def _load_domain_keywords(self) -> Dict[str, float]:
        return {
            'technical': 1.0,
            'specification': 0.8,
            'product': 0.7,
            'feature': 0.6,
            'system': 0.5,
        }

    def _initialize_domain_vectorizer(self):

        return None
