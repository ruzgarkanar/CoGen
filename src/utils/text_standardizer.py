import re
import unicodedata
from typing import Dict, List, Tuple
from langdetect import detect, DetectorFactory
import spacy
from spellchecker import SpellChecker  
import logging
from dataclasses import dataclass
from pathlib import Path

try:
    DetectorFactory().seed = 0 
except Exception as e:
    logging.warning(f"Language detection initialization failed: {e}")

@dataclass
class StandardizationResult:
    text: str
    language: str
    error_regions: List[Tuple[int, int, str]]
    confidence: float
    statistics: Dict

class TextStandardizer:
    def __init__(self, config: Dict = None):
        self.logger = logging.getLogger(__name__)
        
        try:
            self.nlp_models = {
                'en': spacy.load('en_core_web_lg')    
            }
        except OSError:
            import subprocess
            subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_lg'])
            self.nlp_models = {
                'en': spacy.load('en_core_web_lg')
            }
        
        self.spellcheckers = {
            'en': SpellChecker(language='en')
        }
        
        custom_dict = self._load_custom_dictionary('en')
        if custom_dict:
            self.spellcheckers['en'].word_frequency.update(custom_dict)

        self.config = config or self._default_config()
        self.custom_rules = self._load_custom_rules()
        self.abbreviation_map = self._load_abbreviations()
        self.domain_specific_terms = self._load_domain_terms()

        self.technical_patterns = self._compile_technical_patterns()

    def _default_config(self) -> Dict:
        return {
            'min_confidence': 0.7,
            'max_length': 100000,
            'remove_urls': True,
            'fix_encoding': True,
            'normalize_whitespace': True,
            'remove_emojis': True,
            'spell_check': True,
            'min_word_length': 2
        }

    def standardize(self, text: str) -> StandardizationResult:
        try:
            stats = {'original_length': len(text)}
            error_regions = []

            text = self._basic_cleanup(text)
            stats['after_cleanup_length'] = len(text)

            language, lang_confidence = self._detect_language(text)
            stats['detected_language'] = language
            stats['language_confidence'] = lang_confidence

            if self.config['fix_encoding']:
                text = self._fix_encoding(text)

            if language in self.nlp_models:
                text, errors = self._language_specific_processing(text, language)
                error_regions.extend(errors)

            if self.config['spell_check']:
                text, spell_errors = self._check_spelling(text, language)
                error_regions.extend(spell_errors)

            text = self._final_normalization(text)
            
            stats['final_length'] = len(text)
            stats['error_count'] = len(error_regions)

            return StandardizationResult(
                text=text,
                language=language,
                error_regions=error_regions,
                confidence=lang_confidence,
                statistics=stats
            )

        except Exception as e:
            self.logger.error(f"Standardization failed: {str(e)}")
            raise

    def _basic_cleanup(self, text: str) -> str:
        if not text or len(text) > self.config['max_length']:
            raise ValueError("Text is empty or too long")
            
        if self.config['remove_urls']:
            text = re.sub(r'http\S+|www.\S+', '', text)
            
        if self.config['remove_emojis']:
            text = text.encode('ascii', 'ignore').decode('ascii')
            
        text = re.sub(r'\s+', ' ', text)  
        return text.strip()

    def _detect_language(self, text: str) -> Tuple[str, float]:
        try:
            doc = self.nlp_models['en'](text)
            return 'en', 1.0
            
        except Exception as e:
            self.logger.error(f"Language detection failed: {str(e)}")
            return 'en', 0.0

    def _fix_encoding(self, text: str) -> str:
        text = unicodedata.normalize('NFKC', text)
        
        tr_char_map = {
            'İ': 'i', 'I': 'ı',
            'Ğ': 'ğ', 'Ü': 'ü',
            'Ş': 'ş', 'Ö': 'ö',
            'Ç': 'ç'
        }
        
        for old, new in tr_char_map.items():
            text = text.replace(old, new)
            
        return text

    def _language_specific_processing(self, text: str, language: str) -> Tuple[str, List[Tuple[int, int, str]]]:
        errors = []
        nlp = self.nlp_models.get(language)
        
        if not nlp:
            return text, errors
            
        doc = nlp(text)
        processed_text = []
        
        for token in doc:
            if token.is_space or len(token.text) < self.config['min_word_length']:
                continue
                
            if token.is_punct and not self._is_valid_punctuation(token.text):
                errors.append((token.idx, token.idx + len(token.text), "invalid_punct"))
                continue
                
            if hasattr(token._, 'morphology'):
                processed_word = self._apply_morphological_rules(token)
            else:
                processed_word = token.text
                
            processed_text.append(processed_word)
            
        return ' '.join(processed_text), errors

    def _check_spelling(self, text: str, language: str) -> Tuple[str, List[Tuple[int, int, str]]]:
        spellchecker = self.spellcheckers.get(language)
        if not spellchecker:
            return text, []
            
        words = text.split()
        corrected_words = []
        errors = []
        current_pos = 0
        
        for word in words:
            if self._is_technical_term(word) or self._is_special_term(word):
                corrected_words.append(word)
                current_pos += len(word) + 1
                continue
                
            if word.lower() not in spellchecker:
                candidates = spellchecker.candidates(word)
                if candidates:
                    correction = max(candidates, key=lambda x: spellchecker.word_frequency[x.lower()] if x.lower() in spellchecker.word_frequency else 0)
                    corrected_words.append(correction)
                    errors.append((current_pos, current_pos + len(word), "spelling"))
                else:
                    corrected_words.append(word)  
            else:
                corrected_words.append(word)
                
            current_pos += len(word) + 1
            
        return ' '.join(corrected_words), errors

    def _final_normalization(self, text: str) -> str:

        text = re.sub(r'([.,!?])\1+', r'\1', text)
        

        text = re.sub(r'\s*([.,!?])\s*', r'\1 ', text)
        
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        return text.strip()

    def _is_valid_punctuation(self, punct: str) -> bool:
        valid_puncts = {'.', ',', '!', '?', ':', ';', '(', ')', '[', ']', '{', '}', '"', "'"}
        return punct in valid_puncts

    def _calculate_language_scores(self, doc) -> Dict[str, float]:
        word_counts = {}
        total_words = len([token for token in doc if not token.is_space])
        
        if total_words == 0:
            return {'en': 0.0}
            
        for token in doc:
            if token.lang_ in word_counts:
                word_counts[token.lang_] += 1
            else:
                word_counts[token.lang_] = 1
                
        return {lang: count/total_words for lang, count in word_counts.items()}
        
    def _load_custom_rules(self) -> Dict:
        return {
            'technical_terms': r'\b[A-Z0-9]+(?:-[A-Z0-9]+)*\b',
            'measurements': r'\b\d+(?:\.\d+)?\s*(?:mm|cm|m|kg|g|MHz|GB|TB)\b',
            'product_codes': r'\b[A-Z]{2,}-\d{3,}\b'
        }
        
    def _handle_technical_content(self, text: str) -> str:
        for term_type, pattern in self.custom_rules.items():
            matches = re.finditer(pattern, text)            
            for match in matches:
                term = match.group()
                text = text.replace(term, f" {term} ")
        return text
        
    def _normalize_technical_terms(self, text: str) -> str:
        for term, standard_form in self.domain_specific_terms.items():
            text = re.sub(rf'\b{term}\b', standard_form, text, flags=re.IGNORECASE)
        return text

    def _is_technical_term(self, word: str) -> bool:
        try:
            return any(pattern.fullmatch(word) for pattern in self.technical_patterns.values())
        except Exception as e:
            self.logger.error(f"Error in technical term check: {e}")
            return False

    def _is_special_term(self, word: str) -> bool:        
        return word in self.domain_specific_terms

    def _load_custom_dictionary(self, language: str) -> Dict[str, int]:
        try:
            dictionary_path = Path(f"dictionaries/{language}_custom.txt")
            
            if not dictionary_path.exists():
                self.logger.warning(f"No custom dictionary found for language: {language}")
                return {}
                
            custom_dict = {}
            with open(dictionary_path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word:
                        custom_dict[word] = 1
                        
            self.logger.info(f"Loaded {len(custom_dict)} custom words for {language}")
            return custom_dict
            
        except Exception as e:
            self.logger.error(f"Failed to load custom dictionary for {language}: {e}")
            return {}

    def _load_abbreviations(self) -> Dict[str, str]:
        try:
            return {
                "corp.": "corporation",
                "inc.": "incorporated",
                "ltd.": "limited",
                "etc.": "etcetera",
                "e.g.": "for example",
                "i.e.": "that is",
                # Add more as needed
            }
        except Exception as e:
            self.logger.error(f"Failed to load abbreviations: {e}")
            return {}

    def _load_domain_terms(self) -> Dict[str, str]:
        try:
            return {
                "specs": "specifications",
                "config": "configuration",
                "auth": "authentication",
                
                "SaaS": "Software as a Service",
                "API": "Application Programming Interface",
                "UI": "User Interface",
                
            }
        except Exception as e:
            self.logger.error(f"Failed to load domain terms: {e}")
            return {}

    def _compile_technical_patterns(self) -> Dict[str, re.Pattern]:
        try:
            return {
                'technical_terms': re.compile(r'\b[A-Z0-9][-A-Z0-9]*\b'),
                'measurements': re.compile(r'\b\d+(?:\.\d+)?\s*(?:mm|cm|m|kg|g|MHz|GB|TB)\b'),
                'product_codes': re.compile(r'\b[A-Z]{2,}\-\d{3,}\b'),
                'technical_specs': re.compile(r'\b(?:specification|spec|feature)s?\b', re.IGNORECASE),
                'dimensions': re.compile(r'\b\d+(?:\.\d+)?\s*(?:x|\*)\s*\d+(?:\.\d+)?\s*(?:x|\*)?(?:\s*\d+(?:\.\d+)?)?\s*(?:mm|cm|m)?\b'),
                'certifications': re.compile(r'\b(?:ISO|ASTM|EN|DIN|BS)\s*\d+(?:[-:]\d+)*\b'),
                'model_numbers': re.compile(r'\b[A-Z]+\d+(?:[-][A-Z0-9]+)*\b')
            }
        except Exception as e:
            self.logger.error(f"Error compiling technical patterns: {e}")
            return {
                'basic': re.compile(r'\b\w+\b')
            }

    def _is_technical_term(self, word: str) -> bool:
        if not word:
            return False
            
        try:
            word = word.strip()
            for pattern_name, pattern in self.technical_patterns.items():
                try:
                    if pattern.match(word):
                        return True
                except Exception as e:
                    self.logger.warning(f"Pattern '{pattern_name}' matching failed: {e}")
                    continue
            return False
        except Exception as e:
            self.logger.error(f"Technical term check failed for word '{word}': {e}")
            return False