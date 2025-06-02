"""
PII Masking Module
Handles detection and masking of Personally Identifiable Information
"""

import re
import spacy
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class PIIMasker:
    def __init__(self):
        """Initialize PII Masker with NER model and regex patterns"""
        self.nlp = None
        self._load_spacy_model()
        self._compile_regex_patterns()
    
    def _load_spacy_model(self):
        """Load spaCy NER model"""
        try:
            # Try to load the English model
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy English model")
        except IOError:
            logger.warning("spaCy English model not found. Installing...")
            try:
                # Try to download and load
                spacy.cli.download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Downloaded and loaded spaCy English model")
            except Exception as e:
                logger.error(f"Failed to load spaCy model: {str(e)}")
                self.nlp = None
    
    def _compile_regex_patterns(self):
        """Compile regex patterns for PII detection"""
        self.patterns = {
            'email': re.compile(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                re.IGNORECASE
            ),
            'phone_number': re.compile(
                r'(?:\+?\d{1,3}[-.\s]?)?\(?(?:\d{2,4})\)?[-.\s]?\d{2,4}[-.\s]?\d{2,4}[-.\s]?\d{2,4}',
                re.IGNORECASE
            ),
            'credit_debit_no': re.compile(
                r'\b(?:\d{4}[-\s]?){3,4}\d{1,4}\b'
            ),
            'cvv_no': re.compile(
                r'\b(?:cvv|cvc|security\s+code)[\s:]*(\d{3,4})\b',
                re.IGNORECASE
            ),
            'expiry_no': re.compile(
                r'\b(?:exp|expiry|expires?)[\s:]*(\d{1,2}[/\-]\d{2,4}|\d{4})\b',
                re.IGNORECASE
            ),
            'aadhar_num': re.compile(
                r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
            ),
            'dob': re.compile(
                r'\b(?:\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|\d{4}[/\-]\d{1,2}[/\-]\d{1,2}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})\b',
                re.IGNORECASE
            )
        }
    
    def _extract_names_with_spacy(self, text: str) -> List[Dict[str, Any]]:
        """Extract person names using spaCy NER"""
        entities = []
        
        if self.nlp is None:
            return entities
        
        try:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    entities.append({
                        'text': ent.text,
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'label': 'full_name'
                    })
        except Exception as e:
            logger.error(f"Error in spaCy NER: {str(e)}")
        
        return entities
    
    def _extract_names_with_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Extract names using pattern matching"""
        entities = []
        
        # Pattern for "My name is [Name]" or "I am [Name]"
        name_patterns = [
            re.compile(r'(?:my name is|i am|i\'m)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', re.IGNORECASE),
            re.compile(r'([A-Z][a-z]+\s+[A-Z][a-z]+)(?:\s|$)', re.IGNORECASE)  # Two capitalized words
        ]
        
        for pattern in name_patterns:
            for match in pattern.finditer(text):
                if hasattr(match, 'group') and len(match.groups()) > 0:
                    name = match.group(1)
                else:
                    name = match.group(0)
                
                # Filter out common false positives
                if not any(word.lower() in ['subject', 'dear', 'hello', 'hi', 'regards', 'thanks', 'best'] 
                          for word in name.split()):
                    entities.append({
                        'text': name,
                        'start': match.start(1) if len(match.groups()) > 0 else match.start(),
                        'end': match.end(1) if len(match.groups()) > 0 else match.end(),
                        'label': 'full_name'
                    })
        
        return entities
    
    def _extract_with_regex(self, text: str, pattern_name: str) -> List[Dict[str, Any]]:
        """Extract entities using regex patterns"""
        entities = []
        pattern = self.patterns.get(pattern_name)
        
        if pattern is None:
            return entities
        
        for match in pattern.finditer(text):
            entities.append({
                'text': match.group(0),
                'start': match.start(),
                'end': match.end(),
                'label': pattern_name
            })
        
        return entities
    
    def detect_pii(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect all PII in the given text
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of detected PII entities with positions and types
        """
        all_entities = []
        
        # Extract names using spaCy
        name_entities = self._extract_names_with_spacy(text)
        all_entities.extend(name_entities)
        
        # Extract names using patterns (fallback)
        if not name_entities:  # Only use pattern matching if spaCy didn't find names
            pattern_names = self._extract_names_with_patterns(text)
            all_entities.extend(pattern_names)
        
        # Extract other PII types using regex
        pii_types = ['email', 'phone_number', 'credit_debit_no', 'cvv_no', 'expiry_no', 'aadhar_num', 'dob']
        
        for pii_type in pii_types:
            entities = self._extract_with_regex(text, pii_type)
            all_entities.extend(entities)
        
        # Remove overlapping entities (keep the longest one)
        all_entities = self._remove_overlapping_entities(all_entities)
        
        # Sort by position
        all_entities.sort(key=lambda x: x['start'])
        
        return all_entities
    
    def _remove_overlapping_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove overlapping entities, keeping the longest ones"""
        if not entities:
            return entities
        
        # Sort by start position, then by length (descending)
        entities.sort(key=lambda x: (x['start'], -(x['end'] - x['start'])))
        
        non_overlapping = []
        for entity in entities:
            # Check if this entity overlaps with any previously added entity
            overlaps = False
            for existing in non_overlapping:
                if (entity['start'] < existing['end'] and entity['end'] > existing['start']):
                    overlaps = True
                    break
            
            if not overlaps:
                non_overlapping.append(entity)
        
        return non_overlapping
    
    def mask_pii(self, text: str) -> Dict[str, Any]:
        """
        Mask PII in the given text
        
        Args:
            text: Input text to mask
            
        Returns:
            Dictionary containing masked text and entity information
        """
        entities = self.detect_pii(text)
        
        # Create masked text by replacing entities with placeholders
        masked_text = text
        offset = 0
        
        for entity in entities:
            placeholder = f"[{entity['label']}]"
            start = entity['start'] + offset
            end = entity['end'] + offset
            
            masked_text = masked_text[:start] + placeholder + masked_text[end:]
            offset += len(placeholder) - (entity['end'] - entity['start'])
        
        return {
            'original_text': text,
            'masked_text': masked_text,
            'entities': entities
        }
    
    def unmask_pii(self, masked_text: str, entities: List[Dict[str, Any]]) -> str:
        """
        Restore original PII in masked text
        
        Args:
            masked_text: Text with PII placeholders
            entities: List of original entities with their information
            
        Returns:
            Text with original PII restored
        """
        restored_text = masked_text
        
        # Sort entities by start position in reverse order to maintain positions
        entities_sorted = sorted(entities, key=lambda x: x['start'], reverse=True)
        
        for entity in entities_sorted:
            placeholder = f"[{entity['label']}]"
            # Find the placeholder in the masked text and replace with original
            restored_text = restored_text.replace(placeholder, entity['text'], 1)
        
        return restored_text
