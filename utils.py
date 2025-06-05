import re
import json
import pickle
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class TextProcessor:
    """Enhanced text processing utilities."""
    
    @staticmethod
    def extract_content_between_tags(text: str, start_tag: str, end_tag: str) -> Optional[str]:
        """Extract content between specified tags."""
        pattern = f"{re.escape(start_tag)}(.*?){re.escape(end_tag)}"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None
    
    @staticmethod
    def compute_f1_score(pred: str, gold: str) -> float:
        """Compute F1 score between predicted and gold answers."""
        pred_tokens = set(pred.lower().split())
        gold_tokens = set(gold.lower().split())
        
        if not pred_tokens and not gold_tokens:
            return 1.0
        if not pred_tokens or not gold_tokens:
            return 0.0
            
        intersection = pred_tokens.intersection(gold_tokens)
        precision = len(intersection) / len(pred_tokens)
        recall = len(intersection) / len(gold_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def extract_reasoning_steps(text: str) -> List[str]:
        """Extract individual reasoning steps from text."""
        # Split by common step indicators
        step_patterns = [r'\d+\.', r'Step \d+:', r'First,', r'Second,', r'Then,', r'Finally,']
        
        steps = []
        current_step = ""
        
        for line in text.split('\n'):
            line = line.strip()
            if any(re.match(pattern, line) for pattern in step_patterns):
                if current_step:
                    steps.append(current_step.strip())
                current_step = line
            else:
                current_step += " " + line
        
        if current_step:
            steps.append(current_step.strip())
        
        return steps
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\?\!\,\;\:\-\(\)]', '', text)
        
        return text.strip()
    
    @staticmethod
    def compute_similarity(text1: str, text2: str) -> float:
        """Compute similarity between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    @staticmethod
    def extract_entities(text: str) -> List[str]:
        """Extract potential entities from text."""
        # Simple entity extraction using capitalization patterns
        entities = []
        
        # Find capitalized words that aren't at sentence start
        words = text.split()
        for i, word in enumerate(words):
            # Skip first word of sentences
            if i > 0 and not words[i-1].endswith(('.', '!', '?')):
                if word[0].isupper() and len(word) > 1:
                    entities.append(word)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity not in seen:
                seen.add(entity)
                unique_entities.append(entity)
        
        return unique_entities
    
    @staticmethod
    def count_tag_occurrences(text: str, tag: str) -> int:
        """Count occurrences of a specific tag in text."""
        return text.count(tag)


class IOManager:
    """Input/Output management utilities."""
    
    @staticmethod
    def save_json(data: Any, filepath: str, indent: int = 2):
        """Save data to JSON file."""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=False