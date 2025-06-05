"""
Utility functions and helpers for Enhanced R-Search Framework.
"""

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
                json.dump(data, f, indent=indent, ensure_ascii=False)
            logger.info(f"Data saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save JSON to {filepath}: {e}")
            raise
    
    @staticmethod
    def load_json(filepath: str) -> Any:
        """Load data from JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Data loaded from {filepath}")
            return data
        except Exception as e:
            logger.error(f"Failed to load JSON from {filepath}: {e}")
            raise
    
    @staticmethod
    def save_pickle(data: Any, filepath: str):
        """Save data to pickle file."""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Data saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save pickle to {filepath}: {e}")
            raise
    
    @staticmethod
    def load_pickle(filepath: str) -> Any:
        """Load data from pickle file."""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Data loaded from {filepath}")
            return data
        except Exception as e:
            logger.error(f"Failed to load pickle from {filepath}: {e}")
            raise
    
    @staticmethod
    def ensure_directory(filepath: str):
        """Ensure directory exists for the given filepath."""
        directory = Path(filepath).parent
        directory.mkdir(parents=True, exist_ok=True)


class LoggingManager:
    """Logging configuration and management."""
    
    @staticmethod
    def setup_logging(level: str = "INFO", log_file: str = None):
        """Setup logging configuration."""
        log_level = getattr(logging, level.upper(), logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        root_logger.addHandler(console_handler)
        
        # Setup file handler if specified
        if log_file:
            IOManager.ensure_directory(log_file)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        logger.info(f"Logging setup complete - Level: {level}")


class VisualizationUtils:
    """Utilities for visualizing reasoning processes."""
    
    @staticmethod
    def create_reasoning_tree_diagram(tree_nodes: Dict, output_path: str = None):
        """Create a diagram of the reasoning tree."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            from matplotlib.patches import FancyBboxPatch
        except ImportError:
            logger.warning("Matplotlib not available for visualization")
            return None
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Simple tree layout
        levels = {}
        for node_id, node in tree_nodes.items():
            depth = node.depth
            if depth not in levels:
                levels[depth] = []
            levels[depth].append((node_id, node))
        
        # Draw nodes
        y_positions = {}
        for depth, nodes in levels.items():
            y = len(levels) - depth - 1
            for i, (node_id, node) in enumerate(nodes):
                x = i * 2
                y_positions[node_id] = (x, y)
                
                # Color based on score
                color = 'lightgreen' if node.score > 0.7 else 'lightcoral' if node.score < 0.3 else 'lightblue'
                
                # Draw node
                box = FancyBboxPatch((x-0.4, y-0.2), 0.8, 0.4, 
                                   boxstyle="round,pad=0.02", 
                                   facecolor=color, edgecolor='black')
                ax.add_patch(box)
                
                # Add text
                short_content = node.content[:20] + "..." if len(node.content) > 20 else node.content
                ax.text(x, y, short_content, ha='center', va='center', fontsize=8, wrap=True)
        
        # Draw edges
        for node_id, node in tree_nodes.items():
            if node.parent_id and node.parent_id in y_positions:
                parent_pos = y_positions[node.parent_id]
                child_pos = y_positions[node_id]
                ax.plot([parent_pos[0], child_pos[0]], [parent_pos[1], child_pos[1]], 
                       'k-', alpha=0.6)
        
        ax.set_xlim(-1, max([pos[0] for pos in y_positions.values()]) + 1)
        ax.set_ylim(-0.5, len(levels) - 0.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Reasoning Tree Visualization')
        
        if output_path:
            IOManager.ensure_directory(output_path)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Tree diagram saved to {output_path}")
        
        return fig
    
    @staticmethod
    def plot_reward_trends(reward_history: List[Dict], output_path: str = None):
        """Plot reward trends over time."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Matplotlib not available for visualization")
            return None
        
        if not reward_history:
            logger.warning("No reward history to plot")
            return None
        
        # Extract reward components
        episodes = list(range(len(reward_history)))
        
        reward_types = ['answer_reward', 'evidence_reward', 'consistency_reward', 
                       'novelty_reward', 'efficiency_reward', 'total_reward']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, reward_type in enumerate(reward_types):
            if i < len(axes):
                values = [episode.get(reward_type, 0) for episode in reward_history]
                axes[i].plot(episodes, values, linewidth=2)
                axes[i].set_title(reward_type.replace('_', ' ').title())
                axes[i].set_xlabel('Episode')
                axes[i].set_ylabel('Reward')
                axes[i].grid(True, alpha=0.3)
                
                # Add trend line
                if len(values) > 1:
                    z = np.polyfit(episodes, values, 1)
                    p = np.poly1d(z)
                    axes[i].plot(episodes, p(episodes), "r--", alpha=0.8, linewidth=1)
        
        plt.tight_layout()
        
        if output_path:
            IOManager.ensure_directory(output_path)
            plt.savefi