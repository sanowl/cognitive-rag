from typing import Dict, List, Optional
from collections import defaultdict
from .config import EpisodicMemory, EnhancedRSearchConfig


class EpisodicMemoryManager:
    """Manages episodic memory for learning from past experiences."""
    
    def __init__(self, config: EnhancedRSearchConfig):
        self.config = config
        self.memories: List[EpisodicMemory] = []
        self.memory_index: Dict[str, List[int]] = defaultdict(list)
        self.pattern_cache: Dict[str, float] = {}
    
    def store_episode(self, question: str, reasoning_pattern: str, tools_used: List[str], 
                     success_score: float, strategies: List[str]):
        """Store a new episodic memory."""
        question_type = self._classify_question_type(question)
        
        memory = EpisodicMemory(
            question_type=question_type,
            reasoning_pattern=reasoning_pattern,
            successful_tools=tools_used,
            effective_strategies=strategies,
            common_pitfalls=[],
            success_score=success_score
        )
        
        # Check for similar existing memories
        existing_idx = self._find_similar_memory(memory)
        if existing_idx is not None:
            # Update existing memory
            existing = self.memories[existing_idx]
            existing.frequency += 1
            existing.success_score = (existing.success_score + success_score) / 2
            # Merge strategies
            existing.effective_strategies = list(set(existing.effective_strategies + strategies))
            existing.successful_tools = list(set(existing.successful_tools + tools_used))
        else:
            # Add new memory
            if len(self.memories) >= self.config.episodic_memory_size:
                self._remove_oldest_memory()
            
            memory_idx = len(self.memories)
            self.memories.append(memory)
            self.memory_index[question_type].append(memory_idx)
        
        # Clear pattern cache when new memory is added
        self.pattern_cache.clear()
    
    def retrieve_relevant_memories(self, question: str, top_k: int = None) -> List[EpisodicMemory]:
        """Retrieve relevant memories for a given question."""
        if top_k is None:
            top_k = self.config.memory_retrieval_top_k
        
        question_type = self._classify_question_type(question)
        relevant_indices = self.memory_index.get(question_type, [])
        
        # Also look for memories from similar question types
        similar_types = self._get_similar_question_types(question_type)
        for sim_type in similar_types:
            relevant_indices.extend(self.memory_index.get(sim_type, []))
        
        # Remove duplicates
        relevant_indices = list(set(relevant_indices))
        
        # Score memories by relevance and recency
        scored_memories = []
        for idx in relevant_indices:
            if idx < len(self.memories):
                memory = self.memories[idx]
                # Combine success score with frequency and recency
                relevance_score = (
                    memory.success_score * 0.6 + 
                    min(memory.frequency / 10, 0.3) * 0.3 +
                    self._recency_score(idx) * 0.1
                )
                scored_memories.append((relevance_score, memory))
        
        # Sort by score and return top_k
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [memory for _, memory in scored_memories[:top_k]]
    
    def get_pattern_statistics(self) -> Dict[str, Dict]:
        """Get statistics about reasoning patterns."""
        pattern_stats = defaultdict(lambda: {
            'count': 0,
            'avg_success': 0.0,
            'tools_used': set(),
            'strategies': set()
        })
        
        for memory in self.memories:
            pattern = memory.reasoning_pattern
            stats = pattern_stats[pattern]
            
            stats['count'] += memory.frequency
            stats['avg_success'] = (stats['avg_success'] * (stats['count'] - memory.frequency) + 
                                  memory.success_score * memory.frequency) / stats['count']
            stats['tools_used'].update(memory.successful_tools)
            stats['strategies'].update(memory.effective_strategies)
        
        # Convert sets to lists for JSON serialization
        for pattern in pattern_stats:
            pattern_stats[pattern]['tools_used'] = list(pattern_stats[pattern]['tools_used'])
            pattern_stats[pattern]['strategies'] = list(pattern_stats[pattern]['strategies'])
        
        return dict(pattern_stats)
    
    def suggest_approach(self, question: str) -> Dict[str, any]:
        """Suggest reasoning approach based on past experiences."""
        relevant_memories = self.retrieve_relevant_memories(question, top_k=3)
        
        if not relevant_memories:
            return {
                "suggested_tools": ["search"],
                "suggested_strategies": ["decomposition"],
                "confidence": 0.5,
                "reasoning": "No similar past experiences found, using default approach"
            }
        
        # Aggregate suggestions from relevant memories
        tool_votes = defaultdict(int)
        strategy_votes = defaultdict(int)
        total_success = 0
        
        for memory in relevant_memories:
            weight = memory.success_score * memory.frequency
            
            for tool in memory.successful_tools:
                tool_votes[tool] += weight
            
            for strategy in memory.effective_strategies:
                strategy_votes[strategy] += weight
            
            total_success += memory.success_score
        
        # Get top suggestions
        suggested_tools = sorted(tool_votes.keys(), key=lambda x: tool_votes[x], reverse=True)[:3]
        suggested_strategies = sorted(strategy_votes.keys(), key=lambda x: strategy_votes[x], reverse=True)[:2]
        
        confidence = min(total_success / len(relevant_memories), 1.0) if relevant_memories else 0.5
        
        return {
            "suggested_tools": suggested_tools,
            "suggested_strategies": suggested_strategies,
            "confidence": confidence,
            "reasoning": f"Based on {len(relevant_memories)} similar past experiences",
            "relevant_memories": len(relevant_memories)
        }
    
    def learn_from_failure(self, question: str, failed_approach: Dict, error_info: str):
        """Learn from failed reasoning attempts."""
        question_type = self._classify_question_type(question)
        
        # Find memories with similar approaches
        for memory in self.memories:
            if (memory.question_type == question_type and 
                any(tool in memory.successful_tools for tool in failed_approach.get('tools_used', []))):
                
                # Add to common pitfalls
                pitfall = f"Failed approach: {error_info}"
                if pitfall not in memory.common_pitfalls:
                    memory.common_pitfalls.append(pitfall)
                
                # Slightly reduce success score
                memory.success_score = max(0.1, memory.success_score * 0.95)
    
    def _classify_question_type(self, question: str) -> str:
        """Classify question into type categories."""
        question_lower = question.lower()
        
        # Multi-word patterns first
        if any(phrase in question_lower for phrase in ["how many", "how much", "what percentage"]):
            return "quantitative"
        elif any(phrase in question_lower for phrase in ["compare", "difference between", "versus", "vs"]):
            return "comparison"
        
        # Single word patterns
        if any(word in question_lower for word in ["when", "date", "year", "time"]):
            return "temporal"
        elif any(word in question_lower for word in ["who", "person", "people"]):
            return "entity_person"
        elif any(word in question_lower for word in ["where", "location", "place"]):
            return "location"
        elif any(word in question_lower for word in ["what", "which", "describe"]):
            return "factual"
        elif any(word in question_lower for word in ["why", "because", "reason"]):
            return "causal"
        elif any(word in question_lower for word in ["how", "method", "process"]):
            return "procedural"
        else:
            return "general"
    
    def _get_similar_question_types(self, question_type: str) -> List[str]:
        """Get question types similar to the given type."""
        similarity_map = {
            "temporal": ["factual", "entity_person"],
            "entity_person": ["factual", "temporal"],
            "location": ["factual", "entity_person"],
            "factual": ["entity_person", "location"],
            "causal": ["procedural", "factual"],
            "procedural": ["causal", "factual"],
            "quantitative": ["factual", "comparison"],
            "comparison": ["quantitative", "factual"],
            "general": ["factual"]
        }
        return similarity_map.get(question_type, ["factual"])
    
    def _find_similar_memory(self, new_memory: EpisodicMemory) -> Optional[int]:
        """Find similar existing memory."""
        for i, memory in enumerate(self.memories):
            if (memory.question_type == new_memory.question_type and
                memory.reasoning_pattern == new_memory.reasoning_pattern):
                return i
        return None
    
    def _remove_oldest_memory(self):
        """Remove the oldest memory to make space."""
        if self.memories:
            # Remove oldest memory (first in list)
            oldest_memory = self.memories.pop(0)
            
            # Update indices in memory_index
            for question_type, indices in self.memory_index.items():
                # Decrease all indices by 1 and remove index 0
                updated_indices = [i-1 for i in indices if i > 0]
                self.memory_index[question_type] = updated_indices
    
    def _recency_score(self, memory_idx: int) -> float:
        """Calculate recency score for a memory."""
        position_from_end = len(self.memories) - memory_idx
        return 1.0 / (1.0 + position_from_end * 0.1)
    
    def export_memories(self) -> List[Dict]:
        """Export memories to a serializable format."""
        exported_memories = []
        for memory in self.memories:
            exported_memories.append({
                "question_type": memory.question_type,
                "reasoning_pattern": memory.reasoning_pattern,
                "successful_tools": memory.successful_tools,
                "effective_strategies": memory.effective_strategies,
                "common_pitfalls": memory.common_pitfalls,
                "success_score": memory.success_score,
                "frequency": memory.frequency
            })
        return exported_memories
    
    def import_memories(self, memory_data: List[Dict]):
        """Import memories from serialized format."""
        self.memories.clear()
        self.memory_index.clear()
        
        for data in memory_data:
            memory = EpisodicMemory(
                question_type=data["question_type"],
                reasoning_pattern=data["reasoning_pattern"],
                successful_tools=data["successful_tools"],
                effective_strategies=data["effective_strategies"],
                common_pitfalls=data["common_pitfalls"],
                success_score=data["success_score"],
                frequency=data["frequency"]
            )
            
            memory_idx = len(self.memories)
            self.memories.append(memory)
            self.memory_index[memory.question_type].append(memory_idx)


class WorkingMemory:
    """Working memory for maintaining context during reasoning."""
    
    def __init__(self, max_capacity: int = 10):
        self.max_capacity = max_capacity
        self.current_context: Dict[str, any] = {}
        self.recent_facts: List[str] = []
        self.active_hypotheses: List[str] = []
        self.confidence_scores: Dict[str, float] = {}
    
    def add_fact(self, fact: str, confidence: float = 1.0):
        """Add a fact to working memory."""
        self.recent_facts.append(fact)
        self.confidence_scores[fact] = confidence
        
        # Maintain capacity limit
        if len(self.recent_facts) > self.max_capacity:
            removed_fact = self.recent_facts.pop(0)
            if removed_fact in self.confidence_scores:
                del self.confidence_scores[removed_fact]
    
    def add_hypothesis(self, hypothesis: str):
        """Add a hypothesis to working memory."""
        if hypothesis not in self.active_hypotheses:
            self.active_hypotheses.append(hypothesis)
        
        # Maintain capacity limit
        if len(self.active_hypotheses) > self.max_capacity // 2:
            self.active_hypotheses.pop(0)
    
    def update_context(self, key: str, value: any):
        """Update context information."""
        self.current_context[key] = value
    
    def get_relevant_facts(self, query: str, top_k: int = 5) -> List[str]:
        """Get facts relevant to a query."""
        query_words = set(query.lower().split())
        
        # Score facts by word overlap and confidence
        fact_scores = []
        for fact in self.recent_facts:
            fact_words = set(fact.lower().split())
            overlap = len(query_words.intersection(fact_words))
            confidence = self.confidence_scores.get(fact, 1.0)
            score = overlap * confidence
            
            if score > 0:
                fact_scores.append((score, fact))
        
        # Sort by score and return top_k
        fact_scores.sort(key=lambda x: x[0], reverse=True)
        return [fact for _, fact in fact_scores[:top_k]]
    
    def clear_working_memory(self):
        """Clear all working memory contents."""
        self.current_context.clear()
        self.recent_facts.clear()
        self.active_hypotheses.clear()
        self.confidence_scores.clear()
    
    def get_memory_summary(self) -> Dict[str, any]:
        """Get a summary of current working memory state."""
        return {
            "num_facts": len(self.recent_facts),
            "num_hypotheses": len(self.active_hypotheses),
            "context_keys": list(self.current_context.keys()),
            "avg_confidence": sum(self.confidence_scores.values()) / len(self.confidence_scores) if self.confidence_scores else 0.0
        }