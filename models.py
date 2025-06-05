import re
import logging
from typing import Dict, List, Any
from .config import EnhancedRSearchConfig
from .utils import TextProcessor

logger = logging.getLogger(__name__)


class EnhancedRewardModel:
    """Enhanced reward model with multiple sophisticated metrics."""
    
    def __init__(self, config: EnhancedRSearchConfig, cross_family_model, cross_family_tokenizer):
        self.config = config
        self.cross_family_model = cross_family_model
        self.cross_family_tokenizer = cross_family_tokenizer
        self.text_processor = TextProcessor()
        self.reward_history = []
    
    def compute_comprehensive_reward(self, question: str, reasoning_trace: Dict[str, Any], 
                                   gold_answer: str, memory_manager) -> Dict[str, float]:
        """Compute comprehensive multi-dimensional reward."""
        
        # Base rewards
        answer_reward = self._compute_answer_reward(reasoning_trace.get("final_answer", ""), gold_answer)
        evidence_reward = self._compute_evidence_reward(question, reasoning_trace, gold_answer)
        
        # Advanced rewards
        consistency_reward = self._compute_consistency_reward(reasoning_trace)
        novelty_reward = self._compute_novelty_reward(reasoning_trace, memory_manager)
        efficiency_reward = self._compute_efficiency_reward(reasoning_trace)
        
        # Combine rewards
        total_reward = (
            self.config.gamma_answer * answer_reward +
            self.config.gamma_evidence * evidence_reward +
            self.config.gamma_consistency * consistency_reward +
            self.config.gamma_novelty * novelty_reward +
            self.config.gamma_efficiency * efficiency_reward
        )
        
        reward_dict = {
            'answer_reward': answer_reward,
            'evidence_reward': evidence_reward,
            'consistency_reward': consistency_reward,
            'novelty_reward': novelty_reward,
            'efficiency_reward': efficiency_reward,
            'total_reward': total_reward
        }
        
        self.reward_history.append(reward_dict)
        return reward_dict
    
    def _compute_answer_reward(self, predicted_answer: str, gold_answer: str) -> float:
        """Compute answer accuracy reward."""
        return self.text_processor.compute_f1_score(predicted_answer, gold_answer)
    
    def _compute_evidence_reward(self, question: str, reasoning_trace: Dict[str, Any], gold_answer: str) -> float:
        """Compute evidence quality reward using cross-family verification."""
        evidence = reasoning_trace.get("reasoning_results", {})
        if not evidence:
            return 0.0
        
        # Combine all evidence
        combined_evidence = " ".join([
            result.get("result", "") for result in evidence.values()
        ])
        
        # Use cross-family model to verify evidence quality
        evidence_prompt = f"Based on this evidence, answer: {question}\nEvidence: {combined_evidence}\nAnswer:"
        
        try:
            inputs = self.cross_family_tokenizer.encode(evidence_prompt, return_tensors='tf', max_length=512, truncation=True)
            outputs = self.cross_family_model.generate(
                inputs, 
                max_length=inputs.shape[1] + 50,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.cross_family_tokenizer.eos_token_id
            )
            
            cross_answer = self.cross_family_tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True).strip()
            return self.text_processor.compute_f1_score(cross_answer, gold_answer)
        
        except Exception as e:
            logger.warning(f"Evidence reward computation failed: {e}")
            return 0.0
    
    def _compute_consistency_reward(self, reasoning_trace: Dict[str, Any]) -> float:
        """Compute consistency across reasoning steps."""
        results = reasoning_trace.get("reasoning_results", {})
        if len(results) < 2:
            return 1.0
        
        # Check for contradictions
        result_texts = [result.get("result", "") for result in results.values()]
        
        # Simple contradiction detection (can be enhanced)
        contradiction_markers = ["however", "but", "contradicts", "inconsistent", "disagree"]
        
        contradiction_count = 0
        for text in result_texts:
            text_lower = text.lower()
            contradiction_count += sum(1 for marker in contradiction_markers if marker in text_lower)
        
        # Higher consistency = lower contradictions
        consistency = max(0.0, 1.0 - contradiction_count * 0.2)
        return consistency
    
    def _compute_novelty_reward(self, reasoning_trace: Dict[str, Any], memory_manager) -> float:
        """Compute novelty reward for creative reasoning."""
        # Compare with historical reasoning patterns
        current_pattern = self._extract_reasoning_pattern(reasoning_trace)
        
        # Check similarity with stored memories
        similar_count = 0
        for memory in memory_manager.memories[-50:]:  # Check recent memories
            if self._pattern_similarity(current_pattern, memory.reasoning_pattern) > 0.8:
                similar_count += 1
        
        # Higher novelty for less similar patterns
        novelty = max(0.0, 1.0 - similar_count * 0.1)
        return novelty
    
    def _compute_efficiency_reward(self, reasoning_trace: Dict[str, Any]) -> float:
        """Compute efficiency reward based on resource usage."""
        # Factors: number of steps, tools used, time taken
        num_steps = len(reasoning_trace.get("reasoning_results", {}))
        agents_used = len(reasoning_trace.get("agents_involved", []))
        
        # Prefer concise but complete reasoning
        if num_steps == 0:
            return 0.0
        
        efficiency = 1.0 / (1.0 + num_steps * 0.1 + agents_used * 0.05)
        return min(1.0, efficiency)
    
    def _extract_reasoning_pattern(self, reasoning_trace: Dict[str, Any]) -> str:
        """Extract reasoning pattern signature."""
        pattern_elements = []
        
        # Add agent sequence
        agents = reasoning_trace.get("agents_involved", [])
        pattern_elements.append(f"agents:{'-'.join(str(agent) for agent in agents)}")
        
        # Add number of steps
        num_steps = len(reasoning_trace.get("reasoning_results", {}))
        pattern_elements.append(f"steps:{num_steps}")
        
        return "|".join(pattern_elements)
    
    def _pattern_similarity(self, pattern1: str, pattern2: str) -> float:
        """Compute similarity between reasoning patterns."""
        if pattern1 == pattern2:
            return 1.0
        tokens1 = set(pattern1.split("|"))
        tokens2 = set(pattern2.split("|"))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union)
    
    def get_reward_statistics(self) -> Dict[str, float]:
        """Get statistics about reward trends."""
        if not self.reward_history:
            return {}
        
        recent_rewards = self.reward_history[-100:]  # Last 100 episodes
        
        stats = {}
        for key in ['answer_reward', 'evidence_reward', 'consistency_reward', 'novelty_reward', 'efficiency_reward', 'total_reward']:
            values = [reward[key] for reward in recent_rewards if key in reward]
            if values:
                stats[f"{key}_mean"] = sum(values) / len(values)
                stats[f"{key}_max"] = max(values)
                stats[f"{key}_min"] = min(values)
        
        return stats


class ConstitutionalSelfImprovement:
    """Constitutional AI for self-improvement and critique."""
    
    def __init__(self, config: EnhancedRSearchConfig, critic_model, critic_tokenizer):
        self.config = config
        self.critic_model = critic_model
        self.critic_tokenizer = critic_tokenizer
        self.constitutional_principles = [
            "Be helpful and accurate",
            "Avoid harmful or biased responses",
            "Provide evidence-based reasoning",
            "Acknowledge uncertainty when appropriate",
            "Be concise but thorough",
            "Respect factual accuracy over speculation"
        ]
        self.improvement_history = []
    
    def critique_response(self, question: str, response: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate constitutional critique of a response."""
        critique_prompt = f"""
        Constitutional Principles:
        {chr(10).join(f'- {p}' for p in self.constitutional_principles)}
        
        Question: {question}
        Response: {response}
        
        Evaluate this response against the constitutional principles. Identify:
        1. Strengths
        2. Weaknesses  
        3. Specific improvements needed
        4. Overall quality score (0-1)
        
        Critique:
        """
        
        inputs = self.critic_tokenizer.encode(critique_prompt, return_tensors='tf', max_length=512, truncation=True)
        
        outputs = self.critic_model.generate(
            inputs,
            max_length=inputs.shape[1] + 200,
            temperature=0.3,
            do_sample=True,
            pad_token_id=self.critic_tokenizer.eos_token_id
        )
        
        critique = self.critic_tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        # Extract quality score
        score_match = re.search(r'quality score.*?(\d*\.?\d+)', critique.lower())
        quality_score = float(score_match.group(1)) if score_match else 0.5
        
        return {
            "critique": critique,
            "quality_score": quality_score,
            "principles_checked": len(self.constitutional_principles),
            "needs_improvement": quality_score < 0.7
        }
    
    def generate_improvement(self, original_response: str, critique: Dict[str, Any]) -> str:
        """Generate improved response based on critique."""
        if not critique["needs_improvement"]:
            return original_response
        
        improvement_prompt = f"""
        Original response: {original_response}
        
        Critique: {critique['critique']}
        
        Generate an improved response that addresses the critique while maintaining accuracy:
        """
        
        inputs = self.critic_tokenizer.encode(improvement_prompt, return_tensors='tf', max_length=512, truncation=True)
        
        outputs = self.critic_model.generate(
            inputs,
            max_length=inputs.shape[1] + 200,
            temperature=0.5,
            do_sample=True,
            pad_token_id=self.critic_tokenizer.eos_token_id
        )
        
        improved_response = self.critic_tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        # Store improvement for learning
        self.improvement_history.append({
            "original": original_response,
            "improved": improved_response,
            "critique_score": critique["quality_score"]
        })
        
        return improved_response.strip()
    
    def add_custom_principle(self, principle: str):
        """Add a custom constitutional principle."""
        if principle not in self.constitutional_principles:
            self.constitutional_principles.append(principle)
    
    def remove_principle(self, principle: str):
        """Remove a constitutional principle."""
        if principle in self.constitutional_principles:
            self.constitutional_principles.remove(principle)
    
    def get_improvement_statistics(self) -> Dict[str, Any]:
        """Get statistics about improvements made."""
        if not self.improvement_history:
            return {"total_improvements": 0}
        
        total_improvements = len(self.improvement_history)
        avg_critique_score = sum(item["critique_score"] for item in self.improvement_history) / total_improvements
        
        return {
            "total_improvements": total_improvements,
            "average_critique_score": avg_critique_score,
            "recent_improvements": min(10, total_improvements),
            "improvement_trend": "improving" if avg_critique_score > 0.6 else "needs_work"
        }


class PolicyModel:
    """Policy model for reasoning and search generation."""
    
    def __init__(self, config: EnhancedRSearchConfig, model, tokenizer):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.generation_history = []
    
    def generate_response(self, prompt: str, max_length: int = None, temperature: float = 0.8) -> str:
        """Generate response using the policy model."""
        if max_length is None:
            max_length = self.config.max_sequence_length
        
        inputs = self.tokenizer.encode(prompt, return_tensors='tf', max_length=max_length, truncation=True)
        
        outputs = self.model.generate(
            inputs,
            max_length=inputs.shape[1] + 200,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        # Store generation for analysis
        self.generation_history.append({
            "prompt_length": len(prompt),
            "response_length": len(response),
            "temperature": temperature
        })
        
        return response.strip()
    
    def get_log_probabilities(self, input_ids, attention_mask):
        """Get log probabilities for given input."""
        import tensorflow as tf
        
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        return log_probs
    
    def update_special_tokens(self, new_tokens: List[str]):
        """Update special tokens in the tokenizer."""
        self.tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})
        self.model.resize_token_embeddings(len(self.tokenizer))
    
    def get_generation_statistics(self) -> Dict[str, float]:
        """Get statistics about generation patterns."""
        if not self.generation_history:
            return {}
        
        recent_generations = self.generation_history[-100:]  # Last 100 generations
        
        avg_prompt_length = sum(gen["prompt_length"] for gen in recent_generations) / len(recent_generations)
        avg_response_length = sum(gen["response_length"] for gen in recent_generations) / len(recent_generations)
        avg_temperature = sum(gen["temperature"] for gen in recent_generations) / len(recent_generations)
        
        return {
            "total_generations": len(self.generation_history),
            "avg_prompt_length": avg_prompt_length,
            "avg_response_length": avg_response_length,
            "avg_temperature": avg_temperature,
            "response_length_trend": "increasing" if avg_response_length > 100 else "stable"
        }


class MultiModalModel:
    """Multi-modal model for handling different types of inputs."""
    
    def __init__(self, config: EnhancedRSearchConfig):
        self.config = config
        self.text_processor = TextProcessor()
        self.supported_modalities = ["text", "structured_data", "code"]
    
    def process_input(self, input_data: Any, modality: str) -> str:
        """Process input based on its modality."""
        if modality == "text":
            return self._process_text(input_data)
        elif modality == "structured_data":
            return self._process_structured_data(input_data)
        elif modality == "code":
            return self._process_code(input_data)
        else:
            raise ValueError(f"Unsupported modality: {modality}")
    
    def _process_text(self, text: str) -> str:
        """Process text input."""
        # Basic text cleaning and normalization
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        return text
    
    def _process_structured_data(self, data: Dict) -> str:
        """Process structured data input."""
        # Convert structured data to natural language description
        description_parts = []
        
        for key, value in data.items():
            if isinstance(value, (list, tuple)):
                value_str = ", ".join(str(v) for v in value)
                description_parts.append(f"{key}: {value_str}")
            else:
                description_parts.append(f"{key}: {value}")
        
        return "Structured data: " + "; ".join(description_parts)
    
    def _process_code(self, code: str) -> str:
        """Process code input."""
        # Basic code analysis and description
        lines = code.strip().split('\n')
        
        analysis = []
        analysis.append(f"Code snippet with {len(lines)} lines")
        
        # Identify programming constructs
        if any('def ' in line for line in lines):
            analysis.append("contains function definitions")
        if any('class ' in line for line in lines):
            analysis.append("contains class definitions")
        if any('for ' in line or 'while ' in line for line in lines):
            analysis.append("contains loops")
        if any('if ' in line for line in lines):
            analysis.append("contains conditional statements")
        
        return f"Code analysis: {', '.join(analysis)}. Original code: {code}"
    
    def detect_modality(self, input_data: Any) -> str:
        """Automatically detect the modality of input data."""
        if isinstance(input_data, str):
            # Check if it looks like code
            code_indicators = ['def ', 'class ', 'import ', 'for ', 'while ', 'if __name__']
            if any(indicator in input_data for indicator in code_indicators):
                return "code"
            else:
                return "text"
        elif isinstance(input_data, dict):
            return "structured_data"
        else:
            return "text"  # Default to text