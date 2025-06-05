import re
import time
import random
import numpy as np
from typing import Dict, List, Optional, Set
from .config import ThoughtNode, ReasoningState, AgentRole, EnhancedRSearchConfig


class TreeOfThoughts:
    """Tree of Thoughts reasoning with backtracking and pruning."""
    
    def __init__(self, config: EnhancedRSearchConfig):
        self.config = config
        self.nodes: Dict[str, ThoughtNode] = {}
        self.root_id: Optional[str] = None
        self.current_path: List[str] = []
        self.explored_paths: Set[str] = set()
        self.best_leaf: Optional[ThoughtNode] = None
    
    def create_root(self, question: str) -> str:
        """Create root node for the reasoning tree."""
        root_id = f"root_{hash(question) % 10000}"
        root_node = ThoughtNode(
            id=root_id,
            content=f"Question: {question}",
            depth=0,
            state=ReasoningState.PLANNING
        )
        self.nodes[root_id] = root_node
        self.root_id = root_id
        self.current_path = [root_id]
        return root_id
    
    def expand_node(self, node_id: str, new_thoughts: List[str], agent_role: AgentRole) -> List[str]:
        """Expand a node with new thought branches."""
        if node_id not in self.nodes:
            return []
            
        parent_node = self.nodes[node_id]
        new_node_ids = []
        
        for i, thought in enumerate(new_thoughts):
            child_id = f"{node_id}_child_{i}_{int(time.time() * 1000) % 10000}"
            child_node = ThoughtNode(
                id=child_id,
                content=thought,
                parent_id=node_id,
                depth=parent_node.depth + 1,
                agent_role=agent_role,
                state=ReasoningState.REASONING
            )
            
            self.nodes[child_id] = child_node
            parent_node.children_ids.append(child_id)
            new_node_ids.append(child_id)
        
        return new_node_ids
    
    def evaluate_node(self, node_id: str, evaluator_agent) -> float:
        """Evaluate the quality of a thought node."""
        if node_id not in self.nodes:
            return 0.0
            
        node = self.nodes[node_id]
        
        # Create evaluation prompt
        path_content = self._get_path_content(node_id)
        eval_prompt = f"Evaluate the quality and promise of this reasoning path (0-1 scale):\n{path_content}"
        
        response = evaluator_agent.generate_response(eval_prompt)
        
        # Extract score (simple regex, can be enhanced)
        score_match = re.search(r'(\d*\.?\d+)', response)
        score = float(score_match.group(1)) if score_match else 0.5
        score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
        
        node.score = score
        return score
    
    def select_best_path(self) -> List[str]:
        """Select the best reasoning path using tree search."""
        if not self.root_id:
            return []
        
        # Use Monte Carlo Tree Search-like approach
        best_path = []
        best_score = -1
        
        def dfs(node_id: str, current_path: List[str], depth: int):
            nonlocal best_path, best_score
            
            if depth > self.config.max_tree_depth:
                return
            
            if node_id not in self.nodes:
                return
                
            node = self.nodes[node_id]
            
            # If leaf node, evaluate complete path
            if not node.children_ids or node.is_terminal:
                path_score = self._evaluate_path(current_path)
                if path_score > best_score:
                    best_score = path_score
                    best_path = current_path.copy()
                return
            
            # Continue exploring children
            for child_id in node.children_ids:
                if node.score > self.config.pruning_threshold:  # Pruning
                    dfs(child_id, current_path + [child_id], depth + 1)
        
        dfs(self.root_id, [self.root_id], 0)
        return best_path
    
    def backtrack(self, node_id: str) -> Optional[str]:
        """Backtrack from a failed node."""
        if node_id not in self.nodes:
            return None
            
        node = self.nodes[node_id]
        node.backtrack_count += 1
        
        if node.parent_id and node.backtrack_count < 3:
            parent = self.nodes[node.parent_id]
            # Find alternative children or create new ones
            unexplored_children = [cid for cid in parent.children_ids 
                                 if cid in self.nodes and self.nodes[cid].score > self.config.pruning_threshold]
            
            if unexplored_children:
                return random.choice(unexplored_children)
        
        return None
    
    def _get_path_content(self, node_id: str) -> str:
        """Get the content of the path from root to node."""
        path = []
        current_id = node_id
        
        while current_id and current_id in self.nodes:
            node = self.nodes[current_id]
            path.append(f"Step {node.depth}: {node.content}")
            current_id = node.parent_id
        
        return "\n".join(reversed(path))
    
    def _evaluate_path(self, path: List[str]) -> float:
        """Evaluate the quality of a complete reasoning path."""
        scores = [self.nodes[node_id].score for node_id in path if node_id in self.nodes]
        return np.mean(scores) if scores else 0.0
    
    def get_path_summary(self, path: List[str]) -> str:
        """Get a summary of the reasoning path."""
        summary_parts = []
        for node_id in path:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                summary_parts.append(f"{node.agent_role.value}: {node.content[:100]}...")
        
        return "\n".join(summary_parts)
    
    def prune_low_scoring_branches(self, threshold: float = None):
        """Remove branches with scores below threshold."""
        if threshold is None:
            threshold = self.config.pruning_threshold
        
        nodes_to_remove = []
        for node_id, node in self.nodes.items():
            if node.score < threshold and node_id != self.root_id:
                nodes_to_remove.append(node_id)
        
        # Remove nodes and update parent references
        for node_id in nodes_to_remove:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                if node.parent_id and node.parent_id in self.nodes:
                    parent = self.nodes[node.parent_id]
                    if node_id in parent.children_ids:
                        parent.children_ids.remove(node_id)
                del self.nodes[node_id]


class ReasoningPlanner:
    """High-level reasoning planner for complex tasks."""
    
    def __init__(self, config: EnhancedRSearchConfig):
        self.config = config
        self.planning_strategies = {
            "decomposition": self._decomposition_strategy,
            "analogy": self._analogy_strategy,
            "causal": self._causal_strategy,
            "temporal": self._temporal_strategy
        }
    
    def create_reasoning_plan(self, question: str, context: Dict = None) -> Dict:
        """Create a comprehensive reasoning plan for the question."""
        question_type = self._classify_question(question)
        strategy = self._select_strategy(question_type)
        
        plan = {
            "question": question,
            "question_type": question_type,
            "strategy": strategy,
            "steps": self.planning_strategies[strategy](question, context),
            "estimated_complexity": self._estimate_complexity(question),
            "required_tools": self._identify_required_tools(question)
        }
        
        return plan
    
    def _classify_question(self, question: str) -> str:
        """Classify the type of question for appropriate planning."""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["when", "date", "year", "time", "before", "after"]):
            return "temporal"
        elif any(word in question_lower for word in ["why", "because", "reason", "cause", "effect"]):
            return "causal"
        elif any(word in question_lower for word in ["compare", "difference", "similar", "versus"]):
            return "comparison"
        elif any(word in question_lower for word in ["how many", "count", "number", "quantity"]):
            return "quantitative"
        elif "?" in question and len(question.split()) > 10:
            return "complex_multi_part"
        else:
            return "factual"
    
    def _select_strategy(self, question_type: str) -> str:
        """Select reasoning strategy based on question type."""
        strategy_map = {
            "temporal": "temporal",
            "causal": "causal",
            "comparison": "analogy",
            "complex_multi_part": "decomposition",
            "quantitative": "decomposition",
            "factual": "decomposition"
        }
        return strategy_map.get(question_type, "decomposition")
    
    def _decomposition_strategy(self, question: str, context: Dict = None) -> List[str]:
        """Break down complex questions into sub-problems."""
        steps = [
            "Identify the main components of the question",
            "Break down into answerable sub-questions",
            "Determine dependencies between sub-questions",
            "Plan information gathering for each component",
            "Execute reasoning for each sub-component",
            "Integrate results into final answer"
        ]
        return steps
    
    def _analogy_strategy(self, question: str, context: Dict = None) -> List[str]:
        """Use analogical reasoning for comparison questions."""
        steps = [
            "Identify entities or concepts to compare",
            "Gather information about each entity",
            "Identify comparison dimensions",
            "Analyze similarities and differences",
            "Draw conclusions based on comparison"
        ]
        return steps
    
    def _causal_strategy(self, question: str, context: Dict = None) -> List[str]:
        """Use causal reasoning for why/how questions."""
        steps = [
            "Identify the effect or outcome in question",
            "Search for potential causes or mechanisms",
            "Analyze causal relationships",
            "Verify causal links with evidence",
            "Construct causal explanation"
        ]
        return steps
    
    def _temporal_strategy(self, question: str, context: Dict = None) -> List[str]:
        """Use temporal reasoning for time-based questions."""
        steps = [
            "Identify temporal entities and events",
            "Establish timeline of relevant events",
            "Determine temporal relationships",
            "Locate specific temporal information",
            "Construct temporal answer"
        ]
        return steps
    
    def _estimate_complexity(self, question: str) -> str:
        """Estimate the complexity level of the question."""
        word_count = len(question.split())
        question_marks = question.count('?')
        
        if word_count > 20 or question_marks > 1:
            return "high"
        elif word_count > 10:
            return "medium"
        else:
            return "low"
    
    def _identify_required_tools(self, question: str) -> List[str]:
        """Identify which tools are likely needed for the question."""
        question_lower = question.lower()
        tools = []
        
        if any(word in question_lower for word in ["search", "find", "information", "about"]):
            tools.append("search")
        
        if any(word in question_lower for word in ["calculate", "compute", "math", "number"]):
            tools.append("calculator")
        
        if any(word in question_lower for word in ["relationship", "connected", "related"]):
            tools.append("knowledge_graph")
        
        if any(word in question_lower for word in ["code", "program", "execute"]):
            tools.append("code_executor")
        
        if any(word in question_lower for word in ["current", "latest", "recent", "today"]):
            tools.append("web_scraper")
        
        return tools if tools else ["search"]  # Default to search


class VerificationEngine:
    """Engine for verifying reasoning steps and final answers."""
    
    def __init__(self, config: EnhancedRSearchConfig):
        self.config = config
        self.verification_criteria = [
            "factual_accuracy",
            "logical_consistency",
            "completeness",
            "relevance"
        ]
    
    def verify_reasoning_step(self, step_content: str, context: Dict = None) -> Dict:
        """Verify a single reasoning step."""
        verification_result = {
            "step": step_content,
            "verified": True,
            "confidence": 0.8,
            "issues": [],
            "suggestions": []
        }
        
        # Simple verification checks (can be enhanced with ML models)
        if self._check_logical_consistency(step_content):
            verification_result["confidence"] += 0.1
        else:
            verification_result["issues"].append("Potential logical inconsistency")
            verification_result["confidence"] -= 0.2
        
        if self._check_factual_plausibility(step_content):
            verification_result["confidence"] += 0.1
        else:
            verification_result["issues"].append("Factual claims need verification")
            verification_result["confidence"] -= 0.1
        
        verification_result["verified"] = verification_result["confidence"] > self.config.verification_threshold
        
        return verification_result
    
    def verify_final_answer(self, answer: str, question: str, reasoning_trace: List = None) -> Dict:
        """Verify the final answer against the question and reasoning."""
        verification_result = {
            "answer": answer,
            "question": question,
            "verified": True,
            "confidence": 0.7,
            "issues": [],
            "completeness_score": 0.8,
            "relevance_score": 0.9
        }
        
        # Check answer completeness
        if self._check_answer_completeness(answer, question):
            verification_result["completeness_score"] = 0.9
        else:
            verification_result["issues"].append("Answer may be incomplete")
            verification_result["completeness_score"] = 0.6
        
        # Check answer relevance
        if self._check_answer_relevance(answer, question):
            verification_result["relevance_score"] = 0.9
        else:
            verification_result["issues"].append("Answer may not fully address the question")
            verification_result["relevance_score"] = 0.6
        
        # Overall confidence
        verification_result["confidence"] = (
            verification_result["completeness_score"] * 0.5 +
            verification_result["relevance_score"] * 0.5
        )
        
        verification_result["verified"] = verification_result["confidence"] > self.config.verification_threshold
        
        return verification_result
    
    def _check_logical_consistency(self, text: str) -> bool:
        """Check for logical consistency in text."""
        # Simple contradiction detection
        contradiction_words = ["but", "however", "although", "despite", "contradicts"]
        text_lower = text.lower()
        
        contradiction_count = sum(1 for word in contradiction_words if word in text_lower)
        return contradiction_count <= 1  # Allow one contrasting statement
    
    def _check_factual_plausibility(self, text: str) -> bool:
        """Check factual plausibility of claims."""
        implausible_patterns = [
            r'\d{4,}%',  # Very high percentages
            r'always.*never',  # Absolute contradictions
            r'impossible.*definitely'  # Contradictory certainty
        ]
        
        for pattern in implausible_patterns:
            if re.search(pattern, text.lower()):
                return False
        
        return True
    
    def _check_answer_completeness(self, answer: str, question: str) -> bool:
        """Check if answer addresses all parts of the question."""
        question_words = question.lower().split()
        answer_words = answer.lower().split()
        
        # Simple heuristic: answer should contain some key question words
        key_words = [word for word in question_words if len(word) > 3]
        covered_words = [word for word in key_words if word in answer_words]
        
        coverage_ratio = len(covered_words) / len(key_words) if key_words else 1.0
        return coverage_ratio > 0.3  # At least 30% coverage
    
    def _check_answer_relevance(self, answer: str, question: str) -> bool:
        """Check if answer is relevant to the question."""
        # Simple relevance check based on common words
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        
        common_words = question_words.intersection(answer_words)
        relevance_score = len(common_words) / len(question_words) if question_words else 0
        
        return relevance_score > 0.2  # At least 20% word overlap