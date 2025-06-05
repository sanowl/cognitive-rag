
from dataclasses import dataclass, field
from enum import Enum
from typing import List
import time


class AgentRole(Enum):
    """Different agent roles in the multi-agent system."""
    PLANNER = "planner"           # High-level planning and decomposition
    SEARCHER = "searcher"         # Information retrieval specialist
    REASONER = "reasoner"         # Logical reasoning specialist
    VERIFIER = "verifier"         # Answer verification and critique
    CRITIC = "critic"             # Constitutional self-improvement
    SYNTHESIZER = "synthesizer"   # Information integration
    MEMORY_MANAGER = "memory"     # Episodic memory management


class ReasoningState(Enum):
    """States in the reasoning process."""
    PLANNING = "planning"
    SEARCHING = "searching"
    REASONING = "reasoning"
    VERIFYING = "verifying"
    CRITIQUING = "critiquing"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ThoughtNode:
    """Node in the tree of thoughts."""
    id: str
    content: str
    parent_id: str = None
    children_ids: List[str] = field(default_factory=list)
    depth: int = 0
    score: float = 0.0
    state: ReasoningState = ReasoningState.PLANNING
    agent_role: AgentRole = AgentRole.PLANNER
    evidence: List[str] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    is_terminal: bool = False
    backtrack_count: int = 0
    created_at: float = field(default_factory=time.time)


@dataclass
class EpisodicMemory:
    """Episodic memory for cross-session learning."""
    question_type: str
    reasoning_pattern: str
    successful_tools: List[str]
    effective_strategies: List[str]
    common_pitfalls: List[str]
    success_score: float
    frequency: int = 1


@dataclass
class EnhancedRSearchConfig:
    """Enhanced configuration with multi-agent and advanced features."""
    
    # Base model configuration
    policy_model_name: str = "microsoft/DialoGPT-medium"
    cross_family_model_name: str = "microsoft/DialoGPT-small"
    critic_model_name: str = "microsoft/DialoGPT-small"
    max_sequence_length: int = 1024
    
    # Multi-agent configuration
    num_agents: int = 5
    agent_collaboration_threshold: float = 0.7
    max_agent_iterations: int = 10
    
    # Tree of Thoughts configuration
    max_tree_depth: int = 8
    max_children_per_node: int = 4
    tree_search_budget: int = 20
    pruning_threshold: float = 0.3
    backtrack_penalty: float = 0.1
    
    # Self-reflection configuration
    verification_threshold: float = 0.8
    self_critique_iterations: int = 3
    constitutional_learning_rate: float = 0.1
    
    # Memory configuration
    episodic_memory_size: int = 1000
    memory_decay_factor: float = 0.95
    memory_retrieval_top_k: int = 5
    
    # Training configuration
    learning_rate: float = 1e-6
    batch_size: int = 4
    num_epochs: int = 3
    warmup_ratio: float = 0.95
    kl_coefficient: float = 0.001
    
    # Reward configuration
    gamma_evidence: float = 0.3
    gamma_answer: float = 0.3
    gamma_consistency: float = 0.2
    gamma_novelty: float = 0.1
    gamma_efficiency: float = 0.1
    
    # Tool configuration
    available_tools: List[str] = field(default_factory=lambda: [
        "search", "calculator", "knowledge_graph", "code_executor", "web_scraper"
    ])
    max_tool_calls: int = 10
    tool_timeout: float = 30.0
    
    # Progressive reasoning
    difficulty_levels: List[str] = field(default_factory=lambda: ["easy", "medium", "hard", "expert"])
    adaptive_difficulty: bool = True
    
    # Special tokens (enhanced)
    search_start_token: str = "<search>"
    search_end_token: str = "</search>"
    observation_start_token: str = "<observation>"
    observation_end_token: str = "</observation>"
    evidence_start_token: str = "<original_evidence>"
    evidence_end_token: str = "</original_evidence>"
    answer_start_token: str = "<answer>"
    answer_end_token: str = "</answer>"
    thought_start_token: str = "<thought>"
    thought_end_token: str = "</thought>"
    critique_start_token: str = "<critique>"
    critique_end_token: str = "</critique>"
    verification_start_token: str = "<verify>"
    verification_end_token: str = "</verify>"