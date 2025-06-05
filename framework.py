"""
Main Enhanced R-Search Framework orchestrating all components.
"""

import logging
import json
import pickle
from typing import Dict, List, Any, Optional
from transformers import AutoTokenizer, TFAutoModelForCausalLM

from .config import EnhancedRSearchConfig, AgentRole
from .agents import Agent, MultiAgentCoordinator
from .models import EnhancedRewardModel, ConstitutionalSelfImprovement, PolicyModel
from .reasoning import TreeOfThoughts, ReasoningPlanner, VerificationEngine
from .memory import EpisodicMemoryManager, WorkingMemory
from .tools import Tool, create_default_tools
from .utils import TextProcessor, IOManager, LoggingManager, PerformanceMonitor

logger = logging.getLogger(__name__)


class EnhancedRSearchFramework:
    """Enhanced R-Search framework with all advanced features."""
    
    def __init__(self, config: EnhancedRSearchConfig, tools: Dict[str, Tool] = None):
        self.config = config
        self.tools = tools or create_default_tools()
        self.performance_monitor = PerformanceMonitor()
        
        logger.info("Initializing Enhanced R-Search Framework...")
        
        # Initialize models
        self._initialize_models()
        
        # Initialize components
        self._initialize_components()
        
        logger.info("Enhanced R-Search Framework initialized successfully!")
    
    def _initialize_models(self):
        """Initialize all ML models."""
        logger.info("Loading models...")
        
        # Initialize policy model
        self.policy_tokenizer = AutoTokenizer.from_pretrained(self.config.policy_model_name)
        self.policy_model = TFAutoModelForCausalLM.from_pretrained(self.config.policy_model_name)
        
        # Add special tokens
        special_tokens = [
            self.config.search_start_token, self.config.search_end_token,
            self.config.observation_start_token, self.config.observation_end_token,
            self.config.evidence_start_token, self.config.evidence_end_token,
            self.config.answer_start_token, self.config.answer_end_token,
            self.config.thought_start_token, self.config.thought_end_token,
            self.config.critique_start_token, self.config.critique_end_token,
            self.config.verification_start_token, self.config.verification_end_token
        ]
        
        self.policy_tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        self.policy_model.resize_token_embeddings(len(self.policy_tokenizer))
        
        if self.policy_tokenizer.pad_token is None:
            self.policy_tokenizer.pad_token = self.policy_tokenizer.eos_token
        
        # Initialize cross-family model
        self.cross_family_tokenizer = AutoTokenizer.from_pretrained(self.config.cross_family_model_name)
        self.cross_family_model = TFAutoModelForCausalLM.from_pretrained(self.config.cross_family_model_name)
        
        if self.cross_family_tokenizer.pad_token is None:
            self.cross_family_tokenizer.pad_token = self.cross_family_tokenizer.eos_token
        
        # Initialize critic model
        self.critic_tokenizer = AutoTokenizer.from_pretrained(self.config.critic_model_name)
        self.critic_model = TFAutoModelForCausalLM.from_pretrained(self.config.critic_model_name)
        
        if self.critic_tokenizer.pad_token is None:
            self.critic_tokenizer.pad_token = self.critic_tokenizer.eos_token
    
    def _initialize_components(self):
        """Initialize all framework components."""
        logger.info("Initializing components...")
        
        # Initialize models
        self.policy_wrapper = PolicyModel(self.config, self.policy_model, self.policy_tokenizer)
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Initialize core components
        self.tree_of_thoughts = TreeOfThoughts(self.config)
        self.memory_manager = EpisodicMemoryManager(self.config)
        self.working_memory = WorkingMemory()
        self.reasoning_planner = ReasoningPlanner(self.config)
        self.verification_engine = VerificationEngine(self.config)
        
        # Initialize advanced components
        self.constitutional_improver = ConstitutionalSelfImprovement(
            self.config, self.critic_model, self.critic_tokenizer
        )
        self.coordinator = MultiAgentCoordinator(self.config, self.agents)
        self.reward_model = EnhancedRewardModel(
            self.config, self.cross_family_model, self.cross_family_tokenizer
        )
    
    def _initialize_agents(self) -> Dict[AgentRole, Agent]:
        """Initialize specialized agents."""
        agents = {}
        
        agent_roles = [
            AgentRole.PLANNER, AgentRole.SEARCHER, AgentRole.REASONER, 
            AgentRole.VERIFIER, AgentRole.SYNTHESIZER
        ]
        
        for role in agent_roles:
            agents[role] = Agent(role, self.config, self.policy_model, 
                               self.policy_tokenizer, self.tools)
        
        return agents
    
    def enhanced_inference(self, question: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform enhanced inference with all advanced features."""
        logger.info(f"Starting enhanced inference for: {question}")
        self.performance_monitor.start_monitoring()
        
        try:
            # Clear working memory for new question
            self.working_memory.clear_working_memory()
            
            # Initialize tree of thoughts
            root_id = self.tree_of_thoughts.create_root(question)
            
            # Create reasoning plan
            reasoning_plan = self.reasoning_planner.create_reasoning_plan(question, context)
            
            # Retrieve relevant memories
            relevant_memories = self.memory_manager.retrieve_relevant_memories(question)
            memory_suggestions = self.memory_manager.suggest_approach(question)
            
            # Update working memory with relevant information
            self.working_memory.update_context("question", question)
            self.working_memory.update_context("plan", reasoning_plan)
            self.working_memory.update_context("memory_suggestions", memory_suggestions)
            
            # Multi-agent coordination
            coordination_result = self.coordinator.coordinate_reasoning(question, self.tree_of_thoughts)
            
            # Constitutional self-improvement
            initial_answer = coordination_result["final_result"]["final_answer"]
            critique = self.constitutional_improver.critique_response(
                question, initial_answer, coordination_result
            )
            
            improved_answer = initial_answer
            if critique["needs_improvement"]:
                improved_answer = self.constitutional_improver.generate_improvement(
                    initial_answer, critique
                )
            
            # Verification of final answer
            verification_result = self.verification_engine.verify_final_answer(
                improved_answer, question, coordination_result.get("reasoning_results", {})
            )
            
            # Select best reasoning path
            best_path = self.tree_of_thoughts.select_best_path()
            
            # Final verification by verifier agent
            final_verification = self.agents[AgentRole.VERIFIER].generate_response(
                f"Final verification of answer '{improved_answer}' for question '{question}'"
            )
            
            # Store episode in memory
            self._store_episode(question, coordination_result, critique["quality_score"])
            
            # Record performance metrics
            self.performance_monitor.record_metrics({
                "question_length": len(question),
                "answer_length": len(improved_answer),
                "tree_nodes": len(self.tree_of_thoughts.nodes),
                "agents_used": len(coordination_result.get("agents_involved", [])),
                "quality_score": critique["quality_score"],
                "confidence": coordination_result["final_result"]["confidence"]
            })
            
            result = {
                "question": question,
                "reasoning_plan": reasoning_plan,
                "initial_answer": initial_answer,
                "improved_answer": improved_answer,
                "reasoning_trace": coordination_result,
                "critique": critique,
                "verification": verification_result,
                "final_verification": final_verification,
                "best_reasoning_path": best_path,
                "relevant_memories": relevant_memories,
                "memory_suggestions": memory_suggestions,
                "confidence": coordination_result["final_result"]["confidence"],
                "tree_nodes_explored": len(self.tree_of_thoughts.nodes),
                "working_memory_summary": self.working_memory.get_memory_summary()
            }
            
            logger.info(f"Enhanced inference completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error during enhanced inference: {e}")
            raise
    
    def batch_inference(self, questions: List[str], context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Perform batch inference on multiple questions."""
        logger.info(f"Starting batch inference on {len(questions)} questions")
        
        results = []
        for i, question in enumerate(questions):
            logger.info(f"Processing question {i+1}/{len(questions)}")
            try:
                result = self.enhanced_inference(question, context)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process question {i+1}: {e}")
                results.append({
                    "question": question,
                    "error": str(e),
                    "improved_answer": "",
                    "confidence": 0.0
                })
        
        return results
    
    def _store_episode(self, question: str, coordination_result: Dict[str, Any], quality_score: float):
        """Store reasoning episode in episodic memory."""
        reasoning_pattern = self._extract_pattern(coordination_result)
        tools_used = list(self.tools.keys())
        strategies = [str(agent) for agent in coordination_result.get("agents_involved", [])]
        
        self.memory_manager.store_episode(
            question=question,
            reasoning_pattern=reasoning_pattern,
            tools_used=tools_used,
            success_score=quality_score,
            strategies=strategies
        )
    
    def _extract_pattern(self, coordination_result: Dict[str, Any]) -> str:
        """Extract reasoning pattern from coordination result."""
        agents_involved = coordination_result.get("agents_involved", [])
        num_tasks = len(coordination_result.get("reasoning_results", {}))
        
        return f"agents:{'-'.join(str(agent) for agent in agents_involved)}_tasks:{num_tasks}"
    
    def evaluate_comprehensive(self, test_data: List[Dict[str, str]], 
                             output_file: str = None) -> Dict[str, float]:
        """Comprehensive evaluation with multiple metrics."""
        logger.info(f"Comprehensive evaluation on {len(test_data)} examples...")
        
        metrics = {
            "f1_score": 0.0,
            "exact_match": 0.0,
            "consistency": 0.0,
            "efficiency": 0.0,
            "novelty": 0.0,
            "constitutional_compliance": 0.0,
            "verification_score": 0.0
        }
        
        detailed_results = []
        total_examples = len(test_data)
        
        for i, item in enumerate(test_data):
            logger.info(f"Evaluating example {i+1}/{total_examples}")
            
            try:
                result = self.enhanced_inference(item['question'])
                predicted_answer = result['improved_answer']
                gold_answer = item['answer']
                
                # Basic metrics
                f1 = TextProcessor.compute_f1_score(predicted_answer, gold_answer)
                exact_match = 1.0 if predicted_answer.strip().lower() == gold_answer.strip().lower() else 0.0
                
                # Advanced metrics
                reward_dict = self.reward_model.compute_comprehensive_reward(
                    item['question'], result['reasoning_trace'], gold_answer, self.memory_manager
                )
                
                # Verification score
                verification_score = result['verification']['confidence'] if 'verification' in result else 0.5
                
                example_metrics = {
                    "f1_score": f1,
                    "exact_match": exact_match,
                    "consistency": reward_dict.get("consistency_reward", 0.0),
                    "efficiency": reward_dict.get("efficiency_reward", 0.0),
                    "novelty": reward_dict.get("novelty_reward", 0.0),
                    "constitutional_compliance": result['critique']['quality_score'],
                    "verification_score": verification_score
                }
                
                # Accumulate metrics
                for key in metrics:
                    metrics[key] += example_metrics[key]
                
                # Store detailed result
                detailed_results.append({
                    "question": item['question'],
                    "gold_answer": gold_answer,
                    "predicted_answer": predicted_answer,
                    "metrics": example_metrics,
                    "reasoning_trace": result.get('reasoning_trace', {}),
                    "confidence": result.get('confidence', 0.0)
                })
                
            except Exception as e:
                logger.error(f"Failed to evaluate example {i+1}: {e}")
                # Add zero scores for failed examples
                for key in metrics:
                    metrics[key] += 0.0
                
                detailed_results.append({
                    "question": item['question'],
                    "gold_answer": item['answer'],
                    "predicted_answer": "",
                    "error": str(e),
                    "metrics": {key: 0.0 for key in metrics}
                })
        
        # Average all metrics
        for key in metrics:
            metrics[key] /= total_examples
        
        # Add summary statistics
        metrics["total_examples"] = total_examples
        metrics["performance_summary"] = self.performance_monitor.get_summary_statistics()
        
        # Save detailed results if requested
        if output_file:
            evaluation_output = {
                "summary_metrics": metrics,
                "detailed_results": detailed_results,
                "config": self.config.__dict__
            }
            IOManager.save_json(evaluation_output, output_file)
            logger.info(f"Detailed evaluation results saved to {output_file}")
        
        return metrics
    
    def save_framework(self, path: str):
        """Save the entire framework state."""
        logger.info(f"Saving framework to {path}")
        IOManager.ensure_directory(f"{path}/")
        
        # Save models
        self.policy_model.save_pretrained(f"{path}/policy_model")
        self.policy_tokenizer.save_pretrained(f"{path}/policy_model")
        
        # Save memories
        IOManager.save_pickle(self.memory_manager.memories, f"{path}/episodic_memories.pkl")
        
        # Save configuration
        config_dict = {
            attr: getattr(self.config, attr) for attr in dir(self.config)
            if not attr.startswith('_') and not callable(getattr(self.config, attr))
        }
        IOManager.save_json(config_dict, f"{path}/config.json")
        
        # Save performance metrics
        self.performance_monitor.export_metrics(f"{path}/performance_metrics.json")
        
        # Save reward history
        IOManager.save_json(self.reward_model.reward_history, f"{path}/reward_history.json")
        
        # Save memory statistics
        memory_stats = self.memory_manager.get_pattern_statistics()
        IOManager.save_json(memory_stats, f"{path}/memory_statistics.json")
        
        logger.info(f"Framework saved successfully to {path}")
    
    def load_framework(self, path: str):
        """Load framework state."""
        logger.info(f"Loading framework from {path}")
        
        try:
            # Load models
            self.policy_model = TFAutoModelForCausalLM.from_pretrained(f"{path}/policy_model")
            self.policy_tokenizer = AutoTokenizer.from_pretrained(f"{path}/policy_model")
            
            # Load memories
            memories_path = f"{path}/episodic_memories.pkl"
            if Path(memories_path).exists():
                self.memory_manager.memories = IOManager.load_pickle(memories_path)
            else:
                logger.warning("No episodic memories found")
            
            # Load reward history
            reward_history_path = f"{path}/reward_history.json"
            if Path(reward_history_path).exists():
                self.reward_model.reward_history = IOManager.load_json(reward_history_path)
            
            logger.info(f"Framework loaded successfully from {path}")
            
        except Exception as e:
            logger.error(f"Failed to load framework: {e}")
            raise
    
    def get_framework_statistics(self) -> Dict[str, Any]:
        """Get comprehensive framework statistics."""
        stats = {
            "model_info": {
                "policy_model": self.config.policy_model_name,
                "cross_family_model": self.config.cross_family_model_name,
                "critic_model": self.config.critic_model_name
            },
            "memory_stats": {
                "total_memories": len(self.memory_manager.memories),
                "memory_patterns": len(self.memory_manager.get_pattern_statistics()),
                "working_memory": self.working_memory.get_memory_summary()
            },
            "agent_stats": {
                "total_agents": len(self.agents),
                "agent_roles": [role.value for role in self.agents.keys()]
            },
            "tool_stats": {
                "available_tools": list(self.tools.keys()),
                "total_tools": len(self.tools)
            },
            "performance_stats": self.performance_monitor.get_summary_statistics(),
            "reward_stats": self.reward_model.get_reward_statistics(),
            "constitutional_stats": self.constitutional_improver.get_improvement_statistics(),
            "tree_stats": {
                "total_nodes": len(self.tree_of_thoughts.nodes),
                "max_depth": self.config.max_tree_depth,
                "pruning_threshold": self.config.pruning_threshold
            }
        }
        
        return stats
    
    def reset_framework(self):
        """Reset framework to initial state."""
        logger.info("Resetting framework to initial state")
        
        # Clear memories
        self.memory_manager.memories.clear()
        self.memory_manager.memory_index.clear()
        self.working_memory.clear_working_memory()
        
        # Clear tree
        self.tree_of_thoughts.nodes.clear()
        self.tree_of_thoughts.root_id = None
        
        # Clear histories
        self.reward_model.reward_history.clear()
        self.constitutional_improver.improvement_history.clear()
        self.performance_monitor.metrics_history.clear()
        
        # Reset agent expertise scores
        for agent in self.agents.values():
            agent.expertise_score = 1.0
            agent.collaboration_history.clear()
        
        logger.info("Framework reset completed")


def create_framework_from_config(config_path: str, tools: Dict[str, Tool] = None) -> EnhancedRSearchFramework:
    """Create framework instance from configuration file."""
    config_data = IOManager.load_json(config_path)
    
    # Convert dict to config object
    config = EnhancedRSearchConfig()
    for key, value in config_data.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return EnhancedRSearchFramework(config, tools)


def quick_demo(question: str = None) -> Dict[str, Any]:
    """Quick demonstration of the framework."""
    if question is None:
        question = "When was Countrywide bought by the company that bought FleetBoston Financial?"
    
    # Create simple config
    config = EnhancedRSearchConfig(
        policy_model_name="microsoft/DialoGPT-medium",
        cross_family_model_name="microsoft/DialoGPT-small",
        critic_model_name="microsoft/DialoGPT-small",
        max_tree_depth=3,
        batch_size=1
    )
    
    # Initialize framework
    framework = EnhancedRSearchFramework(config)
    
    # Run inference
    result = framework.enhanced_inference(question)
    
    return result