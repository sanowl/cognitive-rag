"""
Multi-agent system for Enhanced R-Search Framework.
"""

import re
import logging
from typing import Dict, List, Any, Optional
from .config import AgentRole, EnhancedRSearchConfig
from .tools import Tool

logger = logging.getLogger(__name__)


class Agent:
    """Individual agent in the multi-agent system."""
    
    def __init__(self, role: AgentRole, config: EnhancedRSearchConfig, model, tokenizer, tools: Dict[str, Tool]):
        self.role = role
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.tools = tools
        self.expertise_score = 1.0
        self.collaboration_history = []
    
    def generate_response(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Generate response based on agent's role and expertise."""
        role_prompt = self._get_role_prompt()
        full_prompt = f"{role_prompt}\n\n{prompt}"
        
        inputs = self.tokenizer.encode(full_prompt, return_tensors='tf', max_length=512, truncation=True)
        
        outputs = self.model.generate(
            inputs,
            max_length=inputs.shape[1] + 150,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    def _get_role_prompt(self) -> str:
        """Get role-specific prompt."""
        prompts = {
            AgentRole.PLANNER: "You are a strategic planner. Break down complex problems into manageable steps and create execution plans.",
            AgentRole.SEARCHER: "You are an information retrieval specialist. Find relevant information efficiently and accurately.",
            AgentRole.REASONER: "You are a logical reasoning expert. Analyze information and draw valid conclusions.",
            AgentRole.VERIFIER: "You are a verification specialist. Check answers for accuracy and consistency.",
            AgentRole.CRITIC: "You are a constructive critic. Identify flaws and suggest improvements.",
            AgentRole.SYNTHESIZER: "You are an information synthesizer. Combine multiple sources into coherent answers.",
            AgentRole.MEMORY_MANAGER: "You manage episodic memory. Store and retrieve relevant past experiences."
        }
        return prompts.get(self.role, "You are a helpful AI assistant.")
    
    def select_tool(self, task_description: str) -> Optional[str]:
        """Select appropriate tool for the task."""
        task_lower = task_description.lower()
        
        if any(word in task_lower for word in ["search", "find", "lookup", "information"]):
            return "search"
        elif any(word in task_lower for word in ["calculate", "math", "compute", "number"]):
            return "calculator"
        elif any(word in task_lower for word in ["relationship", "connected", "entity", "graph"]):
            return "knowledge_graph"
        
        return None
    
    def update_expertise(self, success: bool, task_type: str):
        """Update agent's expertise based on performance."""
        if success:
            self.expertise_score = min(2.0, self.expertise_score + 0.1)
        else:
            self.expertise_score = max(0.1, self.expertise_score - 0.05)


class MultiAgentCoordinator:
    """Coordinates multiple agents in collaborative reasoning."""
    
    def __init__(self, config: EnhancedRSearchConfig, agents: Dict[AgentRole, Agent]):
        self.config = config
        self.agents = agents
        self.coordination_history = []
        self.current_task_assignment = {}
    
    def coordinate_reasoning(self, question: str, tree) -> Dict[str, Any]:
        """Coordinate multi-agent reasoning process."""
        # Phase 1: Planning
        planner = self.agents[AgentRole.PLANNER]
        plan = planner.generate_response(f"Create a step-by-step plan to answer: {question}")
        
        # Phase 2: Information gathering
        searcher = self.agents[AgentRole.SEARCHER]
        search_strategy = searcher.generate_response(f"Plan information gathering for: {question}")
        
        # Phase 3: Reasoning with collaboration
        reasoning_results = self._collaborative_reasoning(question, plan, tree)
        
        # Phase 4: Verification and synthesis
        final_result = self._verify_and_synthesize(question, reasoning_results)
        
        return {
            "plan": plan,
            "search_strategy": search_strategy,
            "reasoning_results": reasoning_results,
            "final_result": final_result,
            "agents_involved": list(self.current_task_assignment.keys())
        }
    
    def _collaborative_reasoning(self, question: str, plan: str, tree) -> Dict[str, Any]:
        """Execute collaborative reasoning across agents."""
        results = {}
        
        # Assign tasks to different agents
        tasks = self._decompose_plan(plan)
        
        for i, task in enumerate(tasks):
            best_agent = self._select_best_agent(task)
            self.current_task_assignment[f"task_{i}"] = best_agent.role
            
            # Create thought node for this task
            if tree.root_id:
                node_id = tree.expand_node(
                    tree.root_id, 
                    [f"Task {i}: {task}"], 
                    best_agent.role
                )[0]
            else:
                node_id = f"task_{i}_node"
            
            # Execute task
            task_result = best_agent.generate_response(f"Execute: {task}\nContext: {question}")
            results[f"task_{i}"] = {
                "task": task,
                "agent": best_agent.role,
                "result": task_result,
                "node_id": node_id
            }
            
            # Evaluate node
            if tree.root_id and node_id in tree.nodes:
                score = tree.evaluate_node(node_id, self.agents[AgentRole.VERIFIER])
                tree.nodes[node_id].score = score
        
        return results
    
    def _decompose_plan(self, plan: str) -> List[str]:
        """Decompose plan into individual tasks."""
        lines = plan.split('\n')
        tasks = []
        for line in lines:
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('1.') or 'step' in line.lower()):
                task = re.sub(r'^[-\d\.\s]*', '', line).strip()
                if task:
                    tasks.append(task)
        
        return tasks[:5]  # Limit to 5 tasks
    
    def _select_best_agent(self, task: str) -> Agent:
        """Select the best agent for a specific task."""
        task_lower = task.lower()
        
        if any(word in task_lower for word in ["search", "find", "gather", "retrieve"]):
            return self.agents[AgentRole.SEARCHER]
        elif any(word in task_lower for word in ["reason", "analyze", "conclude", "infer"]):
            return self.agents[AgentRole.REASONER]
        elif any(word in task_lower for word in ["verify", "check", "validate", "confirm"]):
            return self.agents[AgentRole.VERIFIER]
        elif any(word in task_lower for word in ["combine", "synthesize", "integrate", "merge"]):
            return self.agents[AgentRole.SYNTHESIZER]
        else:
            # Default to reasoner
            return self.agents[AgentRole.REASONER]
    
    def _verify_and_synthesize(self, question: str, reasoning_results: Dict[str, Any]) -> Dict[str, Any]:
        """Verify results and synthesize final answer."""
        verifier = self.agents[AgentRole.VERIFIER]
        synthesizer = self.agents[AgentRole.SYNTHESIZER]
        
        # Collect all task results
        all_results = [result["result"] for result in reasoning_results.values()]
        results_text = "\n".join(f"Result {i}: {result}" for i, result in enumerate(all_results))
        
        # Verification
        verification_prompt = f"Verify these reasoning results for question '{question}':\n{results_text}"
        verification = verifier.generate_response(verification_prompt)
        
        # Synthesis
        synthesis_prompt = f"Synthesize final answer for '{question}' from:\n{results_text}\nVerification: {verification}"
        final_answer = synthesizer.generate_response(synthesis_prompt)
        
        return {
            "verification": verification,
            "final_answer": final_answer,
            "confidence": self._estimate_confidence(verification),
            "sources_used": len(reasoning_results)
        }
    
    def _estimate_confidence(self, verification: str) -> float:
        """Estimate confidence from verification text."""
        verification_lower = verification.lower()
        
        if any(word in verification_lower for word in ["confident", "certain", "verified", "correct"]):
            return 0.9
        elif any(word in verification_lower for word in ["likely", "probable", "seems"]):
            return 0.7
        elif any(word in verification_lower for word in ["uncertain", "unclear", "doubt"]):
            return 0.4
        else:
            return 0.6