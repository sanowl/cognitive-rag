"""
Training components for Enhanced R-Search Framework.
"""

import logging
import numpy as np
import tensorflow as tf
from typing import Dict, List, Any, Tuple
from transformers import TFAutoModelForCausalLM

from .config import EnhancedRSearchConfig
from .utils import PerformanceMonitor

logger = logging.getLogger(__name__)


class GRPOTrainer:
    """Group Relative Policy Optimization trainer for R-Search."""
    
    def __init__(self, config: EnhancedRSearchConfig, policy_model, reward_model, framework):
        self.config = config
        self.policy_model = policy_model
        self.reward_model = reward_model
        self.framework = framework
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
        
        # Reference model (frozen copy of initial policy)
        self.reference_model = TFAutoModelForCausalLM.from_pretrained(config.policy_model_name)
        self.reference_model.trainable = False
        
        self.training_history = []
        self.performance_monitor = PerformanceMonitor()
    
    def train(self, train_data: List[Dict[str, str]], validation_data: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Train the R-Search framework using GRPO."""
        logger.info(f"Starting GRPO training on {len(train_data)} examples...")
        self.performance_monitor.start_monitoring()
        
        questions = [item['question'] for item in train_data]
        answers = [item['answer'] for item in train_data]
        
        num_batches = len(questions) // self.config.batch_size
        
        training_metrics = {
            'epoch_losses': [],
            'epoch_rewards': [],
            'validation_scores': []
        }
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0
            epoch_reward = 0
            
            logger.info(f"Starting epoch {epoch+1}/{self.config.num_epochs}")
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.config.batch_size
                end_idx = start_idx + self.config.batch_size
                
                batch_questions = questions[start_idx:end_idx]
                batch_answers = answers[start_idx:end_idx]
                
                # Training step
                step_metrics = self.train_step(batch_questions, batch_answers)
                
                epoch_loss += step_metrics['loss']
                epoch_reward += step_metrics['reward']
                
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch+1}, Batch {batch_idx+1}/{num_batches}, "
                              f"Loss: {step_metrics['loss']:.4f}, "
                              f"Reward: {step_metrics['reward']:.4f}")
                
                # Record performance metrics
                self.performance_monitor.record_metrics({
                    'batch_loss': step_metrics['loss'],
                    'batch_reward': step_metrics['reward'],
                    'learning_rate': self.config.learning_rate,
                    'kl_coefficient': self.config.kl_coefficient
                })
            
            avg_loss = epoch_loss / num_batches
            avg_reward = epoch_reward / num_batches
            
            training_metrics['epoch_losses'].append(avg_loss)
            training_metrics['epoch_rewards'].append(avg_reward)
            
            # Validation
            if validation_data:
                val_score = self.validate(validation_data)
                training_metrics['validation_scores'].append(val_score)
                logger.info(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, "
                          f"Reward: {avg_reward:.4f}, Validation: {val_score:.4f}")
            else:
                logger.info(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Reward: {avg_reward:.4f}")
            
            # Store epoch metrics
            self.training_history.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'reward': avg_reward,
                'validation_score': training_metrics['validation_scores'][-1] if validation_data else None
            })
        
        training_summary = {
            'total_epochs': self.config.num_epochs,
            'final_loss': training_metrics['epoch_losses'][-1],
            'final_reward': training_metrics['epoch_rewards'][-1],
            'best_validation': max(training_metrics['validation_scores']) if validation_data else None,
            'training_metrics': training_metrics,
            'performance_stats': self.performance_monitor.get_summary_statistics()
        }
        
        logger.info("GRPO training completed successfully")
        return training_summary
    
    def train_step(self, questions: List[str], gold_answers: List[str]) -> Dict[str, float]:
        """Single training step using GRPO."""
        batch_losses = []
        batch_rewards = []
        
        for question, gold_answer in zip(questions, gold_answers):
            # Generate trajectory using framework
            try:
                result = self.framework.enhanced_inference(question)
                trajectory = result['reasoning_trace']
                
                # Compute rewards
                reward_dict = self.reward_model.compute_comprehensive_reward(
                    question, trajectory, gold_answer, self.framework.memory_manager
                )
                total_reward = reward_dict['total_reward']
                batch_rewards.append(total_reward)
                
                # Prepare inputs for gradient computation
                full_input = self._prepare_training_input(question, result)
                inputs = self.framework.policy_tokenizer.encode(
                    full_input, 
                    return_tensors='tf', 
                    max_length=self.config.max_sequence_length, 
                    truncation=True,
                    padding=True
                )
                
                attention_mask = tf.ones_like(inputs)
                loss_mask = self._create_loss_mask(inputs, result)
                
                with tf.GradientTape() as tape:
                    # Get policy logprobs
                    policy_outputs = self.policy_model(inputs, attention_mask=attention_mask)
                    policy_logits = policy_outputs.logits
                    policy_logprobs = tf.nn.log_softmax(policy_logits, axis=-1)
                    
                    # Get reference logprobs
                    ref_outputs = self.reference_model(inputs, attention_mask=attention_mask)
                    ref_logits = ref_outputs.logits
                    ref_logprobs = tf.nn.log_softmax(ref_logits, axis=-1)
                    
                    # Compute KL divergence
                    kl_div = self._compute_kl_divergence(policy_logprobs, ref_logprobs, loss_mask)
                    kl_penalty = self.config.kl_coefficient * kl_div
                    
                    # GRPO loss: -reward + KL penalty
                    loss = -total_reward + tf.reduce_mean(kl_penalty)
                    batch_losses.append(loss)
                
                # Compute gradients and update
                gradients = tape.gradient(loss, self.policy_model.trainable_variables)
                
                # Clip gradients
                gradients = [tf.clip_by_norm(grad, 1.0) if grad is not None else grad for grad in gradients]
                
                self.optimizer.apply_gradients(zip(gradients, self.policy_model.trainable_variables))
                
            except Exception as e:
                logger.warning(f"Training step failed for question: {e}")
                batch_losses.append(tf.constant(0.0))
                batch_rewards.append(0.0)
        
        return {
            'loss': np.mean([loss.numpy() if hasattr(loss, 'numpy') else float(loss) for loss in batch_losses]),
            'reward': np.mean(batch_rewards)
        }
    
    def validate(self, validation_data: List[Dict[str, str]]) -> float:
        """Validate the model on validation data."""
        total_score = 0
        
        for item in validation_data[:min(50, len(validation_data))]:  # Limit validation size
            try:
                result = self.framework.enhanced_inference(item['question'])
                reward_dict = self.reward_model.compute_comprehensive_reward(
                    item['question'], result['reasoning_trace'], item['answer'], 
                    self.framework.memory_manager
                )
                total_score += reward_dict['total_reward']
            except Exception as e:
                logger.warning(f"Validation failed for question: {e}")
                total_score += 0.0
        
        return total_score / min(50, len(validation_data))
    
    def _prepare_training_input(self, question: str, result: Dict[str, Any]) -> str:
        """Prepare input text for training."""
        trajectory = result.get('reasoning_trace', {})
        reasoning_text = self._format_reasoning_trace(trajectory)
        
        return f"Question: {question}\n{reasoning_text}\nAnswer: {result.get('improved_answer', '')}"
    
    def _format_reasoning_trace(self, trajectory: Dict[str, Any]) -> str:
        """Format reasoning trace for training input."""
        formatted_parts = []
        
        if 'plan' in trajectory:
            formatted_parts.append(f"Plan: {trajectory['plan']}")
        
        reasoning_results = trajectory.get('reasoning_results', {})
        for task_id, task_result in reasoning_results.items():
            formatted_parts.append(f"Step: {task_result.get('result', '')}")
        
        if 'final_result' in trajectory:
            formatted_parts.append(f"Synthesis: {trajectory['final_result'].get('final_answer', '')}")
        
        return "\n".join(formatted_parts)
    
    def _create_loss_mask(self, input_ids: tf.Tensor, result: Dict[str, Any]) -> tf.Tensor:
        """Create mask for loss computation."""
        # Simple masking strategy - mask all tokens (can be enhanced)
        mask = tf.ones_like(input_ids, dtype=tf.float32)
        return mask
    
    def _compute_kl_divergence(self, policy_logprobs: tf.Tensor, ref_logprobs: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """Compute KL divergence between policy and reference model."""
        kl_div = policy_logprobs - ref_logprobs
        masked_kl = kl_div * mask
        return tf.reduce_sum(masked_kl, axis=-1)
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get training statistics and metrics."""
        if not self.training_history:
            return {}
        
        losses = [epoch['loss'] for epoch in self.training_history]
        rewards = [epoch['reward'] for epoch in self.training_history]
        
        stats = {
            'total_epochs_trained': len(self.training_history),
            'final_loss': losses[-1],
            'final_reward': rewards[-1],
            'best_loss': min(losses),
            'best_reward': max(rewards),
            'loss_trend': 'decreasing' if losses[-1] < losses[0] else 'increasing',
            'reward_trend': 'increasing' if rewards[-1] > rewards[0] else 'decreasing',
            'performance_metrics': self.performance_monitor.get_summary_statistics()
        }
        
        return stats


class PPOTrainer:
    """Proximal Policy Optimization trainer for R-Search."""
    
    def __init__(self, config: EnhancedRSearchConfig, policy_model, reward_model, framework):
        self.config = config
        self.policy_model = policy_model
        self.reward_model = reward_model
        self.framework = framework
        
        # PPO specific parameters
        self.clip_ratio = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        
        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
        self.value_optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
        
        # Value network (simple MLP for demonstration)
        self.value_network = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        self.training_history = []
    
    def train(self, train_data: List[Dict[str, str]], validation_data: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Train using PPO algorithm."""
        logger.info(f"Starting PPO training on {len(train_data)} examples...")
        
        training_metrics = {
            'epoch_losses': [],
            'epoch_rewards': [],
            'policy_losses': [],
            'value_losses': []
        }
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"PPO Epoch {epoch+1}/{self.config.num_epochs}")
            
            # Collect rollouts
            rollouts = self._collect_rollouts(train_data[:self.config.batch_size])
            
            # Update policy and value networks
            policy_loss, value_loss = self._update_networks(rollouts)
            
            avg_reward = np.mean([rollout['reward'] for rollout in rollouts])
            
            training_metrics['epoch_losses'].append(policy_loss + value_loss)
            training_metrics['epoch_rewards'].append(avg_reward)
            training_metrics['policy_losses'].append(policy_loss)
            training_metrics['value_losses'].append(value_loss)
            
            logger.info(f"Epoch {epoch+1} - Policy Loss: {policy_loss:.4f}, "
                       f"Value Loss: {value_loss:.4f}, Reward: {avg_reward:.4f}")
            
            self.training_history.append({
                'epoch': epoch + 1,
                'policy_loss': policy_loss,
                'value_loss': value_loss,
                'reward': avg_reward
            })
        
        return {
            'total_epochs': self.config.num_epochs,
            'training_metrics': training_metrics,
            'final_reward': training_metrics['epoch_rewards'][-1]
        }
    
    def _collect_rollouts(self, data: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Collect rollouts for PPO training."""
        rollouts = []
        
        for item in data:
            try:
                result = self.framework.enhanced_inference(item['question'])
                
                reward_dict = self.reward_model.compute_comprehensive_reward(
                    item['question'], result['reasoning_trace'], item['answer'],
                    self.framework.memory_manager
                )
                
                rollout = {
                    'question': item['question'],
                    'response': result['improved_answer'],
                    'reward': reward_dict['total_reward'],
                    'trajectory': result['reasoning_trace']
                }
                
                rollouts.append(rollout)
                
            except Exception as e:
                logger.warning(f"Failed to collect rollout: {e}")
        
        return rollouts
    
    def _update_networks(self, rollouts: List[Dict[str, Any]]) -> Tuple[float, float]:
        """Update policy and value networks using PPO."""
        # Simplified PPO update (in practice, this would be more complex)
        
        total_policy_loss = 0
        total_value_loss = 0
        
        for rollout in rollouts:
            # Compute value estimates (simplified)
            question_embedding = self._encode_question(rollout['question'])
            value_estimate = self.value_network(question_embedding)
            
            # PPO policy update (simplified)
            with tf.GradientTape() as policy_tape:
                # Simplified policy loss
                policy_loss = -rollout['reward']  # Negative because we want to maximize reward
                total_policy_loss += policy_loss
            
            # Value network update
            with tf.GradientTape() as value_tape:
                predicted_value = self.value_network(question_embedding)
                value_loss = tf.keras.losses.MSE(rollout['reward'], predicted_value)
                total_value_loss += value_loss
            
            # Apply gradients (simplified)
            try:
                policy_gradients = policy_tape.gradient(policy_loss, self.policy_model.trainable_variables)
                if policy_gradients and any(grad is not None for grad in policy_gradients):
                    self.policy_optimizer.apply_gradients(
                        [(grad, var) for grad, var in zip(policy_gradients, self.policy_model.trainable_variables) 
                         if grad is not None]
                    )
                
                value_gradients = value_tape.gradient(value_loss, self.value_network.trainable_variables)
                if value_gradients and any(grad is not None for grad in value_gradients):
                    self.value_optimizer.apply_gradients(
                        [(grad, var) for grad, var in zip(value_gradients, self.value_network.trainable_variables)
                         if grad is not None]
                    )
            except Exception as e:
                logger.warning(f"Gradient update failed: {e}")
        
        avg_policy_loss = total_policy_loss / len(rollouts) if rollouts else 0
        avg_value_loss = total_value_loss / len(rollouts) if rollouts else 0
        
        return float(avg_policy_loss), float(avg_value_loss)
    
    def _encode_question(self, question: str) -> tf.Tensor:
        """Encode question for value network (simplified)."""
        # Simple encoding - in practice, use proper embeddings
        tokens = self.framework.policy_tokenizer.encode(question, max_length=128, truncation=True, padding=True)
        # Convert to fixed-size representation
        embedding = tf.reduce_mean(tf.cast(tokens, tf.float32), axis=0, keepdims=True)
        return tf.expand_dims(embedding, 0)


class CurriculumLearning:
    """Curriculum learning for gradual difficulty increase."""
    
    def __init__(self, config: EnhancedRSearchConfig):
        self.config = config
        self.current_difficulty = "easy"
        self.difficulty_levels = config.difficulty_levels
        self.performance_threshold = 0.7
        self.difficulty_index = 0
    
    def should_increase_difficulty(self, recent_performance: List[float]) -> bool:
        """Determine if difficulty should be increased."""
        if len(recent_performance) < 10:
            return False
        
        avg_performance = np.mean(recent_performance[-10:])
        return avg_performance > self.performance_threshold
    
    def get_current_difficulty_data(self, all_data: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Filter data based on current difficulty level."""
        # Simple difficulty classification based on question length
        difficulty_thresholds = {
            "easy": 50,    # Characters
            "medium": 100,
            "hard": 200,
            "expert": float('inf')
        }
        
        max_length = difficulty_thresholds[self.current_difficulty]
        min_length = difficulty_thresholds.get(
            self.difficulty_levels[max(0, self.difficulty_index - 1)], 0
        ) if self.difficulty_index > 0 else 0
        
        filtered_data = []
        for item in all_data:
            question_length = len(item['question'])
            if min_length < question_length <= max_length:
                filtered_data.append(item)
        
        return filtered_data
    
    def update_difficulty(self, performance_scores: List[float]):
        """Update difficulty level based on performance."""
        if self.should_increase_difficulty(performance_scores):
            if self.difficulty_index < len(self.difficulty_levels) - 1:
                self.difficulty_index += 1
                self.current_difficulty = self.difficulty_levels[self.difficulty_index]
                logger.info(f"Difficulty increased to: {self.current_difficulty}")


def create_trainer(trainer_type: str, config: EnhancedRSearchConfig, 
                  policy_model, reward_model, framework):
    """Factory function to create trainers."""
    if trainer_type.lower() == "grpo":
        return GRPOTrainer(config, policy_model, reward_model, framework)
    elif trainer_type.lower() == "ppo":
        return PPOTrainer(config, policy_model, reward_model, framework)
    else:
        raise ValueError(f"Unknown trainer type: {trainer_type}")