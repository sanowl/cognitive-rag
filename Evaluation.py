"""
Evaluation metrics and benchmarks for Enhanced R-Search Framework.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import re

from .utils import TextProcessor, IOManager

logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """Comprehensive evaluation metrics for R-Search framework."""
    
    def __init__(self):
        self.text_processor = TextProcessor()
    
    def compute_answer_accuracy(self, predictions: List[str], gold_answers: List[str]) -> Dict[str, float]:
        """Compute answer accuracy metrics."""
        if len(predictions) != len(gold_answers):
            raise ValueError("Predictions and gold answers must have same length")
        
        f1_scores = []
        exact_matches = []
        
        for pred, gold in zip(predictions, gold_answers):
            # F1 Score
            f1 = self.text_processor.compute_f1_score(pred, gold)
            f1_scores.append(f1)
            
            # Exact Match
            em = 1.0 if pred.strip().lower() == gold.strip().lower() else 0.0
            exact_matches.append(em)
        
        return {
            "f1_score": np.mean(f1_scores),
            "exact_match": np.mean(exact_matches),
            "f1_std": np.std(f1_scores),
            "em_std": np.std(exact_matches)
        }
    
    def compute_reasoning_quality(self, reasoning_traces: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute reasoning quality metrics."""
        if not reasoning_traces:
            return {}
        
        consistency_scores = []
        complexity_scores = []
        coherence_scores = []
        
        for trace in reasoning_traces:
            # Consistency: Check for contradictions
            consistency = self._evaluate_consistency(trace)
            consistency_scores.append(consistency)
            
            # Complexity: Number of reasoning steps
            complexity = self._evaluate_complexity(trace)
            complexity_scores.append(complexity)
            
            # Coherence: Flow of reasoning
            coherence = self._evaluate_coherence(trace)
            coherence_scores.append(coherence)
        
        return {
            "consistency": np.mean(consistency_scores),
            "complexity": np.mean(complexity_scores),
            "coherence": np.mean(coherence_scores),
            "consistency_std": np.std(consistency_scores),
            "complexity_std": np.std(complexity_scores),
            "coherence_std": np.std(coherence_scores)
        }
    
    def compute_efficiency_metrics(self, performance_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute efficiency and resource usage metrics."""
        if not performance_data:
            return {}
        
        response_times = [d.get('response_time', 0) for d in performance_data]
        token_counts = [d.get('token_count', 0) for d in performance_data]
        tree_sizes = [d.get('tree_size', 0) for d in performance_data]
        agent_counts = [d.get('agents_used', 0) for d in performance_data]
        
        return {
            "avg_response_time": np.mean(response_times),
            "avg_token_count": np.mean(token_counts),
            "avg_tree_size": np.mean(tree_sizes),
            "avg_agents_used": np.mean(agent_counts),
            "response_time_std": np.std(response_times),
            "efficiency_score": self._compute_efficiency_score(performance_data)
        }
    
    def compute_novelty_metrics(self, reasoning_patterns: List[str]) -> Dict[str, float]:
        """Compute novelty and creativity metrics."""
        if not reasoning_patterns:
            return {}
        
        # Compute pattern diversity
        unique_patterns = set(reasoning_patterns)
        diversity_ratio = len(unique_patterns) / len(reasoning_patterns)
        
        # Compute pattern entropy
        pattern_counts = defaultdict(int)
        for pattern in reasoning_patterns:
            pattern_counts[pattern] += 1
        
        total_patterns = len(reasoning_patterns)
        entropy = 0
        for count in pattern_counts.values():
            prob = count / total_patterns
            if prob > 0:
                entropy -= prob * np.log2(prob)
        
        return {
            "pattern_diversity": diversity_ratio,
            "pattern_entropy": entropy,
            "unique_patterns": len(unique_patterns),
            "total_patterns": len(reasoning_patterns)
        }
    
    def compute_constitutional_compliance(self, critique_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute constitutional AI compliance metrics."""
        if not critique_results:
            return {}
        
        quality_scores = [r.get('quality_score', 0) for r in critique_results]
        improvement_flags = [r.get('needs_improvement', True) for r in critique_results]
        
        compliance_rate = 1.0 - np.mean(improvement_flags)
        avg_quality = np.mean(quality_scores)
        
        return {
            "compliance_rate": compliance_rate,
            "average_quality": avg_quality,
            "quality_std": np.std(quality_scores),
            "improvement_needed_ratio": np.mean(improvement_flags)
        }
    
    def _evaluate_consistency(self, trace: Dict[str, Any]) -> float:
        """Evaluate consistency of reasoning trace."""
        reasoning_results = trace.get('reasoning_results', {})
        if len(reasoning_results) < 2:
            return 1.0
        
        # Check for contradictory statements
        contradiction_markers = ["however", "but", "contradicts", "inconsistent", "disagree"]
        
        contradiction_count = 0
        total_statements = len(reasoning_results)
        
        for result in reasoning_results.values():
            text = result.get('result', '').lower()
            contradiction_count += sum(1 for marker in contradiction_markers if marker in text)
        
        # Higher consistency = fewer contradictions
        consistency = max(0.0, 1.0 - (contradiction_count / total_statements))
        return consistency
    
    def _evaluate_complexity(self, trace: Dict[str, Any]) -> float:
        """Evaluate complexity of reasoning trace."""
        reasoning_results = trace.get('reasoning_results', {})
        num_steps = len(reasoning_results)
        
        # Normalize complexity score (0-1 scale)
        # More steps = higher complexity (up to a point)
        complexity = min(1.0, num_steps / 10.0)
        return complexity
    
    def _evaluate_coherence(self, trace: Dict[str, Any]) -> float:
        """Evaluate coherence of reasoning flow."""
        reasoning_results = trace.get('reasoning_results', {})
        if len(reasoning_results) < 2:
            return 1.0
        
        # Simple coherence metric based on logical flow indicators
        flow_indicators = ["therefore", "thus", "consequently", "because", "since", "given that"]
        
        coherence_count = 0
        total_statements = len(reasoning_results)
        
        for result in reasoning_results.values():
            text = result.get('result', '').lower()
            coherence_count += sum(1 for indicator in flow_indicators if indicator in text)
        
        coherence = min(1.0, coherence_count / total_statements)
        return coherence
    
    def _compute_efficiency_score(self, performance_data: List[Dict[str, Any]]) -> float:
        """Compute overall efficiency score."""
        if not performance_data:
            return 0.0
        
        # Weighted combination of efficiency factors
        scores = []
        for data in performance_data:
            time_score = 1.0 / (1.0 + data.get('response_time', 1.0) / 10.0)  # Normalize by 10s
            token_score = 1.0 / (1.0 + data.get('token_count', 100) / 1000.0)  # Normalize by 1000 tokens
            tree_score = 1.0 / (1.0 + data.get('tree_size', 10) / 50.0)  # Normalize by 50 nodes
            
            efficiency = (time_score + token_score + tree_score) / 3.0
            scores.append(efficiency)
        
        return np.mean(scores)


class BenchmarkSuite:
    """Benchmark suite for evaluating R-Search on standard datasets."""
    
    def __init__(self, framework):
        self.framework = framework
        self.metrics = EvaluationMetrics()
        self.benchmark_results = {}
    
    def run_hotpot_qa_benchmark(self, data: List[Dict[str, str]], sample_size: int = 100) -> Dict[str, Any]:
        """Run evaluation on HotpotQA dataset."""
        logger.info(f"Running HotpotQA benchmark on {min(sample_size, len(data))} examples")
        
        sample_data = data[:sample_size] if len(data) > sample_size else data
        results = self._run_benchmark("HotpotQA", sample_data)
        
        self.benchmark_results["HotpotQA"] = results
        return results
    
    def run_musique_benchmark(self, data: List[Dict[str, str]], sample_size: int = 100) -> Dict[str, Any]:
        """Run evaluation on MuSiQue dataset."""
        logger.info(f"Running MuSiQue benchmark on {min(sample_size, len(data))} examples")
        
        sample_data = data[:sample_size] if len(data) > sample_size else data
        results = self._run_benchmark("MuSiQue", sample_data)
        
        self.benchmark_results["MuSiQue"] = results
        return results
    
    def run_natural_questions_benchmark(self, data: List[Dict[str, str]], sample_size: int = 100) -> Dict[str, Any]:
        """Run evaluation on Natural Questions dataset."""
        logger.info(f"Running Natural Questions benchmark on {min(sample_size, len(data))} examples")
        
        sample_data = data[:sample_size] if len(data) > sample_size else data
        results = self._run_benchmark("NaturalQuestions", sample_data)
        
        self.benchmark_results["NaturalQuestions"] = results
        return results
    
    def run_custom_benchmark(self, name: str, data: List[Dict[str, str]], 
                           sample_size: int = None) -> Dict[str, Any]:
        """Run evaluation on custom dataset."""
        sample_size = sample_size or len(data)
        logger.info(f"Running {name} benchmark on {min(sample_size, len(data))} examples")
        
        sample_data = data[:sample_size] if len(data) > sample_size else data
        results = self._run_benchmark(name, sample_data)
        
        self.benchmark_results[name] = results
        return results
    
    def _run_benchmark(self, benchmark_name: str, data: List[Dict[str, str]]) -> Dict[str, Any]:
        """Run benchmark evaluation on given data."""
        predictions = []
        gold_answers = []
        reasoning_traces = []
        critique_results = []
        performance_data = []
        reasoning_patterns = []
        
        for i, item in enumerate(data):
            logger.info(f"Processing {benchmark_name} example {i+1}/{len(data)}")
            
            try:
                # Track performance
                import time
                start_time = time.time()
                
                # Run inference
                result = self.framework.enhanced_inference(item['question'])
                
                end_time = time.time()
                response_time = end_time - start_time
                
                # Collect data
                predictions.append(result['improved_answer'])
                gold_answers.append(item['answer'])
                reasoning_traces.append(result['reasoning_trace'])
                critique_results.append(result['critique'])
                
                # Extract reasoning pattern
                pattern = self.framework._extract_pattern(result['reasoning_trace'])
                reasoning_patterns.append(pattern)
                
                # Performance data
                performance_data.append({
                    'response_time': response_time,
                    'token_count': len(result['improved_answer'].split()),
                    'tree_size': result['tree_nodes_explored'],
                    'agents_used': len(result['reasoning_trace'].get('agents_involved', []))
                })
                
            except Exception as e:
                logger.error(f"Failed to process example {i+1}: {e}")
                predictions.append("")
                gold_answers.append(item['answer'])
                reasoning_traces.append({})
                critique_results.append({'quality_score': 0.0, 'needs_improvement': True})
                reasoning_patterns.append("error")
                performance_data.append({
                    'response_time': 0,
                    'token_count': 0,
                    'tree_size': 0,
                    'agents_used': 0
                })
        
        # Compute metrics
        accuracy_metrics = self.metrics.compute_answer_accuracy(predictions, gold_answers)
        reasoning_metrics = self.metrics.compute_reasoning_quality(reasoning_traces)
        efficiency_metrics = self.metrics.compute_efficiency_metrics(performance_data)
        novelty_metrics = self.metrics.compute_novelty_metrics(reasoning_patterns)
        constitutional_metrics = self.metrics.compute_constitutional_compliance(critique_results)
        
        # Compile results
        benchmark_results = {
            "benchmark_name": benchmark_name,
            "total_examples": len(data),
            "accuracy_metrics": accuracy_metrics,
            "reasoning_metrics": reasoning_metrics,
            "efficiency_metrics": efficiency_metrics,
            "novelty_metrics": novelty_metrics,
            "constitutional_metrics": constitutional_metrics,
            "detailed_results": {
                "predictions": predictions,
                "gold_answers": gold_answers,
                "reasoning_traces": reasoning_traces,
                "performance_data": performance_data
            }
        }
        
        return benchmark_results
    
    def generate_comparison_report(self, baseline_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate comprehensive comparison report."""
        report = {
            "summary": self._generate_summary(),
            "detailed_metrics": self.benchmark_results,
            "cross_benchmark_analysis": self._analyze_cross_benchmark_performance()
        }
        
        if baseline_results:
            report["baseline_comparison"] = self._compare_with_baseline(baseline_results)
        
        return report
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary of all benchmark results."""
        if not self.benchmark_results:
            return {}
        
        # Aggregate metrics across benchmarks
        all_f1_scores = []
        all_em_scores = []
        all_consistency_scores = []
        all_efficiency_scores = []
        
        for benchmark, results in self.benchmark_results.items():
            accuracy = results.get('accuracy_metrics', {})
            reasoning = results.get('reasoning_metrics', {})
            efficiency = results.get('efficiency_metrics', {})
            
            all_f1_scores.append(accuracy.get('f1_score', 0))
            all_em_scores.append(accuracy.get('exact_match', 0))
            all_consistency_scores.append(reasoning.get('consistency', 0))
            all_efficiency_scores.append(efficiency.get('efficiency_score', 0))
        
        summary = {
            "benchmarks_run": list(self.benchmark_results.keys()),
            "average_f1": np.mean(all_f1_scores) if all_f1_scores else 0,
            "average_em": np.mean(all_em_scores) if all_em_scores else 0,
            "average_consistency": np.mean(all_consistency_scores) if all_consistency_scores else 0,
            "average_efficiency": np.mean(all_efficiency_scores) if all_efficiency_scores else 0,
            "best_benchmark": max(self.benchmark_results.items(), 
                                key=lambda x: x[1].get('accuracy_metrics', {}).get('f1_score', 0))[0] if self.benchmark_results else None,
            "total_examples_evaluated": sum(r.get('total_examples', 0) for r in self.benchmark_results.values())
        }
        
        return summary
    
    def _analyze_cross_benchmark_performance(self) -> Dict[str, Any]:
        """Analyze performance patterns across benchmarks."""
        if len(self.benchmark_results) < 2:
            return {}
        
        # Compare performance across benchmarks
        benchmark_scores = {}
        for name, results in self.benchmark_results.items():
            benchmark_scores[name] = results.get('accuracy_metrics', {}).get('f1_score', 0)
        
        # Find patterns
        best_benchmark = max(benchmark_scores, key=benchmark_scores.get)
        worst_benchmark = min(benchmark_scores, key=benchmark_scores.get)
        score_variance = np.var(list(benchmark_scores.values()))
        
        analysis = {
            "benchmark_scores": benchmark_scores,
            "best_performing_benchmark": best_benchmark,
            "worst_performing_benchmark": worst_benchmark,
            "performance_variance": score_variance,
            "consistent_performance": score_variance < 0.01,  # Low variance indicates consistency
            "performance_range": max(benchmark_scores.values()) - min(benchmark_scores.values())
        }
        
        return analysis
    
    def _compare_with_baseline(self, baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare current results with baseline."""
        comparison = {}
        
        for benchmark_name in self.benchmark_results:
            if benchmark_name in baseline_results:
                current = self.benchmark_results[benchmark_name]
                baseline = baseline_results[benchmark_name]
                
                current_f1 = current.get('accuracy_metrics', {}).get('f1_score', 0)
                baseline_f1 = baseline.get('accuracy_metrics', {}).get('f1_score', 0)
                
                improvement = current_f1 - baseline_f1
                relative_improvement = (improvement / baseline_f1) * 100 if baseline_f1 > 0 else 0
                
                comparison[benchmark_name] = {
                    "current_f1": current_f1,
                    "baseline_f1": baseline_f1,
                    "absolute_improvement": improvement,
                    "relative_improvement_percent": relative_improvement,
                    "improved": improvement > 0
                }
        
        # Overall comparison
        if comparison:
            avg_improvement = np.mean([c["absolute_improvement"] for c in comparison.values()])
            benchmarks_improved = sum(1 for c in comparison.values() if c["improved"])
            
            comparison["overall"] = {
                "average_improvement": avg_improvement,
                "benchmarks_improved": benchmarks_improved,
                "total_benchmarks_compared": len(comparison),
                "improvement_rate": benchmarks_improved / len(comparison)
            }
        
        return comparison
    
    def export_results(self, filepath: str):
        """Export benchmark results to file."""
        report = self.generate_comparison_report()
        IOManager.save_json(report, filepath)
        logger.info(f"Benchmark results exported to {filepath}")


class ErrorAnalysis:
    """Analyze errors and failure modes in the framework."""
    
    def __init__(self):
        self.error_patterns = defaultdict(list)
        self.failure_modes = defaultdict(int)
    
    def analyze_failures(self, predictions: List[str], gold_answers: List[str], 
                        questions: List[str], reasoning_traces: List[Dict]) -> Dict[str, Any]:
        """Analyze failure patterns in predictions."""
        failures = []
        
        for i, (pred, gold, question, trace) in enumerate(zip(predictions, gold_answers, questions, reasoning_traces)):
            f1_score = TextProcessor.compute_f1_score(pred, gold)
            
            if f1_score < 0.5:  # Consider as failure
                failure_info = {
                    "index": i,
                    "question": question,
                    "predicted": pred,
                    "gold": gold,
                    "f1_score": f1_score,
                    "question_type": self._classify_question_type(question),
                    "error_type": self._classify_error_type(pred, gold, trace),
                    "reasoning_quality": self._assess_reasoning_quality(trace)
                }
                failures.append(failure_info)
        
        return self._compile_failure_analysis(failures)
    
    def _classify_question_type(self, question: str) -> str:
        """Classify question type for error analysis."""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["when", "date", "year"]):
            return "temporal"
        elif any(word in question_lower for word in ["who", "person"]):
            return "entity"
        elif any(word in question_lower for word in ["where", "location"]):
            return "location"
        elif any(word in question_lower for word in ["why", "because"]):
            return "causal"
        elif any(word in question_lower for word in ["how many", "count"]):
            return "quantitative"
        else:
            return "factual"
    
    def _classify_error_type(self, prediction: str, gold: str, trace: Dict) -> str:
        """Classify the type of error."""
        if not prediction.strip():
            return "no_answer"
        
        if len(prediction) < 10:
            return "incomplete_answer"
        
        # Check for hallucination patterns
        if any(word in prediction.lower() for word in ["unknown", "unclear", "cannot determine"]):
            return "knowledge_gap"
        
        # Check for reasoning errors
        reasoning_quality = self._assess_reasoning_quality(trace)
        if reasoning_quality < 0.3:
            return "poor_reasoning"
        
        # Check for factual errors
        pred_words = set(prediction.lower().split())
        gold_words = set(gold.lower().split())
        
        if len(pred_words.intersection(gold_words)) / len(gold_words) < 0.2:
            return "factual_error"
        
        return "other"
    
    def _assess_reasoning_quality(self, trace: Dict) -> float:
        """Assess quality of reasoning trace."""
        if not trace or 'reasoning_results' not in trace:
            return 0.0
        
        reasoning_results = trace['reasoning_results']
        if not reasoning_results:
            return 0.0
        
        # Simple heuristic: check for logical indicators
        quality_indicators = ["because", "therefore", "since", "given", "evidence"]
        total_score = 0
        
        for result in reasoning_results.values():
            text = result.get('result', '').lower()
            score = sum(1 for indicator in quality_indicators if indicator in text)
            total_score += min(score / len(quality_indicators), 1.0)
        
        return total_score / len(reasoning_results)
    
    def _compile_failure_analysis(self, failures: List[Dict]) -> Dict[str, Any]:
        """Compile comprehensive failure analysis."""
        if not failures:
            return {"total_failures": 0}
        
        # Group by error type
        error_type_counts = defaultdict(int)
        question_type_failures = defaultdict(int)
        
        for failure in failures:
            error_type_counts[failure["error_type"]] += 1
            question_type_failures[failure["question_type"]] += 1
        
        # Find most common failure patterns
        most_common_error = max(error_type_counts, key=error_type_counts.get)
        most_problematic_question_type = max(question_type_failures, key=question_type_failures.get)
        
        analysis = {
            "total_failures": len(failures),
            "failure_rate": len(failures) / (len(failures) + 100),  # Assuming some success cases
            "error_type_distribution": dict(error_type_counts),
            "question_type_failures": dict(question_type_failures),
            "most_common_error": most_common_error,
            "most_problematic_question_type": most_problematic_question_type,
            "detailed_failures": failures[:10],  # Top 10 failures for detailed analysis
            "recommendations": self._generate_recommendations(error_type_counts, question_type_failures)
        }
        
        return analysis
    
    def _generate_recommendations(self, error_types: Dict, question_types: Dict) -> List[str]:
        """Generate recommendations based on failure analysis."""
        recommendations = []
        
        # Error type recommendations
        if error_types.get("knowledge_gap", 0) > 3:
            recommendations.append("Improve knowledge base coverage and search capabilities")
        
        if error_types.get("poor_reasoning", 0) > 3:
            recommendations.append("Enhance reasoning chain quality and logical flow")
        
        if error_types.get("factual_error", 0) > 3:
            recommendations.append("Strengthen fact verification and source validation")
        
        if error_types.get("incomplete_answer", 0) > 3:
            recommendations.append("Improve answer completeness checking and synthesis")
        
        # Question type recommendations
        if question_types.get("temporal", 0) > 2:
            recommendations.append("Enhance temporal reasoning and date extraction capabilities")
        
        if question_types.get("quantitative", 0) > 2:
            recommendations.append("Improve numerical reasoning and calculation tools")
        
        if question_types.get("causal", 0) > 2:
            recommendations.append("Strengthen causal reasoning and relationship detection")
        
        return recommendations if recommendations else ["Performance is generally good across all categories"]