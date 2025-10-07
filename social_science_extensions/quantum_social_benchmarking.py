# -*- coding: utf-8 -*-
"""
Quantum Social Benchmarking

Evaluate emergent social patterns (e.g., norm convergence, polarization) 
using probabilistic metrics and quantum-enhanced evaluation frameworks.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
from dataclasses import dataclass
from enum import Enum
import time
from qiskit import QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class SocialPatternType(Enum):
    """Types of emergent social patterns to evaluate."""
    NORM_CONVERGENCE = "norm_convergence"
    POLARIZATION = "polarization"
    CONSENSUS_FORMATION = "consensus_formation"
    SOCIAL_FRAGMENTATION = "social_fragmentation"
    CULTURAL_DIFFUSION = "cultural_diffusion"
    OPINION_CLUSTERING = "opinion_clustering"
    INFLUENCE_PROPAGATION = "influence_propagation"
    BEHAVIORAL_SYNCHRONIZATION = "behavioral_synchronization"

class BenchmarkMetric(Enum):
    """Quantum benchmarking metrics for social patterns."""
    QUANTUM_COHERENCE = "quantum_coherence"
    ENTANGLEMENT_MEASURE = "entanglement_measure"
    PATTERN_STABILITY = "pattern_stability"
    EMERGENCE_SPEED = "emergence_speed"
    CULTURAL_DIVERSITY = "cultural_diversity"
    CONSENSUS_STRENGTH = "consensus_strength"
    POLARIZATION_INDEX = "polarization_index"
    NETWORK_RESILIENCE = "network_resilience"

@dataclass
class SocialBenchmarkResult:
    """Results from quantum social benchmarking."""
    benchmark_id: str
    pattern_type: SocialPatternType
    metric_scores: Dict[BenchmarkMetric, float]
    quantum_measurements: Dict[str, Any]
    execution_time: float
    cultural_contexts: List[str]
    agent_count: int
    convergence_achieved: bool

@dataclass
class SocialExperiment:
    """Represents a social science experiment for benchmarking."""
    experiment_id: str
    experiment_description: str
    pattern_types: List[SocialPatternType]
    agent_configurations: List[Dict[str, Any]]
    cultural_contexts: List[str]
    simulation_parameters: Dict[str, Any]

class QuantumSocialBenchmarking:
    """
    Quantum-enhanced benchmarking system for social science patterns.
    
    Evaluates emergent social patterns using quantum probabilistic metrics,
    providing comprehensive assessment of social dynamics with quantum advantage.
    """
    
    def __init__(self, max_qubits: int = 24, max_agents: int = 100):
        """Initialize quantum social benchmarking system."""
        self.max_qubits = max_qubits
        self.max_agents = max_agents
        self.simulator = AerSimulator()
        
        # Benchmarking state
        self.benchmark_results = {}
        self.social_experiments = {}
        self.quantum_benchmark_circuits = {}
        self.pattern_evaluation_history = []
        
        # Quantum metric calculators
        self.metric_calculators = {
            BenchmarkMetric.QUANTUM_COHERENCE: self._calculate_quantum_coherence,
            BenchmarkMetric.ENTANGLEMENT_MEASURE: self._calculate_entanglement_measure,
            BenchmarkMetric.PATTERN_STABILITY: self._calculate_pattern_stability,
            BenchmarkMetric.EMERGENCE_SPEED: self._calculate_emergence_speed,
            BenchmarkMetric.CULTURAL_DIVERSITY: self._calculate_cultural_diversity,
            BenchmarkMetric.CONSENSUS_STRENGTH: self._calculate_consensus_strength,
            BenchmarkMetric.POLARIZATION_INDEX: self._calculate_polarization_index,
            BenchmarkMetric.NETWORK_RESILIENCE: self._calculate_network_resilience
        }
        
        # Pattern-specific quantum encodings
        self.pattern_quantum_signatures = {
            SocialPatternType.NORM_CONVERGENCE: {'phase': np.pi/4, 'entanglement': 'linear'},
            SocialPatternType.POLARIZATION: {'phase': np.pi/2, 'entanglement': 'bipartite'},
            SocialPatternType.CONSENSUS_FORMATION: {'phase': np.pi/6, 'entanglement': 'star'},
            SocialPatternType.SOCIAL_FRAGMENTATION: {'phase': np.pi/3, 'entanglement': 'clustered'},
            SocialPatternType.CULTURAL_DIFFUSION: {'phase': np.pi/5, 'entanglement': 'random'},
            SocialPatternType.OPINION_CLUSTERING: {'phase': 2*np.pi/3, 'entanglement': 'modular'},
            SocialPatternType.INFLUENCE_PROPAGATION: {'phase': np.pi/8, 'entanglement': 'cascade'},
            SocialPatternType.BEHAVIORAL_SYNCHRONIZATION: {'phase': np.pi, 'entanglement': 'complete'}
        }
        
        logger.info(f"Initialized QuantumSocialBenchmarking with {max_qubits} qubits for {max_agents} agents")
    
    def create_social_experiment(self, experiment_id: str, experiment_description: str,
                               pattern_types: List[SocialPatternType],
                               agent_configurations: List[Dict[str, Any]],
                               cultural_contexts: List[str],
                               simulation_parameters: Dict[str, Any] = None) -> SocialExperiment:
        """
        Create a social science experiment for quantum benchmarking.
        
        Args:
            experiment_id: Unique experiment identifier
            experiment_description: Description of the experiment
            pattern_types: Social patterns to evaluate
            agent_configurations: Agent setup configurations
            cultural_contexts: Cultural contexts involved
            simulation_parameters: Additional simulation parameters
            
        Returns:
            SocialExperiment configuration
        """
        if simulation_parameters is None:
            simulation_parameters = {
                'simulation_steps': 100,
                'interaction_probability': 0.7,
                'cultural_influence_strength': 0.5,
                'noise_level': 0.1
            }
        
        social_experiment = SocialExperiment(
            experiment_id=experiment_id,
            experiment_description=experiment_description,
            pattern_types=pattern_types,
            agent_configurations=agent_configurations,
            cultural_contexts=cultural_contexts,
            simulation_parameters=simulation_parameters
        )
        
        self.social_experiments[experiment_id] = social_experiment
        logger.info(f"Created social experiment: {experiment_id} with {len(pattern_types)} patterns")
        
        return social_experiment
    
    def create_quantum_pattern_circuit(self, pattern_type: SocialPatternType,
                                     agent_states: List[Dict[str, float]],
                                     cultural_contexts: List[str]) -> QuantumCircuit:
        """
        Create quantum circuit for social pattern evaluation.
        
        Args:
            pattern_type: Type of social pattern
            agent_states: Current states of agents
            cultural_contexts: Cultural contexts involved
            
        Returns:
            Quantum circuit encoding the social pattern
        """
        num_agents = min(len(agent_states), self.max_qubits)
        qreg = QuantumRegister(num_agents, f'pattern_{pattern_type.value}')
        circuit = QuantumCircuit(qreg)
        
        # Initialize agent states
        for i, agent_state in enumerate(agent_states[:num_agents]):
            # Encode agent opinion/behavior
            opinion = agent_state.get('opinion', 0.5)
            influence = agent_state.get('influence', 0.5)
            cultural_alignment = agent_state.get('cultural_alignment', 0.5)
            
            # Initialize superposition
            circuit.h(qreg[i])
            
            # Encode agent characteristics
            opinion_angle = opinion * np.pi
            influence_angle = influence * np.pi / 2
            cultural_angle = cultural_alignment * np.pi / 3
            
            circuit.ry(opinion_angle, qreg[i])
            circuit.rz(influence_angle, qreg[i])
            circuit.rx(cultural_angle, qreg[i])
        
        # Apply pattern-specific quantum operations
        pattern_signature = self.pattern_quantum_signatures.get(pattern_type, {})
        pattern_phase = pattern_signature.get('phase', np.pi/4)
        entanglement_type = pattern_signature.get('entanglement', 'linear')
        
        # Apply pattern phase
        for i in range(num_agents):
            circuit.rz(pattern_phase, qreg[i])
        
        # Create pattern-specific entanglement
        self._apply_pattern_entanglement(circuit, qreg, entanglement_type, num_agents)
        
        # Add cultural context encoding
        for i, context in enumerate(cultural_contexts[:num_agents]):
            if i < num_agents:
                context_phase = hash(context) % 100 / 100 * np.pi
                circuit.rz(context_phase, qreg[i])
        
        circuit_key = f"{pattern_type.value}_{len(agent_states)}_{hash(str(cultural_contexts))}"
        self.quantum_benchmark_circuits[circuit_key] = circuit
        
        logger.info(f"Created quantum pattern circuit for {pattern_type.value}: {num_agents} agents")
        return circuit
    
    def _apply_pattern_entanglement(self, circuit: QuantumCircuit, qreg: QuantumRegister,
                                  entanglement_type: str, num_agents: int):
        """Apply pattern-specific entanglement to quantum circuit."""
        if entanglement_type == 'linear':
            # Linear chain entanglement
            for i in range(num_agents - 1):
                circuit.cx(qreg[i], qreg[i + 1])
        
        elif entanglement_type == 'bipartite':
            # Bipartite entanglement for polarization
            mid_point = num_agents // 2
            for i in range(mid_point):
                if i + mid_point < num_agents:
                    circuit.cx(qreg[i], qreg[i + mid_point])
        
        elif entanglement_type == 'star':
            # Star topology for consensus formation
            for i in range(1, num_agents):
                circuit.cx(qreg[0], qreg[i])
        
        elif entanglement_type == 'clustered':
            # Clustered entanglement for fragmentation
            cluster_size = max(2, num_agents // 3)
            for cluster_start in range(0, num_agents, cluster_size):
                cluster_end = min(cluster_start + cluster_size, num_agents)
                for i in range(cluster_start, cluster_end - 1):
                    circuit.cx(qreg[i], qreg[i + 1])
        
        elif entanglement_type == 'random':
            # Random entanglement for diffusion
            import random
            for _ in range(num_agents // 2):
                i, j = random.sample(range(num_agents), 2)
                circuit.cx(qreg[i], qreg[j])
        
        elif entanglement_type == 'modular':
            # Modular entanglement for clustering
            module_size = max(3, num_agents // 4)
            for module_start in range(0, num_agents, module_size):
                module_end = min(module_start + module_size, num_agents)
                # Create complete graph within module
                for i in range(module_start, module_end):
                    for j in range(i + 1, module_end):
                        circuit.cx(qreg[i], qreg[j])
        
        elif entanglement_type == 'cascade':
            # Cascade entanglement for influence propagation
            for level in range(int(np.log2(num_agents)) + 1):
                for i in range(0, num_agents, 2**(level+1)):
                    if i + 2**level < num_agents:
                        circuit.cx(qreg[i], qreg[i + 2**level])
        
        elif entanglement_type == 'complete':
            # Complete entanglement for synchronization
            for i in range(num_agents):
                for j in range(i + 1, num_agents):
                    circuit.cx(qreg[i], qreg[j])
    
    def evaluate_social_pattern(self, pattern_type: SocialPatternType,
                              agent_states: List[Dict[str, float]],
                              cultural_contexts: List[str],
                              metrics: List[BenchmarkMetric] = None) -> SocialBenchmarkResult:
        """
        Evaluate a social pattern using quantum benchmarking.
        
        Args:
            pattern_type: Type of social pattern to evaluate
            agent_states: Current states of all agents
            cultural_contexts: Cultural contexts involved
            metrics: Specific metrics to calculate (all if None)
            
        Returns:
            SocialBenchmarkResult with quantum evaluation
        """
        start_time = time.time()
        
        if metrics is None:
            metrics = list(BenchmarkMetric)
        
        # Create quantum circuit for pattern
        circuit = self.create_quantum_pattern_circuit(pattern_type, agent_states, cultural_contexts)
        
        # Measure quantum circuit
        circuit.measure_all()
        job = self.simulator.run(circuit, shots=1024)
        result = job.result()
        counts = result.get_counts()
        
        # Calculate quantum measurements
        quantum_measurements = {
            'measurement_counts': counts,
            'total_shots': sum(counts.values()),
            'state_distribution': {state: count/sum(counts.values()) for state, count in counts.items()},
            'dominant_state': max(counts.keys(), key=counts.get),
            'measurement_entropy': self._calculate_measurement_entropy(counts)
        }
        
        # Calculate benchmark metrics
        metric_scores = {}
        for metric in metrics:
            if metric in self.metric_calculators:
                score = self.metric_calculators[metric](
                    pattern_type, agent_states, cultural_contexts, quantum_measurements
                )
                metric_scores[metric] = score
        
        # Determine convergence
        convergence_achieved = self._assess_pattern_convergence(
            pattern_type, quantum_measurements, metric_scores
        )
        
        execution_time = time.time() - start_time
        
        # Create benchmark result
        benchmark_result = SocialBenchmarkResult(
            benchmark_id=f"{pattern_type.value}_{int(time.time())}",
            pattern_type=pattern_type,
            metric_scores=metric_scores,
            quantum_measurements=quantum_measurements,
            execution_time=execution_time,
            cultural_contexts=cultural_contexts,
            agent_count=len(agent_states),
            convergence_achieved=convergence_achieved
        )
        
        # Store result
        result_key = f"{pattern_type.value}_{len(agent_states)}_{len(cultural_contexts)}"
        self.benchmark_results[result_key] = benchmark_result
        self.pattern_evaluation_history.append(benchmark_result)
        
        logger.info(f"Evaluated {pattern_type.value}: {len(metrics)} metrics, convergence={convergence_achieved}")
        return benchmark_result
    
    def run_comprehensive_benchmark(self, experiment_id: str) -> Dict[str, Any]:
        """
        Run comprehensive quantum benchmarking for a social experiment.
        
        Args:
            experiment_id: Experiment to benchmark
            
        Returns:
            Comprehensive benchmarking results
        """
        if experiment_id not in self.social_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.social_experiments[experiment_id]
        
        comprehensive_results = {
            'experiment_id': experiment_id,
            'experiment_description': experiment.experiment_description,
            'pattern_evaluations': {},
            'comparative_analysis': {},
            'quantum_advantage_metrics': {},
            'execution_summary': {}
        }
        
        start_time = time.time()
        
        # Evaluate each pattern type
        for pattern_type in experiment.pattern_types:
            logger.info(f"Evaluating pattern: {pattern_type.value}")
            
            # Use agent configurations as agent states
            agent_states = experiment.agent_configurations
            
            # Evaluate pattern
            pattern_result = self.evaluate_social_pattern(
                pattern_type, agent_states, experiment.cultural_contexts
            )
            
            comprehensive_results['pattern_evaluations'][pattern_type.value] = {
                'benchmark_result': pattern_result,
                'metric_scores': pattern_result.metric_scores,
                'convergence_achieved': pattern_result.convergence_achieved,
                'execution_time': pattern_result.execution_time
            }
        
        # Comparative analysis across patterns
        if len(experiment.pattern_types) > 1:
            comprehensive_results['comparative_analysis'] = self._perform_comparative_analysis(
                experiment.pattern_types, comprehensive_results['pattern_evaluations']
            )
        
        # Calculate quantum advantage metrics
        comprehensive_results['quantum_advantage_metrics'] = self._calculate_quantum_advantage_metrics(
            comprehensive_results['pattern_evaluations']
        )
        
        # Execution summary
        total_time = time.time() - start_time
        comprehensive_results['execution_summary'] = {
            'total_execution_time': total_time,
            'patterns_evaluated': len(experiment.pattern_types),
            'agents_simulated': len(experiment.agent_configurations),
            'cultural_contexts': len(experiment.cultural_contexts),
            'quantum_circuits_created': len(experiment.pattern_types),
            'average_pattern_time': total_time / len(experiment.pattern_types) if experiment.pattern_types else 0
        }
        
        logger.info(f"Completed comprehensive benchmark for {experiment_id}: {len(experiment.pattern_types)} patterns in {total_time:.2f}s")
        return comprehensive_results
    
    def _perform_comparative_analysis(self, pattern_types: List[SocialPatternType],
                                    pattern_evaluations: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comparative analysis across different social patterns."""
        comparative_analysis = {
            'metric_comparisons': {},
            'pattern_rankings': {},
            'correlation_analysis': {},
            'quantum_coherence_comparison': {}
        }
        
        # Compare metrics across patterns
        all_metrics = set()
        for pattern_eval in pattern_evaluations.values():
            all_metrics.update(pattern_eval['metric_scores'].keys())
        
        for metric in all_metrics:
            metric_values = {}
            for pattern_name, pattern_eval in pattern_evaluations.items():
                if metric in pattern_eval['metric_scores']:
                    metric_values[pattern_name] = pattern_eval['metric_scores'][metric]
            
            if len(metric_values) > 1:
                comparative_analysis['metric_comparisons'][metric.value] = {
                    'values': metric_values,
                    'best_pattern': max(metric_values.keys(), key=metric_values.get),
                    'worst_pattern': min(metric_values.keys(), key=metric_values.get),
                    'variance': np.var(list(metric_values.values()))
                }
        
        # Pattern rankings by overall performance
        pattern_scores = {}
        for pattern_name, pattern_eval in pattern_evaluations.items():
            scores = list(pattern_eval['metric_scores'].values())
            pattern_scores[pattern_name] = np.mean(scores) if scores else 0.0
        
        sorted_patterns = sorted(pattern_scores.items(), key=lambda x: x[1], reverse=True)
        comparative_analysis['pattern_rankings'] = {
            'ranked_patterns': sorted_patterns,
            'best_pattern': sorted_patterns[0][0] if sorted_patterns else None,
            'performance_spread': max(pattern_scores.values()) - min(pattern_scores.values()) if pattern_scores else 0
        }
        
        return comparative_analysis
    
    def _calculate_quantum_advantage_metrics(self, pattern_evaluations: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate quantum advantage metrics for the benchmarking."""
        quantum_advantage = {
            'parallel_evaluation_advantage': len(pattern_evaluations),
            'quantum_coherence_advantage': 0.0,
            'entanglement_utilization': 0.0,
            'measurement_efficiency': 0.0
        }
        
        # Calculate average quantum coherence
        coherence_scores = []
        entanglement_scores = []
        
        for pattern_eval in pattern_evaluations.values():
            metric_scores = pattern_eval['metric_scores']
            
            if BenchmarkMetric.QUANTUM_COHERENCE in metric_scores:
                coherence_scores.append(metric_scores[BenchmarkMetric.QUANTUM_COHERENCE])
            
            if BenchmarkMetric.ENTANGLEMENT_MEASURE in metric_scores:
                entanglement_scores.append(metric_scores[BenchmarkMetric.ENTANGLEMENT_MEASURE])
        
        quantum_advantage['quantum_coherence_advantage'] = np.mean(coherence_scores) if coherence_scores else 0.0
        quantum_advantage['entanglement_utilization'] = np.mean(entanglement_scores) if entanglement_scores else 0.0
        
        # Calculate measurement efficiency
        total_measurements = sum(
            pattern_eval['benchmark_result'].quantum_measurements['total_shots']
            for pattern_eval in pattern_evaluations.values()
        )
        total_patterns = len(pattern_evaluations)
        quantum_advantage['measurement_efficiency'] = total_patterns / (total_measurements / 1024) if total_measurements > 0 else 0.0
        
        return quantum_advantage
    
    def _assess_pattern_convergence(self, pattern_type: SocialPatternType,
                                  quantum_measurements: Dict[str, Any],
                                  metric_scores: Dict[BenchmarkMetric, float]) -> bool:
        """Assess whether a social pattern has converged."""
        # Pattern-specific convergence criteria
        convergence_thresholds = {
            SocialPatternType.NORM_CONVERGENCE: {'coherence': 0.8, 'consensus': 0.7},
            SocialPatternType.POLARIZATION: {'polarization_index': 0.6, 'stability': 0.7},
            SocialPatternType.CONSENSUS_FORMATION: {'consensus_strength': 0.8, 'coherence': 0.7},
            SocialPatternType.SOCIAL_FRAGMENTATION: {'diversity': 0.6, 'stability': 0.5},
            SocialPatternType.CULTURAL_DIFFUSION: {'diversity': 0.7, 'emergence_speed': 0.6}
        }
        
        thresholds = convergence_thresholds.get(pattern_type, {'coherence': 0.7})
        
        # Check convergence criteria
        convergence_checks = []
        
        for criterion, threshold in thresholds.items():
            if criterion == 'coherence' and BenchmarkMetric.QUANTUM_COHERENCE in metric_scores:
                convergence_checks.append(metric_scores[BenchmarkMetric.QUANTUM_COHERENCE] >= threshold)
            elif criterion == 'consensus' and BenchmarkMetric.CONSENSUS_STRENGTH in metric_scores:
                convergence_checks.append(metric_scores[BenchmarkMetric.CONSENSUS_STRENGTH] >= threshold)
            elif criterion == 'polarization_index' and BenchmarkMetric.POLARIZATION_INDEX in metric_scores:
                convergence_checks.append(metric_scores[BenchmarkMetric.POLARIZATION_INDEX] >= threshold)
            elif criterion == 'stability' and BenchmarkMetric.PATTERN_STABILITY in metric_scores:
                convergence_checks.append(metric_scores[BenchmarkMetric.PATTERN_STABILITY] >= threshold)
            elif criterion == 'diversity' and BenchmarkMetric.CULTURAL_DIVERSITY in metric_scores:
                convergence_checks.append(metric_scores[BenchmarkMetric.CULTURAL_DIVERSITY] >= threshold)
            elif criterion == 'emergence_speed' and BenchmarkMetric.EMERGENCE_SPEED in metric_scores:
                convergence_checks.append(metric_scores[BenchmarkMetric.EMERGENCE_SPEED] >= threshold)
        
        # Require majority of criteria to be met
        return sum(convergence_checks) >= len(convergence_checks) * 0.6 if convergence_checks else False
    
    # Metric calculation methods
    def _calculate_quantum_coherence(self, pattern_type: SocialPatternType, agent_states: List[Dict[str, float]],
                                   cultural_contexts: List[str], quantum_measurements: Dict[str, Any]) -> float:
        """Calculate quantum coherence metric."""
        entropy = quantum_measurements.get('measurement_entropy', 0)
        max_entropy = np.log2(len(quantum_measurements.get('measurement_counts', {1: 1})))
        coherence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0
        return coherence
    
    def _calculate_entanglement_measure(self, pattern_type: SocialPatternType, agent_states: List[Dict[str, float]],
                                      cultural_contexts: List[str], quantum_measurements: Dict[str, Any]) -> float:
        """Calculate entanglement measure metric."""
        counts = quantum_measurements.get('measurement_counts', {})
        total_shots = quantum_measurements.get('total_shots', 1)
        
        # Look for entangled states (even parity)
        entangled_count = sum(count for state, count in counts.items() if state.count('1') % 2 == 0)
        entanglement_measure = entangled_count / total_shots
        return entanglement_measure
    
    def _calculate_pattern_stability(self, pattern_type: SocialPatternType, agent_states: List[Dict[str, float]],
                                   cultural_contexts: List[str], quantum_measurements: Dict[str, Any]) -> float:
        """Calculate pattern stability metric."""
        state_distribution = quantum_measurements.get('state_distribution', {})
        if not state_distribution:
            return 0.0
        
        # Stability is inverse of entropy
        probabilities = list(state_distribution.values())
        entropy = -sum(p * np.log2(p + 1e-10) for p in probabilities)
        max_entropy = np.log2(len(probabilities))
        stability = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0
        return stability
    
    def _calculate_emergence_speed(self, pattern_type: SocialPatternType, agent_states: List[Dict[str, float]],
                                 cultural_contexts: List[str], quantum_measurements: Dict[str, Any]) -> float:
        """Calculate emergence speed metric."""
        # Simplified emergence speed based on dominant state probability
        state_distribution = quantum_measurements.get('state_distribution', {})
        if not state_distribution:
            return 0.0
        
        max_probability = max(state_distribution.values())
        emergence_speed = max_probability  # Higher probability indicates faster emergence
        return emergence_speed
    
    def _calculate_cultural_diversity(self, pattern_type: SocialPatternType, agent_states: List[Dict[str, float]],
                                    cultural_contexts: List[str], quantum_measurements: Dict[str, Any]) -> float:
        """Calculate cultural diversity metric."""
        # Diversity based on number of unique cultural contexts
        unique_contexts = len(set(cultural_contexts))
        total_contexts = len(cultural_contexts)
        diversity = unique_contexts / total_contexts if total_contexts > 0 else 0.0
        return diversity
    
    def _calculate_consensus_strength(self, pattern_type: SocialPatternType, agent_states: List[Dict[str, float]],
                                    cultural_contexts: List[str], quantum_measurements: Dict[str, Any]) -> float:
        """Calculate consensus strength metric."""
        state_distribution = quantum_measurements.get('state_distribution', {})
        if not state_distribution:
            return 0.0
        
        # Consensus strength is the probability of the dominant state
        consensus_strength = max(state_distribution.values())
        return consensus_strength
    
    def _calculate_polarization_index(self, pattern_type: SocialPatternType, agent_states: List[Dict[str, float]],
                                    cultural_contexts: List[str], quantum_measurements: Dict[str, Any]) -> float:
        """Calculate polarization index metric."""
        # Polarization based on bimodal distribution
        state_distribution = quantum_measurements.get('state_distribution', {})
        if len(state_distribution) < 2:
            return 0.0
        
        # Sort probabilities and check for bimodal pattern
        probabilities = sorted(state_distribution.values(), reverse=True)
        top_two_prob = sum(probabilities[:2])
        polarization_index = top_two_prob if len(probabilities) >= 2 else 0.0
        return polarization_index
    
    def _calculate_network_resilience(self, pattern_type: SocialPatternType, agent_states: List[Dict[str, float]],
                                    cultural_contexts: List[str], quantum_measurements: Dict[str, Any]) -> float:
        """Calculate network resilience metric."""
        # Resilience based on measurement entropy (higher entropy = more resilient)
        entropy = quantum_measurements.get('measurement_entropy', 0)
        max_entropy = np.log2(len(quantum_measurements.get('measurement_counts', {1: 1})))
        resilience = entropy / max_entropy if max_entropy > 0 else 0.0
        return resilience
    
    def _calculate_measurement_entropy(self, measurement_counts: Dict[str, int]) -> float:
        """Calculate entropy of measurement results."""
        total_shots = sum(measurement_counts.values())
        probabilities = [count/total_shots for count in measurement_counts.values()]
        entropy = -sum(p * np.log2(p + 1e-10) for p in probabilities)
        return entropy
    
    def get_quantum_benchmarking_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics for quantum social benchmarking."""
        return {
            'total_benchmark_results': len(self.benchmark_results),
            'social_experiments': len(self.social_experiments),
            'pattern_evaluations': len(self.pattern_evaluation_history),
            'quantum_circuits_created': len(self.quantum_benchmark_circuits),
            'supported_pattern_types': len(self.pattern_quantum_signatures),
            'supported_metrics': len(self.metric_calculators),
            'max_qubits': self.max_qubits,
            'max_agents': self.max_agents,
            'average_evaluation_time': np.mean([
                result.execution_time for result in self.pattern_evaluation_history
            ]) if self.pattern_evaluation_history else 0.0,
            'convergence_rate': sum(
                1 for result in self.pattern_evaluation_history if result.convergence_achieved
            ) / len(self.pattern_evaluation_history) if self.pattern_evaluation_history else 0.0
        }