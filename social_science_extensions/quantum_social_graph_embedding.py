# -*- coding: utf-8 -*-
"""
Quantum Social Graph Embedding

Encode social networks as entangled graphs representing trust, influence, 
and resistance relationships. Use superposition to represent overlapping 
identities or roles in social systems.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
import networkx as nx
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector, partial_trace
from qiskit_aer import AerSimulator
import logging
from dataclasses import dataclass
from enum import Enum
from social_science_extensions import QuantumSocialGraphEmbedding
graph = QuantumSocialGraphEmbedding(max_qubits=20)
graph.load_citation_network("multilingual_corpus.json")
paths = graph.quantum_walk(start_node="paper_001")

logger = logging.getLogger(__name__)

class SocialRelationType(Enum):
    """Types of social relationships in quantum networks."""
    TRUST = "trust"
    INFLUENCE = "influence"
    RESISTANCE = "resistance"
    COOPERATION = "cooperation"
    COMPETITION = "competition"
    KINSHIP = "kinship"
    AUTHORITY = "authority"
    FRIENDSHIP = "friendship"

class SocialIdentityType(Enum):
    """Types of social identities that can overlap."""
    PROFESSIONAL = "professional"
    CULTURAL = "cultural"
    RELIGIOUS = "religious"
    POLITICAL = "political"
    ECONOMIC = "economic"
    FAMILIAL = "familial"
    EDUCATIONAL = "educational"
    GENERATIONAL = "generational"

@dataclass
class SocialNode:
    """Represents a social actor with quantum properties."""
    node_id: str
    identities: List[SocialIdentityType]
    influence_score: float
    trust_level: float
    resistance_factor: float
    cultural_background: str
    quantum_state: Optional[List[complex]] = None

@dataclass
class SocialEdge:
    """Represents a social relationship with quantum entanglement."""
    source: str
    target: str
    relationship_type: SocialRelationType
    strength: float
    reciprocity: float
    temporal_stability: float
    cultural_compatibility: float
    quantum_entanglement: Optional[float] = None

class QuantumSocialGraphEmbedding:
    """
    Quantum-enhanced social network analysis with entangled graph representations.
    
    Encodes social networks as quantum graphs where:
    - Nodes represent social actors with superposed identities
    - Edges represent entangled social relationships
    - Quantum states preserve overlapping roles and identities
    """
    
    def __init__(self, max_qubits: int = 24, max_actors: int = 50):
        """Initialize quantum social graph embedding system."""
        self.max_qubits = max_qubits
        self.max_actors = max_actors
        self.simulator = AerSimulator()
        
        # Social network state
        self.social_nodes = {}
        self.social_edges = {}
        self.quantum_social_circuits = {}
        self.identity_superpositions = {}
        self.relationship_entanglements = {}
        
        # Social dynamics parameters
        self.social_influence_weights = {
            SocialRelationType.TRUST: 0.9,
            SocialRelationType.INFLUENCE: 0.8,
            SocialRelationType.AUTHORITY: 0.85,
            SocialRelationType.FRIENDSHIP: 0.7,
            SocialRelationType.COOPERATION: 0.75,
            SocialRelationType.KINSHIP: 0.8,
            SocialRelationType.COMPETITION: -0.3,
            SocialRelationType.RESISTANCE: -0.6
        }
        
        logger.info(f"Initialized QuantumSocialGraphEmbedding with {max_qubits} qubits for {max_actors} actors")
    
    def create_social_node(self, node_id: str, identities: List[SocialIdentityType],
                          influence_score: float, trust_level: float,
                          resistance_factor: float, cultural_background: str) -> SocialNode:
        """
        Create a quantum social node with superposed identities.
        
        Args:
            node_id: Unique identifier for the social actor
            identities: List of overlapping social identities
            influence_score: Actor's influence capacity (0-1)
            trust_level: Actor's trustworthiness (0-1)
            resistance_factor: Actor's resistance to change (0-1)
            cultural_background: Cultural context identifier
            
        Returns:
            SocialNode with quantum state encoding
        """
        # Create quantum circuit for identity superposition
        num_identity_qubits = min(len(identities), self.max_qubits // 4)
        qreg = QuantumRegister(num_identity_qubits, f'identities_{node_id}')
        circuit = QuantumCircuit(qreg)
        
        # Initialize superposition of identities
        for i in range(num_identity_qubits):
            circuit.h(qreg[i])
        
        # Encode identity-specific phases
        for i, identity in enumerate(identities[:num_identity_qubits]):
            identity_phase = hash(identity.value) % 100 / 100 * np.pi
            circuit.rz(identity_phase, qreg[i])
        
        # Encode social characteristics
        influence_angle = influence_score * np.pi / 2
        trust_angle = trust_level * np.pi / 2
        resistance_angle = resistance_factor * np.pi / 2
        
        for i in range(num_identity_qubits):
            circuit.ry(influence_angle, qreg[i])
            circuit.rz(trust_angle, qreg[i])
            circuit.rx(resistance_angle, qreg[i])
        
        # Cultural background encoding
        cultural_phase = hash(cultural_background) % 100 / 100 * np.pi
        for i in range(num_identity_qubits):
            circuit.rz(cultural_phase, qreg[i])
        
        # Generate quantum state
        job = self.simulator.run(circuit, shots=1)
        result = job.result()
        statevector = result.get_statevector()
        
        # Create social node
        social_node = SocialNode(
            node_id=node_id,
            identities=identities,
            influence_score=influence_score,
            trust_level=trust_level,
            resistance_factor=resistance_factor,
            cultural_background=cultural_background,
            quantum_state=statevector.data.tolist()
        )
        
        self.social_nodes[node_id] = social_node
        self.quantum_social_circuits[f"node_{node_id}"] = circuit
        
        logger.info(f"Created quantum social node: {node_id} with {len(identities)} identities")
        return social_node
    
    def create_social_edge(self, source_id: str, target_id: str,
                          relationship_type: SocialRelationType, strength: float,
                          reciprocity: float = 0.5, temporal_stability: float = 0.8,
                          cultural_compatibility: float = 0.7) -> SocialEdge:
        """
        Create a quantum-entangled social relationship edge.
        
        Args:
            source_id: Source actor ID
            target_id: Target actor ID
            relationship_type: Type of social relationship
            strength: Relationship strength (0-1)
            reciprocity: Bidirectional relationship strength (0-1)
            temporal_stability: Relationship stability over time (0-1)
            cultural_compatibility: Cultural alignment factor (0-1)
            
        Returns:
            SocialEdge with quantum entanglement properties
        """
        if source_id not in self.social_nodes or target_id not in self.social_nodes:
            raise ValueError("Both source and target nodes must exist")
        
        # Create quantum entanglement circuit
        qreg = QuantumRegister(4, f'relationship_{source_id}_{target_id}')
        circuit = QuantumCircuit(qreg)
        
        # Create Bell state for entanglement
        circuit.h(qreg[0])
        circuit.cx(qreg[0], qreg[1])
        
        # Encode relationship properties
        relationship_phase = self.social_influence_weights.get(relationship_type, 0.5) * np.pi
        circuit.rz(relationship_phase, qreg[0])
        circuit.rz(relationship_phase, qreg[1])
        
        # Encode strength and reciprocity
        strength_angle = strength * np.pi / 2
        reciprocity_angle = reciprocity * np.pi / 2
        
        circuit.ry(strength_angle, qreg[2])
        circuit.ry(reciprocity_angle, qreg[3])
        
        # Create entanglement between relationship properties
        circuit.cx(qreg[2], qreg[3])
        
        # Measure entanglement strength
        circuit.measure_all()
        job = self.simulator.run(circuit, shots=1024)
        result = job.result()
        counts = result.get_counts()
        
        # Calculate quantum entanglement measure
        total_shots = sum(counts.values())
        entangled_states = [state for state in counts.keys() if state.count('1') % 2 == 0]
        entanglement_measure = sum(counts.get(state, 0) for state in entangled_states) / total_shots
        
        # Create social edge
        social_edge = SocialEdge(
            source=source_id,
            target=target_id,
            relationship_type=relationship_type,
            strength=strength,
            reciprocity=reciprocity,
            temporal_stability=temporal_stability,
            cultural_compatibility=cultural_compatibility,
            quantum_entanglement=entanglement_measure
        )
        
        edge_key = f"{source_id}_{target_id}_{relationship_type.value}"
        self.social_edges[edge_key] = social_edge
        self.relationship_entanglements[edge_key] = circuit
        
        logger.info(f"Created quantum social edge: {source_id} -> {target_id} ({relationship_type.value}) with entanglement {entanglement_measure:.3f}")
        return social_edge
    
    def encode_overlapping_identities(self, node_id: str) -> QuantumCircuit:
        """
        Encode overlapping social identities using quantum superposition.
        
        Args:
            node_id: Social actor identifier
            
        Returns:
            Quantum circuit encoding identity superposition
        """
        if node_id not in self.social_nodes:
            raise ValueError(f"Node {node_id} not found")
        
        node = self.social_nodes[node_id]
        identities = node.identities
        
        # Create identity superposition circuit
        num_qubits = min(len(identities), self.max_qubits // 2)
        qreg = QuantumRegister(num_qubits, f'identities_{node_id}')
        circuit = QuantumCircuit(qreg)
        
        # Create uniform superposition of all identities
        for i in range(num_qubits):
            circuit.h(qreg[i])
        
        # Apply identity-specific rotations
        for i, identity in enumerate(identities[:num_qubits]):
            # Professional identity has different quantum signature than cultural
            if identity == SocialIdentityType.PROFESSIONAL:
                circuit.ry(np.pi/4, qreg[i])
            elif identity == SocialIdentityType.CULTURAL:
                circuit.rz(np.pi/3, qreg[i])
            elif identity == SocialIdentityType.RELIGIOUS:
                circuit.rx(np.pi/6, qreg[i])
            elif identity == SocialIdentityType.POLITICAL:
                circuit.ry(np.pi/5, qreg[i])
            elif identity == SocialIdentityType.FAMILIAL:
                circuit.rz(np.pi/2, qreg[i])
            else:
                # Default encoding for other identities
                identity_angle = hash(identity.value) % 100 / 100 * np.pi
                circuit.ry(identity_angle, qreg[i])
        
        # Create entanglement between overlapping identities
        for i in range(num_qubits - 1):
            # Stronger entanglement for related identities
            circuit.cx(qreg[i], qreg[i + 1])
        
        self.identity_superpositions[node_id] = circuit
        logger.info(f"Encoded {len(identities)} overlapping identities for node {node_id}")
        
        return circuit
    
    def simulate_social_influence_propagation(self, source_id: str, influence_message: str,
                                            propagation_steps: int = 5) -> Dict[str, Any]:
        """
        Simulate influence propagation through quantum social network.
        
        Args:
            source_id: Source of influence
            influence_message: Message or influence being propagated
            propagation_steps: Number of propagation steps
            
        Returns:
            Influence propagation results with quantum probabilities
        """
        if source_id not in self.social_nodes:
            raise ValueError(f"Source node {source_id} not found")
        
        # Get all nodes connected to source
        connected_nodes = set()
        for edge_key, edge in self.social_edges.items():
            if edge.source == source_id:
                connected_nodes.add(edge.target)
            elif edge.target == source_id and edge.reciprocity > 0.5:
                connected_nodes.add(edge.source)
        
        if not connected_nodes:
            return {"influenced_nodes": [], "influence_probabilities": {}, "quantum_coherence": 0.0}
        
        # Create influence propagation circuit
        num_nodes = min(len(connected_nodes) + 1, self.max_qubits)
        qreg = QuantumRegister(num_nodes, 'influence_propagation')
        circuit = QuantumCircuit(qreg)
        
        # Initialize source node in |1⟩ state (influenced)
        circuit.x(qreg[0])
        
        # Propagation simulation
        for step in range(propagation_steps):
            # Apply influence based on relationship strengths
            node_idx = 1
            for target_id in list(connected_nodes)[:num_nodes-1]:
                # Find relationship strength
                edge_key = f"{source_id}_{target_id}_influence"
                if edge_key not in self.social_edges:
                    # Try reverse direction
                    edge_key = f"{target_id}_{source_id}_influence"
                
                if edge_key in self.social_edges:
                    edge = self.social_edges[edge_key]
                    influence_strength = edge.strength * edge.quantum_entanglement
                    
                    # Apply controlled rotation based on influence strength
                    rotation_angle = influence_strength * np.pi / 2
                    circuit.cry(rotation_angle, qreg[0], qreg[node_idx])
                
                node_idx += 1
            
            # Add quantum noise for realistic social dynamics
            for i in range(1, num_nodes):
                noise_angle = np.random.normal(0, 0.1)
                circuit.ry(noise_angle, qreg[i])
        
        # Measure influence propagation
        circuit.measure_all()
        
        job = self.simulator.run(circuit, shots=1024)
        result = job.result()
        counts = result.get_counts()
        
        # Analyze results
        total_shots = sum(counts.values())
        influence_probabilities = {}
        influenced_nodes = []
        
        node_list = [source_id] + list(connected_nodes)[:num_nodes-1]
        
        for state, count in counts.items():
            probability = count / total_shots
            for i, bit in enumerate(state[::-1]):  # Reverse for qubit ordering
                if i < len(node_list):
                    node_id = node_list[i]
                    if node_id not in influence_probabilities:
                        influence_probabilities[node_id] = 0.0
                    if bit == '1':
                        influence_probabilities[node_id] += probability
        
        # Determine influenced nodes (probability > 0.5)
        for node_id, prob in influence_probabilities.items():
            if prob > 0.5 and node_id != source_id:
                influenced_nodes.append(node_id)
        
        # Calculate quantum coherence
        probabilities = list(influence_probabilities.values())
        quantum_coherence = 1.0 - (-sum(p * np.log2(p + 1e-10) for p in probabilities) / np.log2(len(probabilities)))
        
        propagation_results = {
            "source_node": source_id,
            "influence_message": influence_message,
            "influenced_nodes": influenced_nodes,
            "influence_probabilities": influence_probabilities,
            "quantum_coherence": quantum_coherence,
            "propagation_steps": propagation_steps,
            "total_reachable_nodes": len(connected_nodes)
        }
        
        logger.info(f"Influence propagation from {source_id}: {len(influenced_nodes)} nodes influenced with coherence {quantum_coherence:.3f}")
        return propagation_results
    
    def analyze_social_network_structure(self) -> Dict[str, Any]:
        """
        Analyze quantum social network structure and properties.
        
        Returns:
            Comprehensive network analysis with quantum metrics
        """
        if not self.social_nodes or not self.social_edges:
            return {"error": "No social network data available"}
        
        # Basic network statistics
        num_nodes = len(self.social_nodes)
        num_edges = len(self.social_edges)
        
        # Calculate quantum entanglement statistics
        entanglement_values = [edge.quantum_entanglement for edge in self.social_edges.values() 
                              if edge.quantum_entanglement is not None]
        avg_entanglement = np.mean(entanglement_values) if entanglement_values else 0.0
        
        # Identity diversity analysis
        all_identities = []
        for node in self.social_nodes.values():
            all_identities.extend(node.identities)
        
        identity_distribution = {}
        for identity in all_identities:
            identity_distribution[identity.value] = identity_distribution.get(identity.value, 0) + 1
        
        # Relationship type distribution
        relationship_distribution = {}
        for edge in self.social_edges.values():
            rel_type = edge.relationship_type.value
            relationship_distribution[rel_type] = relationship_distribution.get(rel_type, 0) + 1
        
        # Cultural diversity
        cultural_backgrounds = [node.cultural_background for node in self.social_nodes.values()]
        cultural_diversity = len(set(cultural_backgrounds))
        
        # Network density (quantum-adjusted)
        max_possible_edges = num_nodes * (num_nodes - 1) / 2
        quantum_density = (num_edges / max_possible_edges) * avg_entanglement if max_possible_edges > 0 else 0.0
        
        # Influence distribution
        influence_scores = [node.influence_score for node in self.social_nodes.values()]
        trust_levels = [node.trust_level for node in self.social_nodes.values()]
        resistance_factors = [node.resistance_factor for node in self.social_nodes.values()]
        
        analysis_results = {
            "network_size": {
                "nodes": num_nodes,
                "edges": num_edges,
                "quantum_density": quantum_density
            },
            "quantum_properties": {
                "average_entanglement": avg_entanglement,
                "entanglement_variance": np.var(entanglement_values) if entanglement_values else 0.0,
                "quantum_coherence_score": avg_entanglement * quantum_density
            },
            "identity_analysis": {
                "identity_distribution": identity_distribution,
                "identity_diversity": len(identity_distribution),
                "average_identities_per_node": len(all_identities) / num_nodes if num_nodes > 0 else 0
            },
            "relationship_analysis": {
                "relationship_distribution": relationship_distribution,
                "relationship_diversity": len(relationship_distribution)
            },
            "cultural_analysis": {
                "cultural_diversity": cultural_diversity,
                "cultural_backgrounds": list(set(cultural_backgrounds))
            },
            "social_dynamics": {
                "average_influence": np.mean(influence_scores) if influence_scores else 0.0,
                "average_trust": np.mean(trust_levels) if trust_levels else 0.0,
                "average_resistance": np.mean(resistance_factors) if resistance_factors else 0.0,
                "influence_inequality": np.var(influence_scores) if influence_scores else 0.0
            }
        }
        
        logger.info(f"Analyzed quantum social network: {num_nodes} nodes, {num_edges} edges, {avg_entanglement:.3f} avg entanglement")
        return analysis_results
    
    def export_quantum_social_network(self, filepath: str):
        """Export quantum social network to file."""
        import json
        
        export_data = {
            "social_nodes": {
                node_id: {
                    "identities": [identity.value for identity in node.identities],
                    "influence_score": node.influence_score,
                    "trust_level": node.trust_level,
                    "resistance_factor": node.resistance_factor,
                    "cultural_background": node.cultural_background
                } for node_id, node in self.social_nodes.items()
            },
            "social_edges": {
                edge_key: {
                    "source": edge.source,
                    "target": edge.target,
                    "relationship_type": edge.relationship_type.value,
                    "strength": edge.strength,
                    "reciprocity": edge.reciprocity,
                    "temporal_stability": edge.temporal_stability,
                    "cultural_compatibility": edge.cultural_compatibility,
                    "quantum_entanglement": edge.quantum_entanglement
                } for edge_key, edge in self.social_edges.items()
            },
            "network_analysis": self.analyze_social_network_structure()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported quantum social network to {filepath}")
    
    def get_quantum_social_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics for quantum social graph embedding."""
        return {
            "total_social_nodes": len(self.social_nodes),
            "total_social_edges": len(self.social_edges),
            "identity_superpositions": len(self.identity_superpositions),
            "relationship_entanglements": len(self.relationship_entanglements),
            "max_qubits": self.max_qubits,
            "max_actors": self.max_actors,
            "quantum_circuits_created": len(self.quantum_social_circuits),
            "social_influence_types": len(self.social_influence_weights),
            "quantum_advantage_factor": len(self.social_nodes) ** 2  # Quadratic advantage in social analysis
        }