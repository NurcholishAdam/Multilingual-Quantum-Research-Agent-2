# -*- coding: utf-8 -*-
"""
REPAIR Model Editing Integration Demo

Demonstrates self-healing capabilities with REPAIR-based model editing.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_section(title):
    """Print section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def demo_1_basic_repair():
    """Demo 1: Basic REPAIR initialization"""
    print_section("Demo 1: Basic REPAIR Initialization")
    
    from multilingual_research_agent import MultilingualResearchAgent, Language
    
    # Create agent with REPAIR enabled
    agent = MultilingualResearchAgent(
        supported_languages=[Language.ENGLISH],
        quantum_enabled=False,
        enable_repair=True
    )
    
    print(f"✓ Agent created with REPAIR: {agent.enable_repair}")
    print(f"✓ Editor initialized: {agent.editor is not None}")
    print(f"✓ Inference wrapper initialized: {agent.inference is not None}")
    print(f"✓ Health checker initialized: {agent.health_checker is not None}")
    
    return agent


def demo_2_manual_edit(agent):
    """Demo 2: Manual model editing"""
    print_section("Demo 2: Manual Model Editing")
    
    if not agent.enable_repair:
        print("⚠ REPAIR not enabled, skipping demo")
        return
    
    # Prepare edit
    query = "What is the capital of France?"
    correct_answer = "Paris"
    locality_prompt = "What is quantum computing?"
    
    print(f"Query: {query}")
    print(f"Correct Answer: {correct_answer}")
    print(f"Locality Prompt: {locality_prompt}")
    
    # Apply edit
    edits = [(query, correct_answer, locality_prompt)]
    agent.editor.apply_edits(edits)
    
    # Get statistics
    stats = agent.editor.get_edit_statistics()
    
    print(f"\n✓ Edit applied successfully")
    print(f"  Total edits: {stats['total_edits']}")
    print(f"  Success rate: {stats['success_rate']:.2%}")
    print(f"  Avg reliability: {stats['avg_reliability']:.3f}")
    print(f"  Avg locality: {stats['avg_locality']:.3f}")
    print(f"  Avg generalization: {stats['avg_generalization']:.3f}")


def demo_3_health_checking(agent):
    """Demo 3: Health checking"""
    print_section("Demo 3: Health Checking")
    
    if not agent.enable_repair:
        print("⚠ REPAIR not enabled, skipping demo")
        return
    
    query = "What is machine learning?"
    
    # Test healthy response
    healthy_response = "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."
    is_healthy = agent.health_checker.is_healthy(query, healthy_response)
    
    print(f"Query: {query}")
    print(f"Response: {healthy_response[:80]}...")
    print(f"✓ Health check: {'PASSED' if is_healthy else 'FAILED'}")
    
    # Test unhealthy response
    unhealthy_response = "Error: unknown"
    is_healthy = agent.health_checker.is_healthy(query, unhealthy_response)
    
    print(f"\nResponse: {unhealthy_response}")
    print(f"✓ Health check: {'PASSED' if is_healthy else 'FAILED'}")
    
    # Get statistics
    stats = agent.health_checker.get_statistics()
    print(f"\nHealth Check Statistics:")
    print(f"  Total checks: {stats['total_checks']}")
    print(f"  Unhealthy count: {stats['unhealthy_count']}")
    print(f"  Unhealthy rate: {stats['unhealthy_rate']:.2%}")


def demo_4_generate_with_repair(agent):
    """Demo 4: Generation with REPAIR"""
    print_section("Demo 4: Generation with REPAIR")
    
    if not agent.enable_repair:
        print("⚠ REPAIR not enabled, skipping demo")
        return
    
    query = "What is the speed of light?"
    
    print(f"Query: {query}")
    print("Generating response with automatic correction...")
    
    # Generate with REPAIR
    response = agent.generate_with_repair(query)
    
    print(f"\n✓ Response: {response[:150]}...")
    
    # Get statistics
    stats = agent.get_repair_statistics()
    
    print(f"\nREPAIR Statistics:")
    print(f"  Edits applied: {stats['editor_stats'].get('total_edits', 0)}")
    print(f"  Inferences: {stats['inference_stats'].get('total_inferences', 0)}")
    print(f"  Health checks: {stats['health_stats'].get('total_checks', 0)}")


def demo_5_batch_editing(agent):
    """Demo 5: Batch editing"""
    print_section("Demo 5: Batch Editing")
    
    if not agent.enable_repair:
        print("⚠ REPAIR not enabled, skipping demo")
        return
    
    # Prepare multiple edits
    edits = [
        ("What is Python?", "Python is a high-level programming language.", "What is Java?"),
        ("What is AI?", "AI is artificial intelligence.", "What is ML?"),
        ("What is quantum?", "Quantum refers to quantum mechanics.", "What is classical?")
    ]
    
    print(f"Applying {len(edits)} edits in batch...")
    
    # Apply all edits
    agent.editor.apply_edits(edits)
    
    # Get statistics
    stats = agent.editor.get_edit_statistics()
    
    print(f"\n✓ Batch editing complete")
    print(f"  Total edits: {stats['total_edits']}")
    print(f"  Successful: {stats['successful_edits']}")
    print(f"  Success rate: {stats['success_rate']:.2%}")
    print(f"  Avg reliability: {stats['avg_reliability']:.3f}")
    print(f"  Avg locality: {stats['avg_locality']:.3f}")


def demo_6_repair_metrics(agent):
    """Demo 6: REPAIR metrics analysis"""
    print_section("Demo 6: REPAIR Metrics Analysis")
    
    if not agent.enable_repair:
        print("⚠ REPAIR not enabled, skipping demo")
        return
    
    # Get comprehensive statistics
    stats = agent.get_repair_statistics()
    
    print("REPAIR Metrics Summary:")
    print("\n1. Editor Statistics:")
    editor_stats = stats.get('editor_stats', {})
    print(f"   Total edits: {editor_stats.get('total_edits', 0)}")
    print(f"   Successful edits: {editor_stats.get('successful_edits', 0)}")
    print(f"   Success rate: {editor_stats.get('success_rate', 0):.2%}")
    print(f"   Avg reliability: {editor_stats.get('avg_reliability', 0):.3f}")
    print(f"   Avg locality: {editor_stats.get('avg_locality', 0):.3f}")
    print(f"   Avg generalization: {editor_stats.get('avg_generalization', 0):.3f}")
    
    print("\n2. Inference Statistics:")
    inference_stats = stats.get('inference_stats', {})
    print(f"   Total inferences: {inference_stats.get('total_inferences', 0)}")
    print(f"   Corrections applied: {inference_stats.get('corrections_applied', 0)}")
    print(f"   Correction rate: {inference_stats.get('correction_rate', 0):.2%}")
    
    print("\n3. Health Check Statistics:")
    health_stats = stats.get('health_stats', {})
    print(f"   Total checks: {health_stats.get('total_checks', 0)}")
    print(f"   Unhealthy count: {health_stats.get('unhealthy_count', 0)}")
    print(f"   Unhealthy rate: {health_stats.get('unhealthy_rate', 0):.2%}")


def demo_7_fallback_behavior():
    """Demo 7: Fallback behavior without REPAIR"""
    print_section("Demo 7: Fallback Behavior (REPAIR Disabled)")
    
    from multilingual_research_agent import MultilingualResearchAgent, Language
    
    # Create agent without REPAIR
    agent = MultilingualResearchAgent(
        supported_languages=[Language.ENGLISH],
        quantum_enabled=False,
        enable_repair=False
    )
    
    print(f"✓ Agent created without REPAIR: {not agent.enable_repair}")
    print(f"✓ Editor: {agent.editor}")
    print(f"✓ Inference wrapper: {agent.inference}")
    print(f"✓ Health checker: {agent.health_checker}")
    
    # Try to get statistics
    stats = agent.get_repair_statistics()
    print(f"\n✓ Statistics call handled gracefully:")
    print(f"  {stats}")


def main():
    """Run all demos"""
    print("\n" + "=" * 70)
    print("  REPAIR Model Editing Integration Demo")
    print("  Version 1.1.0")
    print("=" * 70)
    
    try:
        # Demo 1: Basic initialization
        agent = demo_1_basic_repair()
        
        # Demo 2: Manual editing
        demo_2_manual_edit(agent)
        
        # Demo 3: Health checking
        demo_3_health_checking(agent)
        
        # Demo 4: Generate with REPAIR
        demo_4_generate_with_repair(agent)
        
        # Demo 5: Batch editing
        demo_5_batch_editing(agent)
        
        # Demo 6: Metrics analysis
        demo_6_repair_metrics(agent)
        
        # Demo 7: Fallback behavior
        demo_7_fallback_behavior()
        
        # Summary
        print_section("Demo Complete")
        print("✓ All demos completed successfully!")
        print("\nNext steps:")
        print("  1. Run integration tests: python test_model_editing_integration.py")
        print("  2. Read documentation: MODEL_EDITING_README.md")
        print("  3. Check examples: QUICK_START_GUIDE.md")
        print("  4. Enable REPAIR: export ENABLE_REPAIR=true")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n✗ Demo failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
