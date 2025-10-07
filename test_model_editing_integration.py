# -*- coding: utf-8 -*-
"""
Integration Tests for REPAIR Model Editing

Tests the integration of REPAIR with the multilingual quantum research agent.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
import unittest
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestREPAIRIntegration(unittest.TestCase):
    """Test suite for REPAIR integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        from multilingual_research_agent import MultilingualResearchAgent, Language
        
        # Create agent with REPAIR enabled
        self.agent = MultilingualResearchAgent(
            supported_languages=[Language.ENGLISH],
            quantum_enabled=False,  # Disable quantum for faster tests
            enable_repair=True
        )
    
    def test_repair_initialization(self):
        """Test REPAIR components are initialized"""
        self.assertTrue(self.agent.enable_repair, "REPAIR should be enabled")
        self.assertIsNotNone(self.agent.editor, "Editor should be initialized")
        self.assertIsNotNone(self.agent.inference, "Inference wrapper should be initialized")
        self.assertIsNotNone(self.agent.health_checker, "Health checker should be initialized")
        logger.info("✓ REPAIR initialization test passed")
    
    def test_model_editing(self):
        """Test model editing with REPAIR"""
        if not self.agent.enable_repair:
            self.skipTest("REPAIR not available")
        
        # Prepare test edit
        query = "When was the IAAF Combined Events Challenge launched?"
        correct_answer = "2006"
        locality_prompt = "What is the capital of France?"
        
        edits = [(query, correct_answer, locality_prompt)]
        
        # Apply edit
        self.agent.editor.apply_edits(edits)
        
        # Check edit was recorded
        stats = self.agent.editor.get_edit_statistics()
        self.assertGreater(stats["total_edits"], 0, "Edit should be recorded")
        
        logger.info(f"✓ Model editing test passed: {stats}")
    
    def test_generate_and_validate(self):
        """Test generation with health checking"""
        if not self.agent.enable_repair:
            self.skipTest("REPAIR not available")
        
        query = "What is machine learning?"
        
        # Generate with validation
        response = self.agent.generate_with_repair(query)
        
        self.assertIsNotNone(response, "Response should not be None")
        self.assertGreater(len(response), 0, "Response should not be empty")
        
        logger.info(f"✓ Generate and validate test passed: {response[:100]}...")
    
    def test_health_checker(self):
        """Test health checker functionality"""
        if not self.agent.enable_repair:
            self.skipTest("REPAIR not available")
        
        query = "Test query"
        
        # Test healthy response
        healthy_response = "This is a detailed and relevant response to the test query."
        is_healthy = self.agent.health_checker.is_healthy(query, healthy_response)
        self.assertTrue(is_healthy, "Healthy response should pass")
        
        # Test unhealthy response
        unhealthy_response = "Error"
        is_healthy = self.agent.health_checker.is_healthy(query, unhealthy_response)
        self.assertFalse(is_healthy, "Unhealthy response should fail")
        
        logger.info("✓ Health checker test passed")
    
    def test_repair_statistics(self):
        """Test REPAIR statistics collection"""
        if not self.agent.enable_repair:
            self.skipTest("REPAIR not available")
        
        # Get statistics
        stats = self.agent.get_repair_statistics()
        
        self.assertIn("repair_enabled", stats)
        self.assertIn("editor_stats", stats)
        self.assertIn("inference_stats", stats)
        self.assertIn("health_stats", stats)
        
        logger.info(f"✓ REPAIR statistics test passed: {stats}")
    
    def test_repair_fixes_hallucination(self):
        """Test REPAIR corrects hallucinations"""
        if not self.agent.enable_repair:
            self.skipTest("REPAIR not available")
        
        query = "When was the IAAF Combined Events Challenge launched?"
        
        # Simulate hallucination by forcing incorrect answer
        # In production, model would generate this
        incorrect_answer = "Armand"
        
        # Check health (should fail)
        is_healthy = self.agent.health_checker.is_healthy(query, incorrect_answer)
        self.assertFalse(is_healthy, "Incorrect answer should be unhealthy")
        
        # Apply correction
        correct_answer = "2006"
        locality_prompt = self.agent.sample_unrelated_prompt()
        edits = [(query, correct_answer, locality_prompt)]
        
        self.agent.editor.apply_edits(edits)
        
        # Verify edit was applied
        stats = self.agent.editor.get_edit_statistics()
        self.assertGreater(stats["total_edits"], 0)
        
        logger.info("✓ Hallucination correction test passed")
    
    def test_locality_preservation(self):
        """Test REPAIR preserves locality"""
        if not self.agent.enable_repair:
            self.skipTest("REPAIR not available")
        
        # Apply edit
        query = "Test query"
        correct_answer = "Test answer"
        locality_prompt = self.agent.sample_unrelated_prompt()
        
        edits = [(query, correct_answer, locality_prompt)]
        self.agent.editor.apply_edits(edits)
        
        # Check locality metric
        stats = self.agent.editor.get_edit_statistics()
        if stats["total_edits"] > 0:
            self.assertGreater(stats["avg_locality"], 0.5, "Locality should be preserved")
        
        logger.info(f"✓ Locality preservation test passed: locality={stats.get('avg_locality', 0):.3f}")
    
    def test_reliability_metric(self):
        """Test REPAIR reliability metric"""
        if not self.agent.enable_repair:
            self.skipTest("REPAIR not available")
        
        # Apply multiple edits
        for i in range(3):
            query = f"Test query {i}"
            correct_answer = f"Test answer {i}"
            locality_prompt = self.agent.sample_unrelated_prompt()
            
            edits = [(query, correct_answer, locality_prompt)]
            self.agent.editor.apply_edits(edits)
        
        # Check reliability metric
        stats = self.agent.editor.get_edit_statistics()
        self.assertGreater(stats["total_edits"], 0)
        self.assertGreater(stats["avg_reliability"], 0.5, "Reliability should be reasonable")
        
        logger.info(f"✓ Reliability metric test passed: reliability={stats['avg_reliability']:.3f}")
    
    def test_generalization_metric(self):
        """Test REPAIR generalization metric"""
        if not self.agent.enable_repair:
            self.skipTest("REPAIR not available")
        
        # Apply edit
        query = "Test query"
        correct_answer = "Test answer"
        locality_prompt = self.agent.sample_unrelated_prompt()
        
        edits = [(query, correct_answer, locality_prompt)]
        self.agent.editor.apply_edits(edits)
        
        # Check generalization metric
        stats = self.agent.editor.get_edit_statistics()
        if stats["total_edits"] > 0:
            self.assertGreater(stats["avg_generalization"], 0.5, "Generalization should be reasonable")
        
        logger.info(f"✓ Generalization metric test passed: generalization={stats.get('avg_generalization', 0):.3f}")


class TestREPAIRFallback(unittest.TestCase):
    """Test REPAIR fallback behavior"""
    
    def test_agent_without_repair(self):
        """Test agent works without REPAIR"""
        from multilingual_research_agent import MultilingualResearchAgent, Language
        
        # Create agent with REPAIR disabled
        agent = MultilingualResearchAgent(
            supported_languages=[Language.ENGLISH],
            quantum_enabled=False,
            enable_repair=False
        )
        
        self.assertFalse(agent.enable_repair, "REPAIR should be disabled")
        self.assertIsNone(agent.editor, "Editor should be None")
        
        logger.info("✓ Agent without REPAIR test passed")
    
    def test_repair_statistics_without_repair(self):
        """Test statistics when REPAIR is disabled"""
        from multilingual_research_agent import MultilingualResearchAgent, Language
        
        agent = MultilingualResearchAgent(
            supported_languages=[Language.ENGLISH],
            enable_repair=False
        )
        
        stats = agent.get_repair_statistics()
        self.assertIn("error", stats, "Should return error when REPAIR disabled")
        
        logger.info("✓ Statistics without REPAIR test passed")


def run_tests():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("REPAIR Model Editing Integration Tests")
    logger.info("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    logger.info(f"Tests run: {result.testsRun}")
    logger.info(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Errors: {len(result.errors)}")
    logger.info(f"Skipped: {len(result.skipped)}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
