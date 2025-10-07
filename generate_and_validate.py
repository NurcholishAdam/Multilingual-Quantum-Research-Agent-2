# -*- coding: utf-8 -*-
"""
Generate and Validate with REPAIR

Integrates model generation with health checking and automatic repair.
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def generate_and_validate(
    agent,
    query: str,
    use_repair: bool = True,
    max_retries: int = 2
) -> str:
    """
    Generate response with health checking and automatic repair.
    
    Workflow:
    1. Generate response with current model
    2. Health check for hallucination or outdated facts
    3. If unhealthy, apply REPAIR edit and regenerate
    4. Return validated response
    
    Args:
        agent: MultilingualResearchAgent instance with editor and health_checker
        query: Input query
        use_repair: Whether to use REPAIR for corrections
        max_retries: Maximum number of repair attempts
    
    Returns:
        Validated response string
    """
    logger.info(f"Generate and validate: {query[:50]}...")
    
    # Check if agent has required components
    if not hasattr(agent, 'inference'):
        logger.warning("Agent missing inference wrapper, using fallback")
        return _fallback_generate(agent, query)
    
    if not hasattr(agent, 'health_checker'):
        logger.warning("Agent missing health checker, skipping validation")
        return agent.inference(query)
    
    # Step 1: Generate with current model
    response = agent.inference(query)
    logger.debug(f"Initial response: {response[:100]}...")
    
    # Step 2: Health check
    is_healthy = agent.health_checker.is_healthy(query, response)
    
    if is_healthy:
        logger.info("Response passed health check")
        return response
    
    # Step 3: Apply REPAIR if unhealthy
    if not use_repair:
        logger.warning("Response failed health check but REPAIR disabled")
        return response
    
    logger.warning("Response failed health check, applying REPAIR")
    
    for attempt in range(max_retries):
        try:
            # Fetch correct answer
            correct_answer = agent.fetch_correct_answer(query)
            
            # Sample unrelated prompt for locality
            locality_prompt = agent.sample_unrelated_prompt()
            
            # Prepare edit
            edits = [(query, correct_answer, locality_prompt)]
            
            # Apply REPAIR edit
            logger.info(f"Applying REPAIR edit (attempt {attempt + 1}/{max_retries})")
            agent.editor.apply_edits(edits)
            
            # Regenerate with updated model
            response = agent.inference(query)
            
            # Re-check health
            is_healthy = agent.health_checker.is_healthy(query, response)
            
            if is_healthy:
                logger.info(f"Response corrected after {attempt + 1} attempts")
                return response
            
        except Exception as e:
            logger.error(f"REPAIR attempt {attempt + 1} failed: {e}")
    
    logger.warning(f"Failed to correct response after {max_retries} attempts")
    return response


def _fallback_generate(agent, query: str) -> str:
    """Fallback generation without REPAIR"""
    try:
        if hasattr(agent, 'model'):
            return agent.model.generate(query)
        else:
            return f"Fallback response for: {query}"
    except Exception as e:
        logger.error(f"Fallback generation failed: {e}")
        return f"Error: {str(e)}"


class HealthChecker:
    """
    Health checker for model responses.
    
    Detects:
    - Hallucinations
    - Outdated facts
    - Inconsistencies
    """
    
    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize health checker.
        
        Args:
            confidence_threshold: Minimum confidence for healthy response
        """
        self.confidence_threshold = confidence_threshold
        self.check_count = 0
        self.unhealthy_count = 0
    
    def is_healthy(self, query: str, response: str) -> bool:
        """
        Check if response is healthy.
        
        Args:
            query: Input query
            response: Model response
        
        Returns:
            True if response is healthy, False otherwise
        """
        self.check_count += 1
        
        # Placeholder health checks
        # In production, would use:
        # - Fact verification against knowledge base
        # - Consistency checking
        # - Confidence scoring
        
        # Simple heuristics for demo
        is_healthy = True
        
        # Check 1: Response not empty
        if not response or len(response.strip()) < 10:
            is_healthy = False
        
        # Check 2: Response doesn't contain error markers
        error_markers = ["error", "unknown", "cannot", "don't know"]
        if any(marker in response.lower() for marker in error_markers):
            is_healthy = False
        
        # Check 3: Response is relevant to query
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        overlap = len(query_words & response_words)
        if overlap < 2:
            is_healthy = False
        
        if not is_healthy:
            self.unhealthy_count += 1
            logger.warning(f"Unhealthy response detected ({self.unhealthy_count}/{self.check_count})")
        
        return is_healthy
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get health check statistics"""
        return {
            "total_checks": self.check_count,
            "unhealthy_count": self.unhealthy_count,
            "unhealthy_rate": self.unhealthy_count / max(1, self.check_count),
            "confidence_threshold": self.confidence_threshold
        }
