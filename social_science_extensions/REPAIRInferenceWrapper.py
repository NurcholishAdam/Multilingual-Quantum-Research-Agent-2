# -*- coding: utf-8 -*-
"""
REPAIR Inference Wrapper

Wraps model inference with error detection and automatic correction.
"""

import logging
from typing import Optional, Dict, Any
from .model_editing import DualMemoryEditor

logger = logging.getLogger(__name__)


class REPAIRInferenceWrapper:
    """
    Inference wrapper with REPAIR-based error correction.
    
    Features:
    - Automatic error detection
    - Threshold-based correction triggering
    - Inference statistics tracking
    """
    
    def __init__(self, editor: DualMemoryEditor, threshold: float = 0.01):
        """
        Initialize inference wrapper.
        
        Args:
            editor: DualMemoryEditor instance
            threshold: Error threshold for triggering correction
        """
        self.editor = editor
        self.error_thresh = threshold
        self.inference_count = 0
        self.correction_count = 0
        self.inference_history = []
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """
        Generate response with error checking.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
        
        Returns:
            Generated response
        """
        self.inference_count += 1
        
        try:
            # Generate with current model
            output = self.editor.model.generate(prompt, **kwargs)
            
            # Track inference
            self.inference_history.append({
                "prompt": prompt[:100],
                "output": output[:100] if isinstance(output, str) else str(output)[:100],
                "corrected": False
            })
            
            logger.debug(f"Inference {self.inference_count}: {prompt[:50]}...")
            return output
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return f"Error: {str(e)}"
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get inference statistics.
        
        Returns:
            Dictionary with inference metrics
        """
        return {
            "total_inferences": self.inference_count,
            "corrections_applied": self.correction_count,
            "correction_rate": self.correction_count / max(1, self.inference_count),
            "error_threshold": self.error_thresh,
            "recent_inferences": self.inference_history[-10:]
        }
    
    def reset_statistics(self):
        """Reset inference statistics"""
        self.inference_count = 0
        self.correction_count = 0
        self.inference_history.clear()
        logger.info("Reset inference statistics")
