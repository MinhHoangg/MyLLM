"""
DNotitia Model Wrapper: Backward-compatible interface using the new model manager.

This file maintains compatibility with existing code while using the new
separated model architecture for better debugging and maintenance.

NEW ARCHITECTURE:
- dnotitia_primary_model.py: Handles DNotitia DNA-2.0-8B (primary)
- exaone_fallback_model.py: Handles EXAONE-3.5-7.8B (fallback)  
- model_manager.py: Unified manager with automatic fallback
- dnotitia_model.py: Backward-compatible wrapper (this file)

Benefits:
- Easier debugging of specific model issues
- Clear separation of concerns
- Individual model optimization
- Better error handling and logging
"""

from typing import List, Optional, Dict, Any
import logging
from .model_manager import ModelManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DNotitiaModel:
    """
    Backward-compatible wrapper around the new ModelManager.
    
    This maintains the same interface as before while using the improved
    separated model architecture underneath.
    """
    
    # Keep the same MODEL_CONFIGS for compatibility
    MODEL_CONFIGS = [
        {
            "name": "dnotitia/DNA-2.0-8B",
            "description": "DNotitia DNA 2.0 (8B parameters)",
            "requires_approval": True,
            "size_gb": 16.0
        },
        {
            "name": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
            "description": "LG AI EXAONE 3.5 (7.8B parameters) - Open Access, Instruction-tuned",
            "requires_approval": False,
            "size_gb": 15.6
        }
    ]
    
    def __init__(
        self,
        model_name: str = "dnotitia/DNA-2.0-8B",
        device: Optional[str] = None,
        max_length: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        hf_token: Optional[str] = None,
        use_fallback: bool = True
    ):
        """
        Initialize the model with automatic fallback support.
        
        This now uses the new ModelManager internally but maintains
        the same external interface for backward compatibility.
        """
        logger.info("🔄 Using new separated model architecture")
        logger.info(f"   Primary: dnotitia_primary_model.py")
        logger.info(f"   Fallback: exaone_fallback_model.py")
        logger.info(f"   Manager: model_manager.py")
        
        # Store parameters for compatibility
        self.requested_model = model_name
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.hf_token = hf_token
        self.use_fallback = use_fallback
        
        # Determine preference: prefer primary if requesting DNotitia model
        prefer_primary = (model_name == "dnotitia/DNA-2.0-8B")
        
        try:
            # Initialize the model manager
            self.manager = ModelManager(
                prefer_primary=prefer_primary,
                hf_token=hf_token,
                device=device,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p
            )
            
            # Set compatibility attributes
            model_info = self.manager.get_model_info()
            
            # CRITICAL: Ensure model_name is ALWAYS a string (fix for split error)
            raw_model_name = model_info.get('model_name', 'Unknown')
            if not isinstance(raw_model_name, str):
                logger.error(f"⚠️ BUG FOUND: model_name is {type(raw_model_name)}, not str!")
                logger.error(f"   Value: {repr(raw_model_name)}")
                self.model_name = str(raw_model_name) if raw_model_name else "Unknown"
            else:
                self.model_name = raw_model_name
            
            self.device = model_info.get('device', 'cpu')
            
            logger.info(f"✅ Model wrapper initialized")
            logger.info(f"   Active model: {self.model_name} (type: {type(self.model_name).__name__})")
            logger.info(f"   Device: {self.device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize model manager: {e}")
            # For compatibility, still set these attributes
            self.manager = None
            self.model_name = None
            self.device = device or "cpu"
            raise
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        num_return_sequences: int = 1
    ) -> str:
        """
        Generate text from a prompt.
        
        This delegates to the ModelManager while maintaining the same interface.
        """
        if self.manager is None:
            raise RuntimeError("Model manager not initialized")
        
        return self.manager.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
            # Note: do_sample and num_return_sequences are handled internally
        )
    
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> List[str]:
        """
        Generate text for multiple prompts in batch.
        """
        if self.manager is None:
            raise RuntimeError("Model manager not initialized")
        
        results = []
        for prompt in prompts:
            result = self.manager.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )
            results.append(result)
        
        return results
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """
        Callable interface for the model.
        """
        return self.generate(prompt, **kwargs)
    
    def __str__(self) -> str:
        """String representation to prevent accidental usage in string operations."""
        return f"DNotitiaModel({self.model_name})"
    
    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return f"DNotitiaModel(model_name={self.model_name!r}, device={self.device!r})"
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        This returns the same format as before for compatibility.
        """
        if self.manager is None:
            return {
                'model_name': "Unknown",
                'requested_model': self.requested_model,
                'device': self.device,
                'max_length': self.max_length,
                'temperature': self.temperature,
                'top_p': self.top_p,
                'vocab_size': None,
                'status': 'failed'
            }
        
        # Get info from manager and adapt to old format
        info = self.manager.get_model_info()
        
        # Find config for current model (for backward compatibility)
        model_config = next(
            (cfg for cfg in self.MODEL_CONFIGS if cfg["name"] == info['model_name']),
            None
        )
        
        # Return in the expected format
        result = {
            'model_name': info['model_name'],
            'requested_model': self.requested_model,
            'device': info['device'],
            'max_length': info['max_length'],
            'temperature': info['temperature'],
            'top_p': info['top_p'],
            'vocab_size': info.get('vocab_size')
        }
        
        # Add model config info if available
        if model_config:
            result['description'] = model_config['description']
            result['requires_approval'] = model_config['requires_approval']
            result['size_gb'] = model_config['size_gb']
            result['is_fallback'] = info.get('is_fallback', False)
        
        return result
    
    @classmethod
    def list_available_models(cls) -> List[Dict[str, Any]]:
        """
        Get list of available model configurations.
        """
        return cls.MODEL_CONFIGS.copy()
    
    # New debugging methods (not in original interface)
    def get_manager_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the model manager (for debugging).
        """
        if self.manager is None:
            return {'status': 'not_initialized'}
        
        return {
            'manager_status': 'initialized',
            'available_models': self.manager.get_available_models(),
            'active_model_info': self.manager.get_model_info(),
            'system_requirements': ModelManager.check_system_requirements()
        }
    
    def switch_model(self, use_primary: bool = True) -> bool:
        """
        Switch between primary and fallback models (for debugging).
        """
        if self.manager is None:
            return False
        
        success = self.manager.switch_model(use_primary)
        if success:
            # Update compatibility attributes
            model_info = self.manager.get_model_info()
            self.model_name = model_info['model_name']
        
        return success
