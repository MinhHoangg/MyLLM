"""
Unified Model Manager: Handles primary and fallback models with automatic selection.
"""

from typing import Optional, Dict, Any, Union
import logging
from .dnotitia_primary_model import DNotitiaModel
from .exaone_fallback_model import ExaoneModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    """
    Unified manager for handling primary and fallback models.
    
    This class provides a single interface for model loading with automatic
    fallback from DNotitia DNA-2.0-8B to EXAONE-3.5-7.8B-Instruct.
    
    Benefits of separated models:
    - Easier debugging of specific model issues
    - Clear separation of concerns
    - Individual model optimization
    - Better error handling and logging
    """
    
    def __init__(
        self,
        prefer_primary: bool = True,
        hf_token: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9
    ):
        """
        Initialize the model manager.
        
        Args:
            prefer_primary: If True, try primary model first
            hf_token: Hugging Face token for primary model
            device: Device preference ('cuda', 'mps', 'cpu', or None)
            max_length: Maximum sequence length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        """
        self.prefer_primary = prefer_primary
        self.hf_token = hf_token
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        
        # Model instances
        self.primary_model: Optional[DNotitiaModel] = None
        self.fallback_model: Optional[ExaoneModel] = None
        self.active_model: Optional[Union[DNotitiaModel, ExaoneModel]] = None
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """
        Load models with automatic fallback.
        
        Loading strategy:
        1. If prefer_primary: Try DNotitia → EXAONE
        2. If not prefer_primary: Try EXAONE → DNotitia
        """
        logger.info("🚀 Starting model loading with automatic fallback...")
        
        if self.prefer_primary:
            self._try_primary_then_fallback()
        else:
            self._try_fallback_then_primary()
        
        if self.active_model is None:
            raise RuntimeError("❌ Failed to load any model (both primary and fallback failed)")
        
        # Log final status
        model_info = self.active_model.get_model_info()
        logger.info(f"✅ Active model: {model_info['model_name']}")
        logger.info(f"   Type: {model_info['model_type']}")
        logger.info(f"   Device: {model_info['device']}")
    
    def _try_primary_then_fallback(self):
        """Try primary model first, then fallback."""
        logger.info("🎯 Strategy: Primary (DNotitia) → Fallback (EXAONE)")
        
        # Try primary model
        try:
            logger.info("📍 Step 1: Attempting to load primary model (DNotitia DNA-2.0-8B)")
            self.primary_model = DNotitiaModel(
                device=self.device,
                max_length=self.max_length,
                temperature=self.temperature,
                top_p=self.top_p,
                hf_token=self.hf_token
            )
            self.active_model = self.primary_model
            logger.info("🎉 Primary model loaded successfully!")
            return
            
        except Exception as e:
            logger.warning(f"⚠️ Primary model failed: {str(e)}")
            logger.info("🔄 Falling back to EXAONE model...")
        
        # Try fallback model
        try:
            logger.info("📍 Step 2: Attempting to load fallback model (EXAONE-3.5-7.8B)")
            self.fallback_model = ExaoneModel(
                device=self.device,
                max_length=self.max_length,
                temperature=self.temperature,
                top_p=self.top_p
            )
            self.active_model = self.fallback_model
            logger.info("🎉 Fallback model loaded successfully!")
            logger.info("ℹ️ Using fallback model (primary model unavailable)")
            
        except Exception as e:
            logger.error(f"❌ Fallback model also failed: {str(e)}")
            self.active_model = None
    
    def _try_fallback_then_primary(self):
        """Try fallback model first, then primary."""
        logger.info("🎯 Strategy: Fallback (EXAONE) → Primary (DNotitia)")
        
        # Try fallback model
        try:
            logger.info("📍 Step 1: Attempting to load fallback model (EXAONE-3.5-7.8B)")
            self.fallback_model = ExaoneModel(
                device=self.device,
                max_length=self.max_length,
                temperature=self.temperature,
                top_p=self.top_p
            )
            self.active_model = self.fallback_model
            logger.info("🎉 Fallback model loaded successfully!")
            return
            
        except Exception as e:
            logger.warning(f"⚠️ Fallback model failed: {str(e)}")
            logger.info("🔄 Trying primary model...")
        
        # Try primary model
        try:
            logger.info("📍 Step 2: Attempting to load primary model (DNotitia DNA-2.0-8B)")
            self.primary_model = DNotitiaModel(
                device=self.device,
                max_length=self.max_length,
                temperature=self.temperature,
                top_p=self.top_p,
                hf_token=self.hf_token
            )
            self.active_model = self.primary_model
            logger.info("🎉 Primary model loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Primary model also failed: {str(e)}")
            self.active_model = None
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate text using the active model.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
            
        Raises:
            RuntimeError: If no model is loaded
        """
        if self.active_model is None:
            raise RuntimeError("No model loaded. Cannot generate text.")
        
        return self.active_model.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """Callable interface."""
        return self.generate(prompt, **kwargs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the active model.
        
        Returns:
            Dictionary with model information
        """
        if self.active_model is None:
            return {
                'model_name': 'None',
                'model_type': 'none',
                'status': 'not_loaded',
                'error': 'No model loaded'
            }
        
        info = self.active_model.get_model_info()
        
        # Add manager-specific info
        info['is_fallback'] = isinstance(self.active_model, ExaoneModel)
        info['primary_available'] = self.primary_model is not None
        info['fallback_available'] = self.fallback_model is not None
        
        return info
    
    def switch_model(self, use_primary: bool = True) -> bool:
        """
        Switch between primary and fallback models.
        
        Args:
            use_primary: If True, switch to primary; if False, switch to fallback
            
        Returns:
            True if switch successful, False otherwise
        """
        try:
            if use_primary and self.primary_model is not None:
                self.active_model = self.primary_model
                logger.info("🔄 Switched to primary model (DNotitia)")
                return True
            elif not use_primary and self.fallback_model is not None:
                self.active_model = self.fallback_model
                logger.info("🔄 Switched to fallback model (EXAONE)")
                return True
            else:
                logger.warning("⚠️ Requested model not available")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to switch model: {e}")
            return False
    
    def get_available_models(self) -> Dict[str, bool]:
        """
        Check which models are available.
        
        Returns:
            Dictionary showing availability of each model
        """
        return {
            'primary': self.primary_model is not None,
            'fallback': self.fallback_model is not None,
            'active': self.active_model is not None
        }
    
    def reload_models(self):
        """
        Reload all models (useful for debugging).
        """
        logger.info("🔄 Reloading all models...")
        
        # Clear existing models
        self.primary_model = None
        self.fallback_model = None
        self.active_model = None
        
        # Reload
        self._load_models()
    
    @property
    def model_name(self) -> str:
        """Get the name of the active model (for compatibility)."""
        if self.active_model is None:
            return "None"
        return self.active_model.model_name
    
    @classmethod
    def list_available_models(cls) -> Dict[str, Dict[str, Any]]:
        """
        List all available model configurations.
        
        Returns:
            Dictionary with primary and fallback model configs
        """
        return {
            'primary': DNotitiaModel.get_model_config(),
            'fallback': ExaoneModel.get_model_config()
        }
    
    @classmethod
    def check_system_requirements(cls) -> Dict[str, Any]:
        """
        Check system requirements for both models.
        
        Returns:
            Dictionary with requirement check results
        """
        return {
            'primary': DNotitiaModel.check_requirements(),
            'fallback': ExaoneModel.check_requirements()
        }