"""
DNotitia DNA-2.0-8B Model Wrapper: Primary model with authentication handling.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DNotitiaModel:
    """
    Wrapper for DNotitia DNA-2.0-8B model.
    
    This is the primary model that requires Hugging Face authentication.
    - Model: dnotitia/DNA-2.0-8B (8B parameters)
    - Size: ~16GB VRAM
    - Status: Requires approval/authentication
    """
    
    MODEL_CONFIG = {
        "name": "dnotitia/DNA-2.0-8B",
        "description": "DNotitia DNA 2.0 (8B parameters) - Primary model",
        "requires_approval": True,
        "size_gb": 16.0,
        "recommended_device": ["mps", "cuda"],
        "min_vram_gb": 16
    }
    
    def __init__(
        self,
        device: Optional[str] = None,
        max_length: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        hf_token: Optional[str] = None
    ):
        """
        Initialize the DNotitia DNA-2.0-8B model.
        
        Args:
            device: Device to run model on ('cuda', 'mps', 'cpu', or None for auto)
            max_length: Maximum sequence length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            hf_token: Hugging Face authentication token (required for this model)
        """
        self.model_name = self.MODEL_CONFIG["name"]
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.hf_token = hf_token
        
        # Determine device with MPS support for Mac
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
                logger.warning(f"WARNING: Using CPU - this model will be slow without GPU")
        else:
            self.device = device
        
        logger.info(f"🧠 DNotitia DNA-2.0-8B | Target device: {self.device}")
        
        # Initialize model components
        self.tokenizer = None
        self.model = None
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """
        Load the DNotitia DNA-2.0-8B model.
        
        Raises:
            RuntimeError: If authentication fails or model loading fails
        """
        model_name = self.MODEL_CONFIG["name"]
        logger.info(f" Loading DNotitia model: {model_name}")
        
        # Check authentication
        if not self.hf_token:
            logger.warning(f"WARNING: No HF token provided - attempting without authentication")
        
        # Prepare authentication kwargs
        auth_kwargs = {}
        if self.hf_token:
            auth_kwargs["token"] = self.hf_token
            logger.info(f" Using provided HF token for authentication")
        
        try:
            # Load tokenizer
            logger.info(f" Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                **auth_kwargs
            )
            
            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info(f" Set padding token to EOS token")
            
            # Load model with appropriate dtype
            logger.info(f" Loading model weights...")
            dtype = torch.float16 if self.device in ["cuda", "mps"] else torch.float32
            logger.info(f" Using dtype: {dtype}")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                **auth_kwargs
            )
            
            # Move to device
            logger.info(f"📱 Moving model to {self.device}...")
            self.model.to(self.device)
            self.model.eval()
            
            # Log success
            logger.info(f" DNotitia DNA-2.0-8B loaded successfully!")
            logger.info(f" Device: {self.device}")
            logger.info(f" Vocab size: {self.tokenizer.vocab_size:,}")
            logger.info(f" Model parameters: ~8B")
            
        except Exception as e:
            error_str = str(e)
            
            # Specific error handling
            if "401" in error_str or "403" in error_str:
                error_msg = " Authentication failed for DNotitia DNA-2.0-8B"
                error_msg += "\n This model requires approval or a valid HF token"
                error_msg += f"\n🔗 Request access: https://huggingface.co/{model_name}"
                logger.error(error_msg)
                raise RuntimeError(f"Authentication required: {error_str}")
            elif "out of memory" in error_str.lower():
                error_msg = " Out of memory loading DNotitia DNA-2.0-8B"
                error_msg += f"\n This model requires ~{self.MODEL_CONFIG['min_vram_gb']}GB VRAM"
                error_msg += f"\n Try using CPU (slower) or close other applications"
                logger.error(error_msg)
                raise RuntimeError(f"Memory error: {error_str}")
            else:
                logger.error(f" Failed to load DNotitia DNA-2.0-8B: {error_str}")
                raise RuntimeError(f"Model loading failed: {error_str}")
    
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
        Generate text from a prompt using DNotitia DNA-2.0-8B.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (uses default if None)
            top_p: Nucleus sampling parameter (uses default if None)
            do_sample: Whether to use sampling
            num_return_sequences: Number of sequences to generate
            
        Returns:
            Generated text
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        # Use default parameters if not specified
        temp = temperature if temperature is not None else self.temperature
        top_p_val = top_p if top_p is not None else self.top_p
        
        logger.debug(f" Generating with DNA-2.0-8B | temp={temp}, top_p={top_p_val}")
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temp,
                top_p=top_p_val,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from the output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        logger.debug(f" Generated {len(generated_text)} characters")
        return generated_text
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """Callable interface for the model."""
        return self.generate(prompt, **kwargs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the DNotitia model.
        
        Returns:
            Dictionary with model information
        """
        info = {
            'model_name': self.model_name,
            'model_type': 'primary',
            'description': self.MODEL_CONFIG['description'],
            'device': self.device,
            'max_length': self.max_length,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'requires_approval': self.MODEL_CONFIG['requires_approval'],
            'size_gb': self.MODEL_CONFIG['size_gb'],
            'vocab_size': self.tokenizer.vocab_size if self.tokenizer else None,
            'parameters': '8B',
            'status': 'loaded' if self.model else 'not_loaded'
        }
        
        return info
    
    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """Get the model configuration."""
        return cls.MODEL_CONFIG.copy()
    
    @classmethod
    def check_requirements(cls) -> Dict[str, Any]:
        """
        Check if system meets requirements for this model.
        
        Returns:
            Dictionary with requirement check results
        """
        requirements = {
            'model_name': cls.MODEL_CONFIG['name'],
            'size_gb': cls.MODEL_CONFIG['size_gb'],
            'requires_auth': cls.MODEL_CONFIG['requires_approval'],
            'gpu_available': torch.cuda.is_available() or torch.backends.mps.is_available(),
            'recommended_device': cls.MODEL_CONFIG['recommended_device']
        }
        
        # Check available memory (approximate)
        if torch.backends.mps.is_available():
            requirements['device_type'] = 'mps'
        elif torch.cuda.is_available():
            requirements['device_type'] = 'cuda'
            # Could add CUDA memory check here
        else:
            requirements['device_type'] = 'cpu'
            requirements['warning'] = 'CPU inference will be very slow for 8B model'
        
        return requirements