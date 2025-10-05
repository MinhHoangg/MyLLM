"""
DNotitia Model Wrapper: Interface for language models with automatic fallback.
Supports multiple models optimized for Mac's memory constraints.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DNotitiaModel:
    """
    Wrapper for language models with automatic fallback support.
    
    Primary: dnotitia/DNA-2.0-8B (8B parameters, requires approval)
    Fallback: LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct (7.8B parameters, open access)
    """
    
    # Model configurations in priority order
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
        model_name: str = "dnotitia/DNA-2.0-1.7B",
        device: Optional[str] = None,
        max_length: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        hf_token: Optional[str] = None,
        use_fallback: bool = True
    ):
        """
        Initialize the model with automatic fallback support.
        
        Args:
            model_name: Hugging Face model identifier (primary model to try)
            device: Device to run model on ('cuda', 'mps', 'cpu', or None for auto)
            max_length: Maximum sequence length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            hf_token: Hugging Face authentication token (for gated models)
            use_fallback: If True, automatically fallback to EXAONE if primary fails
        """
        self.requested_model = model_name
        self.model_name = None  # Will be set after successful load
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.hf_token = hf_token
        self.use_fallback = use_fallback
        
        # Determine device with MPS support for Mac
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        logger.info(f"Target device: {self.device}")
        
        # Try to load model with fallback
        self.tokenizer = None
        self.model = None
        self._load_model_with_fallback()
    
    def _load_model_with_fallback(self):
        """Try loading primary model, fallback to other models if it fails."""
        # Determine which models to try
        models_to_try = []
        
        # Add requested model first
        models_to_try.append(self.requested_model)
        
        # Add fallback models if enabled
        if self.use_fallback:
            for config in self.MODEL_CONFIGS:
                if config["name"] != self.requested_model:
                    models_to_try.append(config["name"])
        
        # Try each model in order
        last_error = None
        for model_name in models_to_try:
            try:
                logger.info(f"Attempting to load: {model_name}")
                self._load_model(model_name)
                self.model_name = model_name
                logger.info(f"✅ Successfully loaded: {model_name}")
                return
            except Exception as e:
                last_error = e
                error_str = str(e)
                logger.warning(f"❌ Failed to load {model_name}: {error_str}")
                
                # Check if it's an authentication error
                if "401" in error_str or "403" in error_str or "gated" in error_str.lower():
                    logger.info(f"⚠️ {model_name} requires approval or authentication")
                    if self.use_fallback:
                        logger.info("→ Trying fallback model...")
                        continue
                # Check if it's a memory error
                elif "out of memory" in error_str.lower() or "oom" in error_str.lower():
                    logger.warning(f"⚠️ {model_name} requires too much memory for {self.device}")
                    
                    # Try CPU if we were using MPS/CUDA
                    if self.device in ["mps", "cuda"]:
                        logger.info("→ Attempting to load on CPU instead...")
                        original_device = self.device
                        self.device = "cpu"
                        try:
                            self._load_model(model_name)
                            self.model_name = model_name
                            logger.info(f"✅ Successfully loaded {model_name} on CPU")
                            logger.warning(f"Note: Running on CPU will be slower than {original_device}")
                            return
                        except Exception as cpu_error:
                            logger.warning(f"❌ CPU loading also failed: {str(cpu_error)}")
                            self.device = original_device  # Restore original device
                    
                    if self.use_fallback:
                        logger.info("→ Trying next fallback model...")
                        continue
                else:
                    logger.error(f"→ Error: {error_str}")
                    if self.use_fallback:
                        continue
        
        # If we got here, all models failed
        error_msg = f"Failed to load any model. Last error: {str(last_error)}"
        if "401" in str(last_error) or "403" in str(last_error):
            error_msg += "\n\n💡 The primary model requires Hugging Face authentication."
            error_msg += "\nPlease provide a valid HF token or wait for model approval."
        elif "out of memory" in str(last_error).lower():
            error_msg += "\n\n💡 All models require too much memory."
            error_msg += "\n\nSolutions:"
            error_msg += "\n1. Close other applications to free up memory"
            error_msg += "\n2. Use a smaller model by editing MODEL_CONFIGS"
            error_msg += "\n3. Run with device='cpu' (slower but uses less memory)"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    def _load_model(self, model_name: str):
        """
        Load a specific model.
        
        Args:
            model_name: Model identifier to load
            
        Raises:
            Exception if loading fails
        """
        logger.info(f"Loading model: {model_name} on {self.device}...")
        
        # Prepare authentication kwargs
        auth_kwargs = {}
        if self.hf_token:
            auth_kwargs["token"] = self.hf_token
            logger.info("Using provided HF token for authentication")
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            **auth_kwargs
        )
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with appropriate dtype
        logger.info("Loading model weights...")
        dtype = torch.float16 if self.device in ["cuda", "mps"] else torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            **auth_kwargs
        )
        
        # Move to device
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"✅ Model loaded successfully on {self.device}")
    
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
        # Use default parameters if not specified
        temp = temperature if temperature is not None else self.temperature
        top_p_val = top_p if top_p is not None else self.top_p
        
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
        
        return generated_text
    
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> List[str]:
        """
        Generate text for multiple prompts in batch.
        
        Args:
            prompts: List of input text prompts
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            List of generated texts
        """
        results = []
        for prompt in prompts:
            result = self.generate(
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
        
        Args:
            prompt: Input text prompt
            **kwargs: Additional arguments for generate()
            
        Returns:
            Generated text
        """
        return self.generate(prompt, **kwargs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        # Find config for current model
        model_config = next(
            (cfg for cfg in self.MODEL_CONFIGS if cfg["name"] == self.model_name),
            None
        )
        
        # Ensure model_name is always a string
        model_name_str = str(self.model_name) if self.model_name is not None else "Unknown"
        
        info = {
            'model_name': model_name_str,
            'requested_model': self.requested_model,
            'device': self.device,
            'max_length': self.max_length,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'vocab_size': self.tokenizer.vocab_size if hasattr(self.tokenizer, 'vocab_size') else None
        }
        
        # Add model-specific info if available
        if model_config:
            info['description'] = model_config['description']
            info['requires_approval'] = model_config['requires_approval']
            info['size_gb'] = model_config['size_gb']
            info['is_fallback'] = self.model_name != self.requested_model
        
        return info
    
    @classmethod
    def list_available_models(cls) -> List[Dict[str, Any]]:
        """
        Get list of available model configurations.
        
        Returns:
            List of model configuration dictionaries
        """
        return cls.MODEL_CONFIGS.copy()
