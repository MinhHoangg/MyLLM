"""
EXAONE-3.5-7.8B-Instruct Model Wrapper: Fallback model with open access.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExaoneModel:
    """
    Wrapper for LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct model.
    
    This is the fallback model that doesn't require authentication.
    - Model: LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct (7.8B parameters)
    - Size: ~15.6GB VRAM
    - Status: Open access, no approval needed
    - Optimized: Instruction-tuned for better responses
    """
    
    MODEL_CONFIG = {
        "name": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
        "description": "LG AI EXAONE 3.5 (7.8B parameters) - Fallback model, Instruction-tuned",
        "requires_approval": False,
        "size_gb": 15.6,
        "recommended_device": ["mps", "cuda"],
        "min_vram_gb": 16,
        "features": ["instruction_tuned", "open_access", "multilingual"]
    }
    
    def __init__(
        self,
        device: Optional[str] = None,
        max_length: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9
    ):
        """
        Initialize the EXAONE-3.5-7.8B-Instruct model.
        
        Args:
            device: Device to run model on ('cuda', 'mps', 'cpu', or None for auto)
            max_length: Maximum sequence length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        """
        self.model_name = self.MODEL_CONFIG["name"]
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        
        # Determine device with MPS support for Mac
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
                logger.warning("⚠️ Using CPU - this model will be slow without GPU")
        else:
            self.device = device
        
        logger.info(f"🚀 EXAONE-3.5-7.8B-Instruct | Target device: {self.device}")
        
        # Initialize model components
        self.tokenizer = None
        self.model = None
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """
        Load the EXAONE-3.5-7.8B-Instruct model.
        
        Raises:
            RuntimeError: If model loading fails
        """
        model_name = self.MODEL_CONFIG["name"]
        logger.info(f"🔄 Loading EXAONE model: {model_name}")
        logger.info("🔓 No authentication required (open access)")
        
        try:
            # Load tokenizer
            logger.info("📝 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("🔧 Set padding token to EOS token")
            
            # Load model with appropriate dtype
            logger.info("🚀 Loading model weights...")
            dtype = torch.float16 if self.device in ["cuda", "mps"] else torch.float32
            logger.info(f"📊 Using dtype: {dtype}")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=dtype,
                low_cpu_mem_usage=True
            )
            
            # Move to device
            logger.info(f"📱 Moving model to {self.device}...")
            self.model.to(self.device)
            self.model.eval()
            
            # Log success
            logger.info("✅ EXAONE-3.5-7.8B-Instruct loaded successfully!")
            logger.info(f"   Device: {self.device}")
            logger.info(f"   Vocab size: {self.tokenizer.vocab_size:,}")
            logger.info(f"   Model parameters: ~7.8B")
            logger.info("   🎯 Instruction-tuned for better responses")
            
        except Exception as e:
            error_str = str(e)
            
            # Specific error handling
            if "out of memory" in error_str.lower():
                error_msg = "💾 Out of memory loading EXAONE-3.5-7.8B"
                error_msg += f"\n📊 This model requires ~{self.MODEL_CONFIG['min_vram_gb']}GB VRAM"
                error_msg += f"\n💡 Try using CPU (slower) or close other applications"
                logger.error(error_msg)
                raise RuntimeError(f"Memory error: {error_str}")
            elif "connection" in error_str.lower() or "timeout" in error_str.lower():
                error_msg = "🌐 Network error loading EXAONE-3.5-7.8B"
                error_msg += "\n💡 Check internet connection or try again later"
                logger.error(error_msg)
                raise RuntimeError(f"Network error: {error_str}")
            else:
                logger.error(f"❌ Failed to load EXAONE-3.5-7.8B: {error_str}")
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
        Generate text from a prompt using EXAONE-3.5-7.8B-Instruct.
        
        This model is instruction-tuned, so it works better with structured prompts.
        
        Args:
            prompt: Input text prompt (can be conversational)
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
        
        logger.debug(f"🤖 Generating with EXAONE-3.5-7.8B | temp={temp}, top_p={top_p_val}")
        
        # Format prompt for instruction-tuned model
        formatted_prompt = self._format_instruction_prompt(prompt)
        
        # Tokenize input
        inputs = self.tokenizer(
            formatted_prompt,
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
        
        # Remove the formatted prompt from the output
        if generated_text.startswith(formatted_prompt):
            generated_text = generated_text[len(formatted_prompt):].strip()
        
        # Clean up instruction formatting artifacts
        generated_text = self._clean_instruction_output(generated_text)
        
        logger.debug(f"✅ Generated {len(generated_text)} characters")
        return generated_text
    
    def _format_instruction_prompt(self, prompt: str) -> str:
        """
        Format prompt for instruction-tuned model.
        
        EXAONE works better with structured instruction format.
        
        Args:
            prompt: Original prompt
            
        Returns:
            Formatted prompt
        """
        # Check if prompt already looks like an instruction
        if any(marker in prompt.lower() for marker in ["### instruction", "human:", "user:", "question:"]):
            return prompt
        
        # Format as instruction
        formatted = f"### Human: {prompt}\n\n### Assistant:"
        return formatted
    
    def _clean_instruction_output(self, text: str) -> str:
        """
        Clean up instruction formatting artifacts from output.
        
        Args:
            text: Generated text
            
        Returns:
            Cleaned text
        """
        # Remove common instruction artifacts
        text = text.replace("### Human:", "").replace("### Assistant:", "")
        text = text.replace("Human:", "").replace("Assistant:", "")
        
        # Clean up extra whitespace
        text = text.strip()
        
        return text
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """Callable interface for the model."""
        return self.generate(prompt, **kwargs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the EXAONE model.
        
        Returns:
            Dictionary with model information
        """
        info = {
            'model_name': self.model_name,
            'model_type': 'fallback',
            'description': self.MODEL_CONFIG['description'],
            'device': self.device,
            'max_length': self.max_length,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'requires_approval': self.MODEL_CONFIG['requires_approval'],
            'size_gb': self.MODEL_CONFIG['size_gb'],
            'vocab_size': self.tokenizer.vocab_size if self.tokenizer else None,
            'parameters': '7.8B',
            'features': self.MODEL_CONFIG['features'],
            'status': 'loaded' if self.model else 'not_loaded',
            'instruction_tuned': True
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
            'recommended_device': cls.MODEL_CONFIG['recommended_device'],
            'open_access': True
        }
        
        # Check available memory (approximate)
        if torch.backends.mps.is_available():
            requirements['device_type'] = 'mps'
        elif torch.cuda.is_available():
            requirements['device_type'] = 'cuda'
        else:
            requirements['device_type'] = 'cpu'
            requirements['warning'] = 'CPU inference will be very slow for 7.8B model'
        
        return requirements
    
    def chat(self, message: str, **kwargs) -> str:
        """
        Conversational interface optimized for instruction-tuned model.
        
        Args:
            message: User message
            **kwargs: Additional generation parameters
            
        Returns:
            Model response
        """
        # Use conversational prompt format
        prompt = f"### Human: {message}\n\n### Assistant:"
        return self.generate(prompt, **kwargs)