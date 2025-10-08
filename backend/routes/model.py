"""
API Routes for Model Management
"""
from fastapi import APIRouter, HTTPException
from backend.schemas import ModelInfo, ModelConfig, Settings
from backend.app_state import state
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/model", tags=["model"])

@router.get("/status")
async def get_model_status():
    """Get model loading status - lightweight endpoint for polling"""
    return {
        "is_loaded": state.model is not None,
        "high_parameter": state.high_parameter,
        "model_size": "7.8B" if state.high_parameter else "2.4B"
    }

@router.get("/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the current model"""
    try:
        if state.model is None:
            # Model not loaded yet
            return ModelInfo(
                model_name="Loading...",
                model_type="DNotitia",
                device="N/A",
                is_loaded=False,
                high_parameter=state.high_parameter,
                max_tokens=4096,
                parameters="7.8B" if state.high_parameter else "2.4B"
            )
        
        # Get model info
        model_info = {
            "model_name": state.model.model_name if hasattr(state.model, 'model_name') else "DNotitia",
            "model_type": type(state.model).__name__,
            "device": str(state.model.device) if hasattr(state.model, 'device') else "unknown",
            "is_loaded": True,
            "high_parameter": state.high_parameter,
            "max_tokens": 4096,
            "parameters": "7.8B" if state.high_parameter else "2.4B"
        }
        
        return ModelInfo(**model_info)
        
    except Exception as e:
        logger.error(f"Get model info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/config")
async def update_model_config(config: ModelConfig):
    """
    Update model configuration.
    Note: Changing high_parameter requires reloading the model.
    """
    try:
        if config.high_parameter != state.high_parameter:
            logger.info(f"Switching high_parameter: {state.high_parameter} -> {config.high_parameter}")
            
            # Update state
            state.high_parameter = config.high_parameter
            
            # Reload model if already loaded
            if state.model is not None:
                logger.info("Reloading model with new configuration...")
                from models.dnotitia_model import DNotitiaModel
                state.model = DNotitiaModel(high_parameter=config.high_parameter)
                
                # Update chatbot's model reference
                if state.chatbot:
                    state.chatbot.model = state.model
                
                return {
                    "status": "success",
                    "message": "Model reloaded with new configuration",
                    "high_parameter": config.high_parameter,
                    "model_parameters": "7.8B" if config.high_parameter else "2.4B"
                }
            else:
                return {
                    "status": "success",
                    "message": "Configuration updated (model will load with new settings on first request)",
                    "high_parameter": config.high_parameter
                }
        else:
            return {
                "status": "no_change",
                "message": "Configuration already matches current settings"
            }
        
    except Exception as e:
        logger.error(f"Update model config error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/settings", response_model=Settings)
async def get_settings():
    """Get current application settings"""
    try:
        # Default settings
        settings = Settings(
            use_rag=None,  # Auto-detection
            temperature=0.7,
            max_tokens=4096,
            include_history=True,
            similarity_threshold=0.3
        )
        
        return settings
        
    except Exception as e:
        logger.error(f"Get settings error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
