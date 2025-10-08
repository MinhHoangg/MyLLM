"""
API Routes for Chat functionality
"""
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from backend.schemas import ChatRequest, ChatResponse
from backend.app_state import state
import logging
import json

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])

@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a chat message and get a response.
    Supports RAG auto-detection and conversation history.
    """
    try:
        # Check if model is loaded
        if state.model is None:
            raise HTTPException(status_code=503, detail="Model is still loading. Please wait.")
        
        if not state.chatbot:
            raise HTTPException(status_code=500, detail="Chatbot not initialized")
        
        # Process chat request
        response = state.chatbot.chat(
            question=request.question,
            use_rag=request.use_rag,
            include_history=request.include_history,
            similarity_threshold=request.similarity_threshold
        )
        
        return ChatResponse(**response)
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clear-history")
async def clear_history():
    """Clear conversation history"""
    try:
        if state.chatbot:
            state.chatbot.clear_history()
            return {"status": "success", "message": "History cleared"}
        raise HTTPException(status_code=500, detail="Chatbot not initialized")
    except Exception as e:
        logger.error(f"Clear history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/ws")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for streaming chat responses.
    Send JSON: {"question": "...", "use_rag": null}
    Receive chunks of the response as they're generated.
    """
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            request_data = json.loads(data)
            
            question = request_data.get("question")
            use_rag = request_data.get("use_rag")
            
            if not question:
                await websocket.send_json({"error": "Question is required"})
                continue
            
            # Check if model is loaded
            if state.model is None:
                await websocket.send_json({
                    "error": "Model is still loading. Please wait."
                })
                continue
            
            # Send status
            await websocket.send_json({"status": "processing"})
            
            # Get response
            response = state.chatbot.chat(
                question=question,
                use_rag=use_rag,
                include_history=request_data.get("include_history", True)
            )
            
            # Send complete response
            await websocket.send_json({
                "status": "complete",
                "response": response
            })
            
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass
