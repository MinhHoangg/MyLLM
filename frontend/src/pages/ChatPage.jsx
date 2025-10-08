import { useState, useEffect } from 'react'
import { useChat } from '../hooks/useChat'
import ChatMessage from '../components/ChatMessage'
import ChatInput from '../components/ChatInput'
import ModelToggle from '../components/ModelToggle'
import LoadingScreen from '../components/LoadingScreen'
import { modelAPI } from '../services/api'
import './ChatPage.css'

const ChatPage = () => {
  const { messages, isLoading, sendMessage, clearHistory, error } = useChat()
  const [modelReady, setModelReady] = useState(false)
  const [modelSize, setModelSize] = useState('2.4B')

  // Check model status on mount and poll until ready
  useEffect(() => {
    let pollInterval

    const checkModelStatus = async () => {
      try {
        const status = await modelAPI.getStatus()
        setModelSize(status.model_size)
        
        if (status.is_loaded) {
          setModelReady(true)
          if (pollInterval) {
            clearInterval(pollInterval)
          }
        }
      } catch (error) {
        console.error('Failed to check model status:', error)
      }
    }

    // Initial check
    checkModelStatus()

    // Poll every 2 seconds until model is ready
    pollInterval = setInterval(checkModelStatus, 2000)

    return () => {
      if (pollInterval) {
        clearInterval(pollInterval)
      }
    }
  }, [])

  const handleSendMessage = async (message, options) => {
    try {
      await sendMessage(message, options)
    } catch (err) {
      console.error('Failed to send message:', err)
    }
  }

  const handleClearHistory = async () => {
    if (window.confirm('Clear conversation history?')) {
      try {
        await clearHistory()
      } catch (err) {
        console.error('Failed to clear history:', err)
      }
    }
  }

  // Show loading screen if model is not ready
  if (!modelReady) {
    return <LoadingScreen modelSize={modelSize} message="Loading model" />
  }

  return (
    <div className="chat-page">
      <div className="chat-header">
        <h2 className="page-title">Chat</h2>
        {messages.length > 0 && (
          <button onClick={handleClearHistory} className="clear-button">
            Clear History
          </button>
        )}
      </div>

      <ModelToggle />

      {error && (
        <div className="error-message">
          Error: {error.message || 'Failed to send message'}
        </div>
      )}

      <div className="chat-container">
        <div className="messages-container">
          {messages.length === 0 ? (
            <div className="empty-state">
              <h3>Welcome to Multimodal RAG Chatbot</h3>
              <p>Ask questions about your uploaded documents or just chat!</p>
              <ul>
                <li>📄 RAG auto-detection for document queries</li>
                <li>🖼️ Image understanding with CLIP</li>
                <li>💬 Conversation history support</li>
                <li>🚀 Powered by EXAONE-3.5 (2.4B or 7.8B)</li>
              </ul>
            </div>
          ) : (
            messages.map((message, index) => (
              <ChatMessage key={index} message={message} />
            ))
          )}
        </div>

        <ChatInput onSendMessage={handleSendMessage} isLoading={isLoading} />
      </div>
    </div>
  )
}

export default ChatPage
