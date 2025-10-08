import { useState, useEffect } from 'react'
import { useApp } from '../context/AppContext'
import LoadingScreen from './LoadingScreen'
import { modelAPI } from '../services/api'
import './ModelToggle.css'

const ModelToggle = () => {
  const { modelInfo, updateModelConfig, loadModelInfo } = useApp()
  const [localHighParam, setLocalHighParam] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [hasChanges, setHasChanges] = useState(false)
  const [showFullScreenLoading, setShowFullScreenLoading] = useState(false)

  // Initialize from modelInfo and localStorage
  useEffect(() => {
    if (modelInfo) {
      setLocalHighParam(modelInfo.high_parameter)
      // Save to localStorage
      localStorage.setItem('high_parameter', modelInfo.high_parameter.toString())
    } else {
      // Load from localStorage on first render
      const saved = localStorage.getItem('high_parameter')
      if (saved !== null) {
        setLocalHighParam(saved === 'true')
      }
    }
  }, [modelInfo])

  // Check if there are unsaved changes
  useEffect(() => {
    if (modelInfo) {
      setHasChanges(localHighParam !== modelInfo.high_parameter)
    }
  }, [localHighParam, modelInfo])

  const handleToggle = () => {
    setLocalHighParam(!localHighParam)
  }

  const handleApply = async () => {
    setIsLoading(true)
    setShowFullScreenLoading(true)
    
    try {
      await updateModelConfig(localHighParam)
      // Save to localStorage
      localStorage.setItem('high_parameter', localHighParam.toString())
      
      // Poll for model status until ready
      let attempts = 0
      const maxAttempts = 60 // 2 minutes max
      
      while (attempts < maxAttempts) {
        try {
          const status = await modelAPI.getStatus()
          if (status.is_loaded) {
            break
          }
        } catch (error) {
          console.error('Status check failed:', error)
        }
        
        // Wait 2 seconds before next check
        await new Promise(resolve => setTimeout(resolve, 2000))
        attempts++
      }
      
      // Reload model info to get updated status
      await loadModelInfo()
      setHasChanges(false)
    } catch (error) {
      console.error('Failed to update model config:', error)
      alert('Failed to update model configuration. Please try again.')
    } finally {
      setIsLoading(false)
      setShowFullScreenLoading(false)
    }
  }

  const getModelName = () => {
    if (modelInfo && modelInfo.model_name && modelInfo.model_name !== 'Not loaded (lazy loading)') {
      return modelInfo.model_name
    }
    return localHighParam ? 'EXAONE-3.5-7.8B' : 'EXAONE-4.0-1.2B'
  }

  return (
    <div className="model-toggle-container">
      <div className="model-info-display">
        <span className="model-icon">🤖</span>
        <span className="model-name">{getModelName()}</span>
        {modelInfo && !modelInfo.is_loaded && (
          <span className="badge badge-warning">Not Loaded</span>
        )}
      </div>

      <div className="model-toggle-controls">
        <label className="toggle-label">
          <span className="toggle-text">⚡ High Performance Mode</span>
          <div className="toggle-switch">
            <input
              type="checkbox"
              checked={localHighParam}
              onChange={handleToggle}
              disabled={isLoading}
            />
            <span className="slider"></span>
          </div>
          <span className="toggle-value">
            {localHighParam ? '7.8B' : '2.4B'}
          </span>
        </label>

        {hasChanges && (
          <button
            className="apply-button"
            onClick={handleApply}
            disabled={isLoading}
          >
            {isLoading ? (
              <>
                <span className="spinner"></span>
                Loading Model...
              </>
            ) : (
              <>Apply & Reload Model</>
            )}
          </button>
        )}
      </div>

      {showFullScreenLoading && (
        <LoadingScreen 
          modelSize={localHighParam ? '7.8B' : '2.4B'} 
          message="Switching model"
        />
      )}
    </div>
  )
}

export default ModelToggle
