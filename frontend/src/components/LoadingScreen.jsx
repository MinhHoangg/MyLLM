import { useEffect, useState } from 'react'
import './LoadingScreen.css'

const LoadingScreen = ({ modelSize = '2.4B', message = 'Loading model' }) => {
  const [dots, setDots] = useState('')
  const [progress, setProgress] = useState(0)

  useEffect(() => {
    // Animated dots
    const dotsInterval = setInterval(() => {
      setDots((prev) => (prev.length >= 3 ? '' : prev + '.'))
    }, 500)

    // Simulated progress (just for visual feedback)
    const progressInterval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 90) return prev // Stop at 90%, wait for actual load
        return prev + Math.random() * 15
      })
    }, 1000)

    return () => {
      clearInterval(dotsInterval)
      clearInterval(progressInterval)
    }
  }, [])

  return (
    <div className="loading-screen-overlay">
      <div className="loading-screen-content">
        <div className="loading-spinner-container">
          <div className="loading-spinner">
            <div className="spinner-ring"></div>
            <div className="spinner-ring"></div>
            <div className="spinner-ring"></div>
            <div className="spinner-core">
              <span className="model-badge">{modelSize}</span>
            </div>
          </div>
        </div>

        <h2 className="loading-title">
          {message}
          <span className="loading-dots">{dots}</span>
        </h2>

        <p className="loading-subtitle">
          Loading {modelSize} model parameters
        </p>

        <div className="loading-progress-bar">
          <div 
            className="loading-progress-fill" 
            style={{ width: `${Math.min(progress, 100)}%` }}
          ></div>
        </div>

        <p className="loading-hint">
          This may take 30-60 seconds on first load
        </p>

        <div className="loading-features">
          <div className="feature-item">
            <span className="feature-icon">🧠</span>
            <span className="feature-text">Neural Network</span>
          </div>
          <div className="feature-item">
            <span className="feature-icon">🖼️</span>
            <span className="feature-text">Image Understanding</span>
          </div>
          <div className="feature-item">
            <span className="feature-icon">📄</span>
            <span className="feature-text">Document Analysis</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default LoadingScreen
