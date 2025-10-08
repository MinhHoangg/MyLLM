import { createContext, useContext, useState, useEffect } from 'react'
import { modelAPI } from '../services/api'

const AppContext = createContext()

export const useApp = () => {
  const context = useContext(AppContext)
  if (!context) {
    throw new Error('useApp must be used within AppProvider')
  }
  return context
}

export const AppProvider = ({ children }) => {
  const [modelInfo, setModelInfo] = useState(null)
  const [settings, setSettings] = useState({
    useRag: null,
    temperature: 0.7,
    maxTokens: 4096,
    includeHistory: true,
    similarityThreshold: 0.3,
  })
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadModelInfo()
    loadSettings()
  }, [])

  const loadModelInfo = async () => {
    try {
      const info = await modelAPI.getInfo()
      setModelInfo(info)
    } catch (error) {
      console.error('Failed to load model info:', error)
    }
  }

  const loadSettings = async () => {
    try {
      const s = await modelAPI.getSettings()
      setSettings(s)
    } catch (error) {
      console.error('Failed to load settings:', error)
    } finally {
      setLoading(false)
    }
  }

  const updateModelConfig = async (highParameter) => {
    try {
      await modelAPI.updateConfig(highParameter)
      await loadModelInfo()
    } catch (error) {
      console.error('Failed to update model config:', error)
      throw error
    }
  }

  const value = {
    modelInfo,
    settings,
    loading,
    updateModelConfig,
    loadModelInfo,
  }

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>
}
