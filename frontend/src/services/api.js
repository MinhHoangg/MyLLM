import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Chat API
export const chatAPI = {
  sendMessage: async (question, useRag = null, includeHistory = true, similarityThreshold = 0.3) => {
    const response = await api.post('/api/chat/', {
      question,
      use_rag: useRag,
      include_history: includeHistory,
      similarity_threshold: similarityThreshold,
    })
    return response.data
  },

  clearHistory: async () => {
    const response = await api.post('/api/chat/clear-history')
    return response.data
  },
}

// Documents API
export const documentsAPI = {
  upload: async (files, useAdaptiveChunking = true) => {
    const formData = new FormData()
    files.forEach((file) => {
      formData.append('files', file)
    })
    formData.append('use_adaptive_chunking', useAdaptiveChunking)

    const response = await api.post('/api/documents/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  },

  list: async (limit = null, contentType = null) => {
    const params = {}
    if (limit) params.limit = limit
    if (contentType) params.content_type = contentType
    
    const response = await api.get('/api/documents/', { params })
    return response.data
  },

  delete: async (documentId) => {
    const response = await api.delete(`/api/documents/${documentId}`)
    return response.data
  },

  clearAll: async () => {
    const response = await api.delete('/api/documents/')
    return response.data
  },

  getStats: async () => {
    const response = await api.get('/api/documents/stats')
    return response.data
  },
}

// Search API
export const searchAPI = {
  search: async (query, contentType = null, similarityThreshold = 0.3) => {
    const response = await api.post('/api/search/', {
      query,
      content_type: contentType,
      similarity_threshold: similarityThreshold,
    })
    return response.data
  },
}

// Model API
export const modelAPI = {
  getStatus: async () => {
    const response = await api.get('/api/model/status')
    return response.data
  },

  getInfo: async () => {
    const response = await api.get('/api/model/info')
    return response.data
  },

  updateConfig: async (highParameter) => {
    const response = await api.post('/api/model/config', {
      high_parameter: highParameter,
    })
    return response.data
  },

  getSettings: async () => {
    const response = await api.get('/api/model/settings')
    return response.data
  },
}

// Health check
export const healthAPI = {
  check: async () => {
    const response = await api.get('/api/health')
    return response.data
  },
}

export default api
