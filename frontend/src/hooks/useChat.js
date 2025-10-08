import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { chatAPI } from '../services/api'

export const useChat = () => {
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)

  const sendMessageMutation = useMutation({
    mutationFn: ({ question, useRag, includeHistory, similarityThreshold }) =>
      chatAPI.sendMessage(question, useRag, includeHistory, similarityThreshold),
    onSuccess: (data, variables) => {
      // Add user message
      setMessages((prev) => [
        ...prev,
        {
          role: 'user',
          content: variables.question,
          timestamp: new Date().toISOString(),
        },
      ])

      // Add assistant message
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: data.answer,
          context: data.context,
          sources: data.sources,
          method: data.method,
          usedRag: data.used_rag,
          timestamp: new Date().toISOString(),
        },
      ])
    },
  })

  const sendMessage = async (question, options = {}) => {
    setIsLoading(true)
    try {
      await sendMessageMutation.mutateAsync({
        question,
        useRag: options.useRag ?? null,
        includeHistory: options.includeHistory ?? true,
        similarityThreshold: options.similarityThreshold ?? 0.3,
      })
    } finally {
      setIsLoading(false)
    }
  }

  const clearHistory = async () => {
    try {
      await chatAPI.clearHistory()
      setMessages([])
    } catch (error) {
      console.error('Failed to clear history:', error)
      throw error
    }
  }

  return {
    messages,
    isLoading,
    sendMessage,
    clearHistory,
    error: sendMessageMutation.error,
  }
}
