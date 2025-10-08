import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { searchAPI } from '../services/api'

export const useSearch = () => {
  const [results, setResults] = useState([])

  const searchMutation = useMutation({
    mutationFn: ({ query, contentType, similarityThreshold }) =>
      searchAPI.search(query, contentType, similarityThreshold),
    onSuccess: (data) => {
      setResults(data.documents || [])
    },
  })

  const search = async (query, options = {}) => {
    return await searchMutation.mutateAsync({
      query,
      contentType: options.contentType ?? null,
      similarityThreshold: options.similarityThreshold ?? 0.3,
    })
  }

  const clearResults = () => {
    setResults([])
  }

  return {
    results,
    search,
    clearResults,
    isSearching: searchMutation.isPending,
    error: searchMutation.error,
  }
}
