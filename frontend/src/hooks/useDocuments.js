import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { documentsAPI } from '../services/api'

export const useDocuments = () => {
  const queryClient = useQueryClient()

  const { data: documents, isLoading, error } = useQuery({
    queryKey: ['documents'],
    queryFn: () => documentsAPI.list(),
  })

  const { data: stats } = useQuery({
    queryKey: ['documents-stats'],
    queryFn: () => documentsAPI.getStats(),
  })

  const uploadMutation = useMutation({
    mutationFn: ({ files, useAdaptiveChunking }) =>
      documentsAPI.upload(files, useAdaptiveChunking),
    onSuccess: () => {
      queryClient.invalidateQueries(['documents'])
      queryClient.invalidateQueries(['documents-stats'])
    },
  })

  const deleteMutation = useMutation({
    mutationFn: (documentId) => documentsAPI.delete(documentId),
    onSuccess: () => {
      queryClient.invalidateQueries(['documents'])
      queryClient.invalidateQueries(['documents-stats'])
    },
  })

  const clearAllMutation = useMutation({
    mutationFn: () => documentsAPI.clearAll(),
    onSuccess: () => {
      queryClient.invalidateQueries(['documents'])
      queryClient.invalidateQueries(['documents-stats'])
    },
  })

  return {
    documents: documents || [],
    stats,
    isLoading,
    error,
    uploadDocuments: uploadMutation.mutateAsync,
    deleteDocument: deleteMutation.mutateAsync,
    clearAll: clearAllMutation.mutateAsync,
    isUploading: uploadMutation.isPending,
    isDeleting: deleteMutation.isPending,
  }
}
