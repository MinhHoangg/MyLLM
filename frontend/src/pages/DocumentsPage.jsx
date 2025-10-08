import { useState } from 'react'
import { useDocuments } from '../hooks/useDocuments'
import DocumentUpload from '../components/DocumentUpload'
import DocumentList from '../components/DocumentList'
import './DocumentsPage.css'

const DocumentsPage = () => {
  const {
    documents,
    stats,
    isLoading,
    uploadDocuments,
    deleteDocument,
    clearAll,
    isUploading,
  } = useDocuments()

  const [uploadStatus, setUploadStatus] = useState(null)

  const handleUpload = async (files) => {
    try {
      setUploadStatus(null)
      const result = await uploadDocuments({ files, useAdaptiveChunking: true })
      setUploadStatus({
        type: 'success',
        message: `Successfully uploaded ${result.files_processed} files (${result.chunks} chunks, ${result.images} images)`,
      })
    } catch (error) {
      setUploadStatus({
        type: 'error',
        message: `Upload failed: ${error.message}`,
      })
    }
  }

  const handleDelete = async (documentId) => {
    if (window.confirm('Delete this document?')) {
      try {
        await deleteDocument(documentId)
      } catch (error) {
        console.error('Failed to delete document:', error)
      }
    }
  }

  const handleClearAll = async () => {
    if (window.confirm('Delete ALL documents? This cannot be undone!')) {
      try {
        await clearAll()
      } catch (error) {
        console.error('Failed to clear documents:', error)
      }
    }
  }

  return (
    <div className="documents-page">
      <div className="page-header">
        <h2 className="page-title">Documents</h2>
        {documents.length > 0 && (
          <button onClick={handleClearAll} className="danger-button">
            Clear All
          </button>
        )}
      </div>

      {uploadStatus && (
        <div className={`status-message ${uploadStatus.type}`}>
          {uploadStatus.message}
        </div>
      )}

      {stats && (
        <div className="stats-panel">
          <div className="stat-item">
            <span className="stat-label">Total Documents</span>
            <span className="stat-value">{stats.total_documents}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Text Documents</span>
            <span className="stat-value">{stats.text_documents}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Images</span>
            <span className="stat-value">{stats.image_documents}</span>
          </div>
        </div>
      )}

      <DocumentUpload onUpload={handleUpload} isUploading={isUploading} />

      <DocumentList
        documents={documents}
        isLoading={isLoading}
        onDelete={handleDelete}
      />
    </div>
  )
}

export default DocumentsPage
