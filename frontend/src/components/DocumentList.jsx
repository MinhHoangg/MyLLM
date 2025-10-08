import { FileText, Image, Trash2 } from 'lucide-react'
import './DocumentList.css'

const DocumentList = ({ documents, isLoading, onDelete }) => {
  if (isLoading) {
    return (
      <div className="document-list-container">
        <div className="loading">Loading documents</div>
      </div>
    )
  }

  if (documents.length === 0) {
    return (
      <div className="document-list-container">
        <div className="empty-state">
          <p>No documents uploaded yet</p>
        </div>
      </div>
    )
  }

  return (
    <div className="document-list-container">
      <h3 className="list-title">Uploaded Documents ({documents.length})</h3>
      <div className="document-grid">
        {documents.map((doc) => {
          const isImage = doc.metadata?.content_type === 'image'
          const fileName =
            doc.metadata?.original_file_name || doc.metadata?.file_name || 'Unknown'
          const chunkIndex = doc.metadata?.chunk_index

          return (
            <div key={doc.id} className="document-card">
              <div className="document-header">
                <div className="document-icon">
                  {isImage ? (
                    <Image size={20} />
                  ) : (
                    <FileText size={20} />
                  )}
                </div>
                <button
                  onClick={() => onDelete(doc.id)}
                  className="delete-button"
                  title="Delete document"
                >
                  <Trash2 size={16} />
                </button>
              </div>

              <div className="document-body">
                <div className="document-name">{fileName}</div>
                {chunkIndex !== undefined && (
                  <div className="document-chunk">Chunk {chunkIndex}</div>
                )}
                <div className="document-preview">
                  {doc.content.substring(0, 150)}
                  {doc.content.length > 150 && '...'}
                </div>
              </div>

              {doc.similarity && (
                <div className="document-similarity">
                  Similarity: {(doc.similarity * 100).toFixed(1)}%
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default DocumentList
