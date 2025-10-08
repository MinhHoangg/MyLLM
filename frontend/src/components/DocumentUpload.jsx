import { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload } from 'lucide-react'
import './DocumentUpload.css'

const DocumentUpload = ({ onUpload, isUploading }) => {
  const onDrop = useCallback(
    (acceptedFiles) => {
      if (acceptedFiles.length > 0) {
        onUpload(acceptedFiles)
      }
    },
    [onUpload]
  )

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    disabled: isUploading,
  })

  return (
    <div
      {...getRootProps()}
      className={`dropzone ${isDragActive ? 'dropzone-active' : ''} ${
        isUploading ? 'dropzone-disabled' : ''
      }`}
    >
      <input {...getInputProps()} />
      <Upload size={48} />
      {isDragActive ? (
        <p>Drop files here...</p>
      ) : isUploading ? (
        <p>Uploading...</p>
      ) : (
        <>
          <p>Drag & drop files here, or click to select</p>
          <p className="dropzone-hint">Supports text documents and images</p>
        </>
      )}
    </div>
  )
}

export default DocumentUpload
