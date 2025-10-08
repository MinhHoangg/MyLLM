import ReactMarkdown from 'react-markdown'
import { User, Bot, FileText } from 'lucide-react'
import './ChatMessage.css'

const ChatMessage = ({ message }) => {
  const isUser = message.role === 'user'

  return (
    <div className={`message ${isUser ? 'message-user' : 'message-assistant'}`}>
      <div className="message-icon">
        {isUser ? <User size={20} /> : <Bot size={20} />}
      </div>

      <div className="message-content">
        <div className="message-text">
          <ReactMarkdown>{message.content}</ReactMarkdown>
        </div>

        {!isUser && message.sources && message.sources.length > 0 && (
          <div className="message-sources">
            <div className="sources-header">
              <FileText size={14} />
              <span>Sources ({message.sources.length})</span>
            </div>
            <div className="sources-list">
              {message.sources.map((source, idx) => (
                <div key={idx} className="source-item">
                  <span className="source-name">{source.file_name || 'Document'}</span>
                  {source.similarity && (
                    <span className="source-similarity">
                      {(source.similarity * 100).toFixed(1)}%
                    </span>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default ChatMessage
