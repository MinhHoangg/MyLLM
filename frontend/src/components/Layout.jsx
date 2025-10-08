import { Link, useLocation } from 'react-router-dom'
import { useApp } from '../context/AppContext'
import { MessageSquare, FileText, Search, Settings } from 'lucide-react'
import './Layout.css'

const Layout = ({ children }) => {
  const location = useLocation()
  const { modelInfo } = useApp()

  const navItems = [
    { path: '/', label: 'Chat', icon: MessageSquare },
    { path: '/documents', label: 'Documents', icon: FileText },
    { path: '/search', label: 'Search', icon: Search },
  ]

  return (
    <div className="layout">
      <nav className="navbar">
        <div className="navbar-content">
          <div className="navbar-brand">
            <MessageSquare size={24} />
            <h1>Multimodal RAG Chatbot</h1>
          </div>

          <div className="navbar-links">
            {navItems.map((item) => {
              const Icon = item.icon
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`nav-link ${location.pathname === item.path ? 'active' : ''}`}
                >
                  <Icon size={18} />
                  <span>{item.label}</span>
                </Link>
              )
            })}
          </div>

          <div className="navbar-info">
            {modelInfo && (
              <div className="model-info">
                <Settings size={16} />
                <span>{modelInfo.parameters} Model</span>
                {!modelInfo.is_loaded && <span className="badge">Not Loaded</span>}
              </div>
            )}
          </div>
        </div>
      </nav>

      <main className="main-content">{children}</main>

      <footer className="footer">
        <p>Powered by EXAONE-3.5 • CLIP • FastAPI • React</p>
      </footer>
    </div>
  )
}

export default Layout
