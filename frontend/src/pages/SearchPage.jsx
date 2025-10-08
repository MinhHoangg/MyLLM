import { useState } from 'react'
import { useSearch } from '../hooks/useSearch'
import { Search } from 'lucide-react'
import DocumentList from '../components/DocumentList'
import './SearchPage.css'

const SearchPage = () => {
  const { results, search, clearResults, isSearching, error } = useSearch()
  const [query, setQuery] = useState('')
  const [contentType, setContentType] = useState('')
  const [similarityThreshold, setSimilarityThreshold] = useState(0.3)

  const handleSearch = async (e) => {
    e.preventDefault()
    if (query.trim()) {
      try {
        await search(query, {
          contentType: contentType || null,
          similarityThreshold,
        })
      } catch (err) {
        console.error('Search failed:', err)
      }
    }
  }

  const handleClear = () => {
    setQuery('')
    clearResults()
  }

  return (
    <div className="search-page">
      <h2 className="page-title">Search Documents</h2>

      <div className="search-container">
        <form onSubmit={handleSearch} className="search-form">
          <div className="search-input-group">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search your documents..."
              className="search-input"
            />
            <button
              type="submit"
              className="search-button"
              disabled={!query.trim() || isSearching}
            >
              <Search size={20} />
              {isSearching ? 'Searching...' : 'Search'}
            </button>
          </div>

          <div className="search-filters">
            <div className="filter-group">
              <label>Content Type</label>
              <select
                value={contentType}
                onChange={(e) => setContentType(e.target.value)}
                className="filter-select"
              >
                <option value="">All</option>
                <option value="text">Text Only</option>
                <option value="image">Images Only</option>
              </select>
            </div>

            <div className="filter-group">
              <label>Similarity Threshold: {(similarityThreshold * 100).toFixed(0)}%</label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={similarityThreshold}
                onChange={(e) => setSimilarityThreshold(parseFloat(e.target.value))}
                className="filter-slider"
              />
            </div>

            {results.length > 0 && (
              <button type="button" onClick={handleClear} className="clear-button">
                Clear Results
              </button>
            )}
          </div>
        </form>

        {error && (
          <div className="error-message">
            Search failed: {error.message}
          </div>
        )}

        {results.length > 0 && (
          <div className="search-results">
            <div className="results-header">
              <h3>Search Results</h3>
              <span className="results-count">{results.length} documents found</span>
            </div>
            <DocumentList documents={results} isLoading={false} onDelete={() => {}} />
          </div>
        )}

        {!isSearching && query && results.length === 0 && (
          <div className="empty-state">
            <p>No documents found matching your search.</p>
            <p className="hint">Try adjusting your query or lowering the similarity threshold.</p>
          </div>
        )}
      </div>
    </div>
  )
}

export default SearchPage
