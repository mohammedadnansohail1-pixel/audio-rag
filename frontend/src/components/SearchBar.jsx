import { useState } from 'react';
import { Search, Settings, Loader2 } from 'lucide-react';

export default function SearchBar({ onSearch, isLoading, collections = [] }) {
  const [query, setQuery] = useState('');
  const [showOptions, setShowOptions] = useState(false);
  const [options, setOptions] = useState({
    collection: 'audio_rag_contextual',
    searchType: 'hybrid',
    generateAnswer: true,
    reranking: true,
  });
  
  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim()) {
      onSearch(query, options);
    }
  };
  
  return (
    <form onSubmit={handleSubmit} className="w-full">
      <div className="relative">
        <div className="flex items-center gap-2">
          <div className="relative flex-1">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-400" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask a question about your audio content..."
              className="w-full pl-12 pr-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent text-lg"
              disabled={isLoading}
            />
          </div>
          
          <button
            type="button"
            onClick={() => setShowOptions(!showOptions)}
            className={`p-3 rounded-xl border transition-colors ${
              showOptions ? 'bg-blue-100 border-blue-300' : 'border-gray-300 hover:bg-gray-50'
            }`}
          >
            <Settings className="h-5 w-5 text-gray-600" />
          </button>
          
          <button
            type="submit"
            disabled={isLoading || !query.trim()}
            className="px-6 py-3 bg-blue-600 text-white rounded-xl font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {isLoading ? (
              <>
                <Loader2 className="h-5 w-5 animate-spin" />
                Searching...
              </>
            ) : (
              'Search'
            )}
          </button>
        </div>
        
        {/* Options panel */}
        {showOptions && (
          <div className="absolute top-full left-0 right-0 mt-2 p-4 bg-white rounded-xl border border-gray-200 shadow-lg z-10">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Collection</label>
                <select
                  value={options.collection}
                  onChange={(e) => setOptions({ ...options, collection: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                >
                  {collections.map((c) => (
                    <option key={c} value={c}>{c}</option>
                  ))}
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Search Type</label>
                <select
                  value={options.searchType}
                  onChange={(e) => setOptions({ ...options, searchType: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                >
                  <option value="hybrid">Hybrid (Best)</option>
                  <option value="dense">Dense Only</option>
                  <option value="sparse">Sparse Only</option>
                </select>
              </div>
              
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  id="generateAnswer"
                  checked={options.generateAnswer}
                  onChange={(e) => setOptions({ ...options, generateAnswer: e.target.checked })}
                  className="h-4 w-4 text-blue-600 rounded"
                />
                <label htmlFor="generateAnswer" className="text-sm text-gray-700">Generate Answer</label>
              </div>
              
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  id="reranking"
                  checked={options.reranking}
                  onChange={(e) => setOptions({ ...options, reranking: e.target.checked })}
                  className="h-4 w-4 text-blue-600 rounded"
                />
                <label htmlFor="reranking" className="text-sm text-gray-700">Reranking</label>
              </div>
            </div>
          </div>
        )}
      </div>
    </form>
  );
}
