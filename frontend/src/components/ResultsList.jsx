import { User, Clock, FileAudio, ChevronDown, ChevronUp } from 'lucide-react';
import { useState } from 'react';

function formatTime(seconds) {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

function ResultCard({ result, index }) {
  const [expanded, setExpanded] = useState(false);
  
  return (
    <div className="bg-white rounded-xl border border-gray-200 overflow-hidden hover:shadow-md transition-shadow">
      <div className="p-4">
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1">
            <div className="flex items-center gap-2 text-sm text-gray-500 mb-2">
              <span className="bg-blue-100 text-blue-700 px-2 py-0.5 rounded font-medium">
                #{index + 1}
              </span>
              <span className="flex items-center gap-1">
                <User className="h-3 w-3" />
                {result.speaker || 'Unknown'}
              </span>
              <span className="flex items-center gap-1">
                <Clock className="h-3 w-3" />
                {formatTime(result.start)} - {formatTime(result.end)}
              </span>
              <span className="text-gray-400">
                Score: {(result.score * 100).toFixed(1)}%
              </span>
            </div>
            
            <p className={`text-gray-800 ${expanded ? '' : 'line-clamp-3'}`}>
              {result.text}
            </p>
            
            {result.text.length > 200 && (
              <button
                onClick={() => setExpanded(!expanded)}
                className="mt-2 text-sm text-blue-600 hover:text-blue-700 flex items-center gap-1"
              >
                {expanded ? (
                  <>
                    <ChevronUp className="h-4 w-4" /> Show less
                  </>
                ) : (
                  <>
                    <ChevronDown className="h-4 w-4" /> Show more
                  </>
                )}
              </button>
            )}
          </div>
        </div>
        
        {result.metadata?.source_filename && (
          <div className="mt-3 pt-3 border-t border-gray-100 flex items-center gap-2 text-sm text-gray-500">
            <FileAudio className="h-4 w-4" />
            {result.metadata.source_filename}
          </div>
        )}
      </div>
    </div>
  );
}

export default function ResultsList({ results, answer, searchInfo }) {
  if (!results) return null;
  
  return (
    <div className="space-y-6">
      {/* Generated Answer */}
      {answer && (
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl p-6 border border-blue-200">
          <h3 className="text-lg font-semibold text-gray-900 mb-3 flex items-center gap-2">
            <span className="bg-blue-600 text-white px-2 py-1 rounded text-sm">AI Answer</span>
          </h3>
          <p className="text-gray-800 whitespace-pre-wrap">{answer}</p>
        </div>
      )}
      
      {/* Search Info */}
      {searchInfo && (
        <div className="flex items-center gap-4 text-sm text-gray-500">
          <span>{results.length} results</span>
          <span>•</span>
          <span>Search: {searchInfo.search_type}</span>
          {searchInfo.reranked && <span className="text-green-600">✓ Reranked</span>}
          {searchInfo.hyde_used && <span className="text-purple-600">✓ HyDE</span>}
        </div>
      )}
      
      {/* Results */}
      <div className="space-y-4">
        {results.map((result, index) => (
          <ResultCard key={index} result={result} index={index} />
        ))}
      </div>
      
      {results.length === 0 && (
        <div className="text-center py-12 text-gray-500">
          No results found. Try a different query.
        </div>
      )}
    </div>
  );
}
