import { useState, useEffect } from 'react';
import SearchBar from '../components/SearchBar';
import ResultsList from '../components/ResultsList';
import { searchAudio, listCollections } from '../api/client';

export default function Search() {
  const [results, setResults] = useState(null);
  const [answer, setAnswer] = useState(null);
  const [searchInfo, setSearchInfo] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [collections, setCollections] = useState([]);
  
  useEffect(() => {
    listCollections().then(setCollections).catch(console.error);
  }, []);
  
  const handleSearch = async (query, options) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await searchAudio(query, options);
      
      setResults(response.results || []);
      setAnswer(response.generated_answer);
      setSearchInfo({
        search_type: response.search_type,
        reranked: response.reranked,
        hyde_used: response.hyde_used,
      });
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Search failed');
      setResults(null);
      setAnswer(null);
    } finally {
      setIsLoading(false);
    }
  };
  
  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-gray-900 mb-2">Search Audio</h1>
        <p className="text-gray-600">
          Ask questions about your audio content. Get AI-generated answers with sources.
        </p>
      </div>
      
      <SearchBar 
        onSearch={handleSearch} 
        isLoading={isLoading}
        collections={collections}
      />
      
      {error && (
        <div className="p-4 bg-red-50 text-red-700 rounded-xl">
          {error}
        </div>
      )}
      
      <ResultsList 
        results={results} 
        answer={answer}
        searchInfo={searchInfo}
      />
    </div>
  );
}
