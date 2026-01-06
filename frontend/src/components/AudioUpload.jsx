import { useState, useCallback } from 'react';
import { Upload, FileAudio, X, CheckCircle, Loader2, AlertCircle } from 'lucide-react';

export default function AudioUpload({ onUpload, collections = [] }) {
  const [file, setFile] = useState(null);
  const [collection, setCollection] = useState('audio_rag_contextual');
  const [dragActive, setDragActive] = useState(false);
  const [status, setStatus] = useState('idle'); // idle, uploading, success, error
  const [progress, setProgress] = useState(null);
  
  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);
  
  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.type.startsWith('audio/') || droppedFile.name.match(/\.(mp3|wav|m4a|flac|ogg)$/i)) {
        setFile(droppedFile);
        setStatus('idle');
      }
    }
  }, []);
  
  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setStatus('idle');
    }
  };
  
  const handleUpload = async () => {
    if (!file) return;
    
    setStatus('uploading');
    setProgress('Uploading...');
    
    try {
      const result = await onUpload(file, collection);
      setStatus('success');
      setProgress(`Job ID: ${result.job_id}`);
    } catch (error) {
      setStatus('error');
      setProgress(error.message || 'Upload failed');
    }
  };
  
  const clearFile = () => {
    setFile(null);
    setStatus('idle');
    setProgress(null);
  };
  
  return (
    <div className="space-y-4">
      {/* Drop zone */}
      <div
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        className={`relative border-2 border-dashed rounded-xl p-8 text-center transition-colors ${
          dragActive
            ? 'border-blue-500 bg-blue-50'
            : file
            ? 'border-green-300 bg-green-50'
            : 'border-gray-300 hover:border-gray-400'
        }`}
      >
        {file ? (
          <div className="flex items-center justify-center gap-4">
            <FileAudio className="h-12 w-12 text-green-600" />
            <div className="text-left">
              <p className="font-medium text-gray-900">{file.name}</p>
              <p className="text-sm text-gray-500">
                {(file.size / (1024 * 1024)).toFixed(2)} MB
              </p>
            </div>
            <button
              onClick={clearFile}
              className="p-2 hover:bg-gray-200 rounded-full"
            >
              <X className="h-5 w-5 text-gray-500" />
            </button>
          </div>
        ) : (
          <>
            <Upload className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <p className="text-lg font-medium text-gray-700">
              Drop audio file here
            </p>
            <p className="text-sm text-gray-500 mt-1">
              or click to browse
            </p>
            <input
              type="file"
              accept="audio/*,.mp3,.wav,.m4a,.flac,.ogg"
              onChange={handleFileSelect}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            />
          </>
        )}
      </div>
      
      {/* Options */}
      {file && status === 'idle' && (
        <div className="flex items-center gap-4">
          <div className="flex-1">
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Collection
            </label>
            <select
              value={collection}
              onChange={(e) => setCollection(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg"
            >
              {collections.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
              <option value="new">+ Create new collection</option>
            </select>
          </div>
          
          <button
            onClick={handleUpload}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 mt-6"
          >
            Start Ingestion
          </button>
        </div>
      )}
      
      {/* Status */}
      {status !== 'idle' && (
        <div className={`flex items-center gap-3 p-4 rounded-lg ${
          status === 'uploading' ? 'bg-blue-50 text-blue-700' :
          status === 'success' ? 'bg-green-50 text-green-700' :
          'bg-red-50 text-red-700'
        }`}>
          {status === 'uploading' && <Loader2 className="h-5 w-5 animate-spin" />}
          {status === 'success' && <CheckCircle className="h-5 w-5" />}
          {status === 'error' && <AlertCircle className="h-5 w-5" />}
          <span>{progress}</span>
        </div>
      )}
    </div>
  );
}
