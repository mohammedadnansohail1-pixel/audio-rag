import { useState, useEffect } from 'react';
import AudioUpload from '../components/AudioUpload';
import { ingestAudio, listCollections } from '../api/client';
import { FileAudio, CheckCircle, Clock } from 'lucide-react';

export default function Upload() {
  const [collections, setCollections] = useState([]);
  const [recentJobs, setRecentJobs] = useState([]);
  
  useEffect(() => {
    listCollections().then(setCollections).catch(console.error);
  }, []);
  
  const handleUpload = async (file, collection) => {
    const result = await ingestAudio(file, collection);
    
    setRecentJobs(prev => [{
      id: result.job_id,
      filename: file.name,
      collection,
      status: 'queued',
      timestamp: new Date().toLocaleTimeString(),
    }, ...prev].slice(0, 5));
    
    return result;
  };
  
  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-gray-900 mb-2">Upload Audio</h1>
        <p className="text-gray-600">
          Upload audio files for transcription and indexing. Supports MP3, WAV, M4A, FLAC.
        </p>
      </div>
      
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <AudioUpload 
          onUpload={handleUpload}
          collections={collections}
        />
      </div>
      
      {/* Recent Jobs */}
      {recentJobs.length > 0 && (
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Recent Uploads</h2>
          <div className="space-y-3">
            {recentJobs.map((job) => (
              <div key={job.id} className="flex items-center gap-4 p-3 bg-gray-50 rounded-lg">
                <FileAudio className="h-5 w-5 text-gray-400" />
                <div className="flex-1">
                  <p className="font-medium text-gray-900">{job.filename}</p>
                  <p className="text-sm text-gray-500">{job.collection}</p>
                </div>
                <div className="flex items-center gap-2 text-sm">
                  {job.status === 'queued' ? (
                    <Clock className="h-4 w-4 text-yellow-500" />
                  ) : (
                    <CheckCircle className="h-4 w-4 text-green-500" />
                  )}
                  <span className="text-gray-500">{job.timestamp}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {/* Info */}
      <div className="bg-blue-50 rounded-xl p-6 border border-blue-200">
        <h3 className="font-semibold text-blue-900 mb-2">Processing Pipeline</h3>
        <ol className="list-decimal list-inside space-y-1 text-blue-800">
          <li>Audio transcription with Whisper large-v3</li>
          <li>Speaker diarization with NeMo</li>
          <li>Word-to-speaker alignment</li>
          <li>Contextual chunk generation</li>
          <li>BGE-M3 embedding (dense + sparse)</li>
          <li>Indexing in Qdrant</li>
        </ol>
      </div>
    </div>
  );
}
