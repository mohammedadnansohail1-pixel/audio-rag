import StreamingMic from '../components/StreamingMic';
import { AlertCircle } from 'lucide-react';

export default function Streaming() {
  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-gray-900 mb-2">Live Transcription</h1>
        <p className="text-gray-600">
          Stream audio from your microphone for real-time transcription.
        </p>
      </div>
      
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <StreamingMic />
      </div>
      
      {/* Info */}
      <div className="bg-amber-50 rounded-xl p-6 border border-amber-200">
        <div className="flex items-start gap-3">
          <AlertCircle className="h-5 w-5 text-amber-600 mt-0.5" />
          <div>
            <h3 className="font-semibold text-amber-900 mb-1">Browser Requirements</h3>
            <ul className="text-sm text-amber-800 space-y-1">
              <li>• Microphone access permission required</li>
              <li>• Works best in Chrome or Firefox</li>
              <li>• Ensure API server is running on localhost:8000</li>
              <li>• Processing delay: ~5-7 seconds per chunk</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
