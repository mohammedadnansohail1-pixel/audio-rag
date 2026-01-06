import { Link } from 'react-router-dom';
import { Search, Upload, Mic, Database, Zap, Shield } from 'lucide-react';

const features = [
  {
    icon: Search,
    title: 'Semantic Search',
    description: 'Find content by meaning, not just keywords. Hybrid search combines dense + sparse vectors.',
  },
  {
    icon: Mic,
    title: 'Real-time Transcription',
    description: 'Stream audio from your microphone and get instant transcriptions.',
  },
  {
    icon: Zap,
    title: 'AI-Generated Answers',
    description: 'Get synthesized answers from your audio content using local LLMs.',
  },
  {
    icon: Shield,
    title: 'Privacy First',
    description: 'All processing happens locally. Your data never leaves your servers.',
  },
];

const stats = [
  { label: 'Query Latency', value: '141ms' },
  { label: 'Precision', value: '+47%' },
  { label: 'Real-time Factor', value: '0.66x' },
  { label: 'Languages', value: '100+' },
];

export default function Home() {
  return (
    <div className="space-y-12">
      {/* Hero */}
      <div className="text-center py-12">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          Search Your Audio Content
        </h1>
        <p className="text-xl text-gray-600 max-w-2xl mx-auto mb-8">
          Ingest lectures, meetings, podcasts. Search semantically. Get AI-generated answers with speaker attribution.
        </p>
        <div className="flex items-center justify-center gap-4">
          <Link
            to="/search"
            className="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-xl font-medium hover:bg-blue-700 transition-colors"
          >
            <Search className="h-5 w-5" />
            Start Searching
          </Link>
          <Link
            to="/upload"
            className="flex items-center gap-2 px-6 py-3 bg-white text-gray-700 rounded-xl font-medium border border-gray-300 hover:bg-gray-50 transition-colors"
          >
            <Upload className="h-5 w-5" />
            Upload Audio
          </Link>
        </div>
      </div>
      
      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {stats.map((stat) => (
          <div key={stat.label} className="bg-white rounded-xl p-6 text-center border border-gray-200">
            <div className="text-3xl font-bold text-blue-600">{stat.value}</div>
            <div className="text-sm text-gray-500 mt-1">{stat.label}</div>
          </div>
        ))}
      </div>
      
      {/* Features */}
      <div className="grid md:grid-cols-2 gap-6">
        {features.map((feature) => (
          <div key={feature.title} className="bg-white rounded-xl p-6 border border-gray-200">
            <feature.icon className="h-10 w-10 text-blue-600 mb-4" />
            <h3 className="text-lg font-semibold text-gray-900 mb-2">{feature.title}</h3>
            <p className="text-gray-600">{feature.description}</p>
          </div>
        ))}
      </div>
      
      {/* Tech Stack */}
      <div className="bg-gradient-to-r from-gray-900 to-gray-800 rounded-xl p-8 text-white">
        <h2 className="text-2xl font-bold mb-6 text-center">Powered By</h2>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-6 text-center">
          {['Whisper', 'NeMo', 'BGE-M3', 'Qdrant', 'Ollama'].map((tech) => (
            <div key={tech} className="flex flex-col items-center">
              <Database className="h-8 w-8 mb-2 text-blue-400" />
              <span className="font-medium">{tech}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
