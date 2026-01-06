import { useState, useRef, useEffect } from 'react';
import { Mic, MicOff, Loader2 } from 'lucide-react';
import { getStreamingUrl } from '../api/client';

export default function StreamingMic() {
  const [isRecording, setIsRecording] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [transcripts, setTranscripts] = useState([]);
  const [error, setError] = useState(null);
  
  const wsRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const processorRef = useRef(null);
  const audioContextRef = useRef(null);
  
  const startRecording = async () => {
    setError(null);
    setIsConnecting(true);
    
    try {
      // Get microphone access
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        } 
      });
      mediaStreamRef.current = stream;
      
      // Connect WebSocket
      const wsUrl = getStreamingUrl({ chunkDuration: 5 });
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;
      
      ws.onopen = () => {
        setIsConnecting(false);
        setIsRecording(true);
        startAudioProcessing(stream);
      };
      
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.type === 'transcript' && data.text) {
          setTranscripts(prev => [...prev, {
            text: data.text,
            start: data.start,
            end: data.end,
            isFinal: data.is_final,
            timestamp: new Date().toLocaleTimeString(),
          }]);
        } else if (data.type === 'error') {
          setError(data.message);
        }
      };
      
      ws.onerror = () => {
        setError('WebSocket connection failed');
        stopRecording();
      };
      
      ws.onclose = () => {
        setIsRecording(false);
        setIsConnecting(false);
      };
      
    } catch (err) {
      setError(err.message || 'Failed to access microphone');
      setIsConnecting(false);
    }
  };
  
  const startAudioProcessing = (stream) => {
    const audioContext = new AudioContext({ sampleRate: 16000 });
    audioContextRef.current = audioContext;
    
    const source = audioContext.createMediaStreamSource(stream);
    const processor = audioContext.createScriptProcessor(4096, 1, 1);
    processorRef.current = processor;
    
    processor.onaudioprocess = (e) => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        const inputData = e.inputBuffer.getChannelData(0);
        // Convert float32 to int16
        const int16Data = new Int16Array(inputData.length);
        for (let i = 0; i < inputData.length; i++) {
          int16Data[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768));
        }
        wsRef.current.send(int16Data.buffer);
      }
    };
    
    source.connect(processor);
    processor.connect(audioContext.destination);
  };
  
  const stopRecording = () => {
    // Send stop command
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ command: 'stop' }));
    }
    
    // Cleanup audio
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }
    
    // Close WebSocket
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    
    setIsRecording(false);
  };
  
  const clearTranscripts = () => {
    setTranscripts([]);
  };
  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopRecording();
    };
  }, []);
  
  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="flex items-center justify-center gap-4">
        <button
          onClick={isRecording ? stopRecording : startRecording}
          disabled={isConnecting}
          className={`flex items-center gap-3 px-8 py-4 rounded-full font-medium text-lg transition-all ${
            isRecording
              ? 'bg-red-600 hover:bg-red-700 text-white'
              : 'bg-blue-600 hover:bg-blue-700 text-white'
          } disabled:opacity-50`}
        >
          {isConnecting ? (
            <>
              <Loader2 className="h-6 w-6 animate-spin" />
              Connecting...
            </>
          ) : isRecording ? (
            <>
              <MicOff className="h-6 w-6" />
              Stop Recording
            </>
          ) : (
            <>
              <Mic className="h-6 w-6" />
              Start Recording
            </>
          )}
        </button>
        
        {transcripts.length > 0 && (
          <button
            onClick={clearTranscripts}
            className="px-4 py-2 text-gray-600 hover:text-gray-800"
          >
            Clear
          </button>
        )}
      </div>
      
      {/* Recording indicator */}
      {isRecording && (
        <div className="flex items-center justify-center gap-2 text-red-600">
          <span className="relative flex h-3 w-3">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-3 w-3 bg-red-500"></span>
          </span>
          Recording... Speak into your microphone
        </div>
      )}
      
      {/* Error */}
      {error && (
        <div className="p-4 bg-red-50 text-red-700 rounded-lg text-center">
          {error}
        </div>
      )}
      
      {/* Transcripts */}
      <div className="bg-white rounded-xl border border-gray-200 min-h-[300px] max-h-[500px] overflow-y-auto">
        {transcripts.length === 0 ? (
          <div className="flex items-center justify-center h-[300px] text-gray-400">
            Transcripts will appear here...
          </div>
        ) : (
          <div className="p-4 space-y-3">
            {transcripts.map((t, i) => (
              <div key={i} className="flex gap-3">
                <span className="text-xs text-gray-400 mt-1 whitespace-nowrap">
                  {t.timestamp}
                </span>
                <p className={`flex-1 ${t.isFinal ? 'text-gray-900' : 'text-gray-500 italic'}`}>
                  {t.text}
                </p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
