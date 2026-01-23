import { useState, useRef, useCallback } from 'react'
import './App.css'

interface PredictionResult {
  class: string
  class_id: number
  confidence: number
}

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function App() {
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isRecording, setIsRecording] = useState(false)
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null)
  const [dragActive, setDragActive] = useState(false)

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const fileInputRef = useRef<HTMLInputElement>(null)

  const classifyAudio = async (file: File | Blob) => {
    setIsLoading(true)
    setError(null)
    setResult(null)

    const formData = new FormData()
    if (file instanceof File) {
      formData.append('file', file)
    } else {
      formData.append('file', file, 'recording.wav')
    }

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`)
      }

      const data: PredictionResult = await response.json()
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to classify audio')
    } finally {
      setIsLoading(false)
    }
  }

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      setAudioBlob(null)
      classifyAudio(file)
    }
  }

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    const file = e.dataTransfer.files?.[0]
    if (file && file.type.startsWith('audio/')) {
      setAudioBlob(null)
      classifyAudio(file)
    } else {
      setError('Please upload an audio file')
    }
  }, [])

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      })

      mediaRecorderRef.current = mediaRecorder
      chunksRef.current = []

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data)
        }
      }

      mediaRecorder.onstop = async () => {
        const webmBlob = new Blob(chunksRef.current, { type: 'audio/webm' })

        // Convert webm to wav using AudioContext
        const wavBlob = await convertToWav(webmBlob)
        setAudioBlob(wavBlob)
        classifyAudio(wavBlob)

        stream.getTracks().forEach(track => track.stop())
      }

      mediaRecorder.start()
      setIsRecording(true)
      setError(null)
      setResult(null)
    } catch (err) {
      setError('Microphone access denied. Please allow microphone access.')
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
    }
  }

  const convertToWav = async (webmBlob: Blob): Promise<Blob> => {
    const audioContext = new AudioContext({ sampleRate: 22050 })
    const arrayBuffer = await webmBlob.arrayBuffer()
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer)

    // Get mono channel
    const channelData = audioBuffer.getChannelData(0)
    const wavBuffer = encodeWav(channelData, audioContext.sampleRate)

    audioContext.close()
    return new Blob([wavBuffer], { type: 'audio/wav' })
  }

  const encodeWav = (samples: Float32Array, sampleRate: number): ArrayBuffer => {
    const buffer = new ArrayBuffer(44 + samples.length * 2)
    const view = new DataView(buffer)

    // WAV header
    const writeString = (offset: number, str: string) => {
      for (let i = 0; i < str.length; i++) {
        view.setUint8(offset + i, str.charCodeAt(i))
      }
    }

    writeString(0, 'RIFF')
    view.setUint32(4, 36 + samples.length * 2, true)
    writeString(8, 'WAVE')
    writeString(12, 'fmt ')
    view.setUint32(16, 16, true) // Subchunk1Size
    view.setUint16(20, 1, true) // AudioFormat (PCM)
    view.setUint16(22, 1, true) // NumChannels (Mono)
    view.setUint32(24, sampleRate, true)
    view.setUint32(28, sampleRate * 2, true) // ByteRate
    view.setUint16(32, 2, true) // BlockAlign
    view.setUint16(34, 16, true) // BitsPerSample
    writeString(36, 'data')
    view.setUint32(40, samples.length * 2, true)

    // Convert float to 16-bit PCM
    let offset = 44
    for (let i = 0; i < samples.length; i++) {
      const sample = Math.max(-1, Math.min(1, samples[i]))
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true)
      offset += 2
    }

    return buffer
  }

  const formatClassName = (className: string) => {
    return className.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
  }

  return (
    <div className="app">
      <header className="header">
        <h1>Echo</h1>
        <p className="subtitle">Urban Sound Classification</p>
      </header>

      <main className="main">
        <section className="upload-section">
          <div
            className={`drop-zone ${dragActive ? 'drag-active' : ''}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="audio/*,.wav"
              onChange={handleFileUpload}
              className="file-input"
            />
            <div className="drop-zone-content">
              <svg className="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
              <p>Drop audio file here or click to upload</p>
              <span className="file-hint">.wav files recommended</span>
            </div>
          </div>
        </section>

        <div className="divider">
          <span>or</span>
        </div>

        <section className="record-section">
          <button
            className={`record-btn ${isRecording ? 'recording' : ''}`}
            onClick={isRecording ? stopRecording : startRecording}
            disabled={isLoading}
          >
            {isRecording ? (
              <>
                <span className="pulse"></span>
                Stop Recording
              </>
            ) : (
              <>
                <svg className="mic-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
                  <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                  <line x1="12" y1="19" x2="12" y2="23" />
                  <line x1="8" y1="23" x2="16" y2="23" />
                </svg>
                Record Audio
              </>
            )}
          </button>
          {isRecording && (
            <p className="recording-hint">Recording... Speak or make sounds for 2-4 seconds</p>
          )}
        </section>

        {isLoading && (
          <section className="result-section">
            <div className="loader"></div>
            <p>Analyzing audio...</p>
          </section>
        )}

        {error && (
          <section className="result-section error">
            <p>{error}</p>
          </section>
        )}

        {result && !isLoading && (
          <section className="result-section success">
            <h2>Classification Result</h2>
            <div className="result-card">
              <div className="result-class">
                <span className="label">Detected Sound</span>
                <span className="value">{formatClassName(result.class)}</span>
              </div>
              <div className="result-confidence">
                <span className="label">Confidence</span>
                <div className="confidence-bar">
                  <div
                    className="confidence-fill"
                    style={{ width: `${result.confidence * 100}%` }}
                  ></div>
                </div>
                <span className="confidence-value">{(result.confidence * 100).toFixed(1)}%</span>
              </div>
            </div>
          </section>
        )}

        {audioBlob && !isLoading && (
          <section className="playback-section">
            <audio controls src={URL.createObjectURL(audioBlob)} />
          </section>
        )}
      </main>

      <footer className="footer">
        <p>Classifies urban sounds into 10 categories using deep learning</p>
      </footer>
    </div>
  )
}

export default App
