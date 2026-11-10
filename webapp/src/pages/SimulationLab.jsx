import { useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api'

const MODEL_COPY = {
  mnist: {
    title: 'MNIST digit classifier',
    description: '784 normalized pixels → 128 → 64 → 10 classes',
    visualization: '28 × 28 input image',
  },
  kws: {
    title: 'Keyword spotter',
    description: '1,000 audio features → 128 → 64 → 10 classes',
    visualization: '50 × 20 audio feature map',
  },
}

function heatColor(value, min, max) {
  const ratio = max === min ? 0 : (value - min) / (max - min)
  const clamped = Math.max(0, Math.min(1, ratio))
  return `hsl(${220 - clamped * 190} 90% ${12 + clamped * 58}%)`
}

export default function SimulationLab() {
  const [model, setModel] = useState('mnist')
  const [inputVariant, setInputVariant] = useState(0)
  const [customInput, setCustomInput] = useState(null)
  const [audioUrl, setAudioUrl] = useState('')
  const [recording, setRecording] = useState(false)
  const recorderRef = useRef(null)
  const [sramCap, setSramCap] = useState(520)
  const [result, setResult] = useState(null)
  const [running, setRunning] = useState(false)
  const [autoRun, setAutoRun] = useState(false)
  const [error, setError] = useState('')

  async function runInference() {
    if (running) return
    setRunning(true)
    setError('')
    try {
      setResult(await api.simulationInfer(model, sramCap, inputVariant, customInput))
    } catch (e) {
      setError(e.message)
    } finally {
      setRunning(false)
    }
  }

  useEffect(() => {
    runInference()
  }, [model, inputVariant, customInput])

  useEffect(() => {
    if (!autoRun) return undefined
    const timer = setInterval(runInference, 2500)
    return () => clearInterval(timer)
  }, [autoRun, model, sramCap, running])

  const inputRange = useMemo(() => {
    const input = result?.input || []
    return { min: Math.min(...input), max: Math.max(...input) }
  }, [result])

  const copy = MODEL_COPY[model]
  const probabilities = result?.probabilities || []

  async function loadAudio(file) {
    const context = new AudioContext()
    const buffer = await context.decodeAudioData(await file.arrayBuffer())
    const channel = buffer.getChannelData(0)
    const samples = Array.from({ length: 16000 }, (_, index) => channel[Math.floor(index * channel.length / 16000)] || 0)
    const values = Array.from({ length: 1000 }, (_, index) => {
      const start = index * 16
      return samples.slice(start, start + 16).reduce((sum, value) => sum + value, 0) / 16
    })
    setCustomInput(values)
    setAudioUrl(URL.createObjectURL(file))
    setInputVariant(0)
    await context.close()
  }

  async function startRecording() {
    if (recording) {
      recorderRef.current?.stop()
      return
    }
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    const chunks = []
    const recorder = new MediaRecorder(stream)
    recorder.ondataavailable = event => chunks.push(event.data)
    recorder.onstop = async () => {
      stream.getTracks().forEach(track => track.stop())
      const blob = new Blob(chunks, { type: recorder.mimeType })
      await loadAudio(new File([blob], 'recorded-keyword.webm', { type: recorder.mimeType }))
      setRecording(false)
    }
    recorderRef.current = recorder
    recorder.start()
    setRecording(true)
  }

  function paintDigit(event) {
    if (event.buttons !== 1 && event.type !== 'pointerdown') return
    const rect = event.currentTarget.getBoundingClientRect()
    const x = Math.floor((event.clientX - rect.left) / rect.width * 28)
    const y = Math.floor((event.clientY - rect.top) / rect.height * 28)
    if (x < 0 || x >= 28 || y < 0 || y >= 28) return
    const next = [...(customInput || Array(784).fill(-0.4242))]
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const cellX = x + dx
      const cellY = y + dy
      const index = cellY * 28 + cellX
      if (cellX >= 0 && cellX < 28 && cellY >= 0 && cellY < 28) next[index] = 2.8
    }
    setInputVariant(0)
    setCustomInput(next)
  }

  return (
    <div className="page-body stack">
      <div className="simulation-hero">
        <div>
          <div className="eyebrow">LIVE SIMULATION MODE</div>
          <h1>See the model decide</h1>
          <p>Run the actual ColdInfer C++ engine inside a configurable SRAM budget and visualize its input, prediction, and confidence distribution.</p>
        </div>
        <div className="simulation-pulse"><span className="dot dot-green dot-pulse" /> Native inference ready</div>
      </div>

      <div className="card">
        <div className="grid-3 simulation-controls">
          <div className="form-group">
            <label className="form-label">Demo model</label>
            <select className="form-select" value={model} onChange={e => setModel(e.target.value)}>
              <option value="mnist">MNIST — handwritten digit</option>
              <option value="kws">Keywords — audio feature map</option>
            </select>
          </div>
          <div className="form-group">
            <label className="form-label">Input frame</label>
            <select className="form-select" value={inputVariant} onChange={e => setInputVariant(Number(e.target.value))}>
              <option value={0}>Reference sample</option>
              <option value={1}>Inverted sample</option>
              <option value={2}>Shifted sample</option>
              <option value={3}>Contrast variation</option>
            </select>
          </div>
          {model !== 'mnist' && (
            <div className="form-group">
              <label className="form-label">Your audio input</label>
              <div className="row">
                <input className="form-input" type="file" accept="audio/*" onChange={e => e.target.files[0] && loadAudio(e.target.files[0])} />
                <button className="btn btn-ghost" onClick={startRecording}>{recording ? 'Stop recording' : 'Record'}</button>
              </div>
              {audioUrl && <audio className="audio-player" controls src={audioUrl} />}
            </div>
          )}
          <div className="form-group">
            <label className="form-label">Simulated SRAM budget: {sramCap} KB</label>
            <input type="range" min="128" max="1024" step="64" value={sramCap} onChange={e => setSramCap(Number(e.target.value))} />
          </div>
          <div className="row" style={{ alignSelf: 'end', justifyContent: 'flex-end' }}>
            <label className="toggle-label">
              <input type="checkbox" checked={autoRun} onChange={e => setAutoRun(e.target.checked)} />
              Repeat live
            </label>
            <button className="btn btn-primary" onClick={runInference} disabled={running}>
              {running ? '⟳ Running...' : '▶ Run inference'}
            </button>
          </div>
        </div>
        <div className="simulation-model-copy"><strong>{copy.title}</strong><span>{copy.description}</span></div>
      </div>

      {error && <div className="card error-banner">⚠ {error}</div>}

      {result && (
        <>
          <div className="grid-4">
            <div className="stat-card"><div className="stat-label">Prediction</div><div className="stat-value prediction-value">{result.predicted_label}</div><div className="stat-sub">top class</div></div>
            <div className="stat-card"><div className="stat-label">Confidence</div><div className="stat-value">{(probabilities[result.predicted_index] * 100).toFixed(1)}<span className="unit">%</span></div><div className="stat-sub">softmax probability</div></div>
            <div className="stat-card"><div className="stat-label">Native latency</div><div className="stat-value">{result.inference_us}<span className="unit"> µs</span></div><div className="stat-sub">inside C++ engine</div></div>
            <div className="stat-card"><div className="stat-label">Simulation cap</div><div className="stat-value">{result.sram_cap_kb}<span className="unit"> KB</span></div><div className="stat-sub">RLIMIT_DATA applied</div></div>
          </div>

          <div className="grid-2 simulation-main-grid">
            <div className="card">
              <div className="card-title">{copy.visualization}</div>
              <div
                className={`input-visualization ${model === 'kws' ? 'audio-map' : 'digit-map'} ${model === 'mnist' ? 'digit-input-grid' : ''}`}
                onPointerDown={model === 'mnist' ? paintDigit : undefined}
                onPointerMove={model === 'mnist' ? paintDigit : undefined}
              >
                {(customInput && model === 'mnist' ? customInput : result.input).map((value, index) => <span key={index} style={{ backgroundColor: heatColor(value, inputRange.min, inputRange.max) }} title={`${value.toFixed(3)}`} />)}
              </div>
              {model === 'mnist' && <div className="digit-drawing"><span>Draw directly on the 28 x 28 grid</span><button className="btn btn-ghost btn-sm" onClick={() => setCustomInput(Array(784).fill(-0.4242))}>Clear grid</button></div>}
              <div className="visual-caption">{model === 'mnist' ? 'The baked sample is rendered from the same 784 values consumed by the native binary.' : 'Each cell is one feature from the 1,000-value audio representation consumed by the native binary.'}</div>
            </div>
            <div className="card">
              <div className="card-title">Class confidence</div>
              <div className="confidence-list">
                {result.labels.map((label, index) => (
                  <div className="confidence-row" key={label}>
                    <span className={index === result.predicted_index ? 'confidence-label winner' : 'confidence-label'}>{label}</span>
                    <div className="confidence-track"><div className={`confidence-fill ${index === result.predicted_index ? 'winner-fill' : ''}`} style={{ width: `${Math.max(1, probabilities[index] * 100)}%` }} /></div>
                    <span className="confidence-number">{(probabilities[index] * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
