import { useMemo, useState } from 'react'
import {
  BarElement, BarController, CategoryScale, Chart as ChartJS, LinearScale,
  LineElement, PointElement, Title, Tooltip, Legend,
} from 'chart.js'
import { Bar, Line } from 'react-chartjs-2'
import { Activity, BarChart3, CheckCircle2, FileImage, Gauge, Play, RotateCcw, Terminal } from 'lucide-react'
import { api } from '../api'

ChartJS.register(CategoryScale, LinearScale, BarElement, BarController, PointElement, LineElement, Title, Tooltip, Legend)

const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  animation: false,
  plugins: { legend: { labels: { color: '#F7F1E3' } } },
  scales: {
    x: { ticks: { color: '#D8CFBF' }, grid: { color: 'rgba(74,120,148,.35)' } },
    y: { ticks: { color: '#D8CFBF' }, grid: { color: 'rgba(74,120,148,.35)' } },
  },
}

export default function YoloLab() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState('')
  const [result, setResult] = useState(null)
  const [comparison, setComparison] = useState(null)
  const [benchmark, setBenchmark] = useState(null)
  const [confidence, setConfidence] = useState(0.25)
  const [target, setTarget] = useState('simulated-esp32')
  const [arenaKb, setArenaKb] = useState(520)
  const [running, setRunning] = useState(false)
  const [benchmarking, setBenchmarking] = useState(false)
  const [error, setError] = useState('')
  const [consoleLines, setConsoleLines] = useState([
    '[ready] ColdInfer YOLO console',
    '[ready] Select a COCO8 image or upload an image to begin',
  ])
  const [activeTab, setActiveTab] = useState('detect')
  const [consoleTab, setConsoleTab] = useState('console')

  function chooseFile(event) {
    const next = event.target.files?.[0]
    if (!next) return
    setFile(next)
    setPreview(URL.createObjectURL(next))
    setResult(null)
    setComparison(null)
    setError('')
    setConsoleLines(lines => [...lines, `[input] selected ${next.name}`])
  }

  async function detect() {
    if (!file || running) return
    setRunning(true)
    setError('')
    setConsoleLines(lines => [...lines, `[run] starting ${target} detection`, `[input] uploading ${file.name}`])
    try {
      const next = await api.yoloDetect(file, confidence, target, arenaKb)
      setResult(next)
      setConsoleLines(lines => [...lines, ...(next.console || []), '[done] detection request completed'])
    } catch (e) {
      setError(e.message)
      setConsoleLines(lines => [...lines, `[error] ${e.message}`])
    } finally {
      setRunning(false)
    }
  }

  async function compare() {
    if (!file || running) return
    setRunning(true)
    setError('')
    setConsoleLines(lines => [...lines, '[run] running normal and quantized models on the same image'])
    try {
      const next = await api.yoloCompare(file, confidence)
      setComparison(next)
      setConsoleLines(lines => [...lines, `[compare] normal detections: ${next.normal_count}`, `[compare] quantized detections: ${next.quantized_count}`, `[done] comparison completed`])
    } catch (e) {
      setError(e.message)
      setConsoleLines(lines => [...lines, `[error] ${e.message}`])
    } finally {
      setRunning(false)
    }
  }

  async function runBenchmark() {
    if (benchmarking) return
    setBenchmarking(true)
    setError('')
    setConsoleLines(lines => [...lines, '[benchmark] loading COCO8 validation set', '[benchmark] running both models on every image'])
    try {
      const next = await api.yoloBenchmark(confidence)
      setBenchmark(next)
      setConsoleLines(lines => [...lines, `[benchmark] completed ${next.images} images`, `[done] benchmark averages: ${next.summary.normal_avg_ms} ms normal / ${next.summary.quantized_avg_ms} ms quantized`])
    } catch (e) {
      setError(e.message)
      setConsoleLines(lines => [...lines, `[error] ${e.message}`])
    } finally {
      setBenchmarking(false)
    }
  }

  const boxes = useMemo(() => result?.detections || [], [result])
  const comparisonChart = comparison ? {
    labels: ['Detections', 'Average confidence'],
    datasets: [
      { label: 'Normal FP32', data: [comparison.normal_count, averageConfidence(comparison.normal.detections) * 100], backgroundColor: '#F7F1E3' },
      { label: 'Quantized INT8', data: [comparison.quantized_count, averageConfidence(comparison.quantized.detections) * 100], backgroundColor: '#F97316' },
    ],
  } : null
  const benchmarkChart = benchmark ? {
    labels: benchmark.rows.map(row => row.image),
    datasets: [
      { label: 'Normal FP32 ms', data: benchmark.rows.map(row => row.normal_ms), borderColor: '#F7F1E3', backgroundColor: '#F7F1E3' },
      { label: 'Quantized INT8 ms', data: benchmark.rows.map(row => row.quantized_ms), borderColor: '#F97316', backgroundColor: '#F97316' },
    ],
  } : null

  return (
    <div className="page-body stack">
      <div className="simulation-hero">
        <div>
          <div className="eyebrow">EMBEDDED COMPUTER VISION</div>
          <h1>YOLO evaluation lab</h1>
          <p>Run the quantized target inside a fixed arena, compare it with normal FP32 inference on the same image, and benchmark both models over the bundled COCO8 validation set.</p>
        </div>
        <span className="badge badge-purple"><Activity size={14} /> INT8 deployment profile</span>
      </div>

      <div className="tab-bar">
        <button className={activeTab === 'detect' ? 'tab-button active' : 'tab-button'} onClick={() => setActiveTab('detect')}><FileImage size={16} /> Detection</button>
        <button className={activeTab === 'compare' ? 'tab-button active' : 'tab-button'} onClick={() => setActiveTab('compare')}><BarChart3 size={16} /> Same-image comparison</button>
        <button className={activeTab === 'benchmark' ? 'tab-button active' : 'tab-button'} onClick={() => setActiveTab('benchmark')}><Activity size={16} /> Dataset benchmark</button>
      </div>

      <div className="card">
        <div className="row yolo-controls">
          <div className="form-group file-control"><label className="form-label">Test image</label><input className="form-input" type="file" accept="image/*" onChange={chooseFile} /></div>
          <div className="form-group compact-control"><label className="form-label">Confidence {confidence.toFixed(2)}</label><input type="range" min="0.05" max="0.9" step="0.05" value={confidence} onChange={e => setConfidence(Number(e.target.value))} /></div>
          <div className="form-group compact-control"><label className="form-label">Target</label><select className="form-select" value={target} onChange={e => setTarget(e.target.value)}><option value="simulated-esp32">Simulated ESP32 INT8</option><option value="simulated-esp32-full">Simulated ESP32 FP32 (slow)</option><option value="local">Desktop reference</option></select></div>
          {target !== 'local' && <div className="form-group compact-control"><label className="form-label">Arena {arenaKb} KB</label><input type="range" min="128" max="8192" step="64" value={arenaKb} onChange={e => setArenaKb(Number(e.target.value))} /></div>}
          {activeTab !== 'benchmark' && <button className="btn btn-primary" onClick={activeTab === 'compare' ? compare : detect} disabled={!file || running}><Play size={16} /> {running ? 'Running' : activeTab === 'compare' ? 'Compare models' : 'Run detection'}</button>}
          {activeTab === 'benchmark' && <button className="btn btn-primary" onClick={runBenchmark} disabled={benchmarking}><RotateCcw size={16} /> {benchmarking ? 'Benchmarking' : 'Run COCO8 benchmark'}</button>}
        </div>
      </div>

      <div className="card yolo-console-card yolo-telemetry-panel">
        <div className="console-tabs"><button className={consoleTab === 'console' ? 'console-tab active' : 'console-tab'} onClick={() => setConsoleTab('console')}><Terminal size={15} /> Console</button><button className={consoleTab === 'metrics' ? 'console-tab active' : 'console-tab'} onClick={() => setConsoleTab('metrics')}><Gauge size={15} /> Run metrics</button><button className="btn btn-ghost btn-sm console-clear" onClick={() => setConsoleLines([])}>Clear</button></div>
        {consoleTab === 'console' ? <div className="yolo-console" aria-live="polite">{consoleLines.map((line, index) => <div key={`${line}-${index}`} className={line.includes('[error]') ? 'console-line error' : line.includes('[done]') ? 'console-line success' : 'console-line'}>{line}</div>)}{(running || benchmarking) && <div className="console-line running">[run] work in progress<span className="console-cursor">_</span></div>}</div> : <MetricsView result={result} comparison={comparison} benchmark={benchmark} />}
      </div>

      {error && <div className="card error-banner">{error}</div>}

      {activeTab === 'detect' && <DetectionView preview={preview} result={result} boxes={boxes} />}
      {activeTab === 'compare' && <ComparisonView preview={preview} result={comparison} chart={comparisonChart} />}
      {activeTab === 'benchmark' && <BenchmarkView result={benchmark} chart={benchmarkChart} />}

    </div>
  )
}

function averageConfidence(detections = []) {
  return detections.length ? detections.reduce((sum, item) => sum + item.confidence, 0) / detections.length : 0
}

function DetectionView({ preview, result, boxes }) {
  return <div className="grid-2"><div className="card yolo-preview-card"><div className="card-title">Annotated frame</div><Frame preview={preview} result={result} boxes={boxes} /></div><div className="card"><div className="card-title row-sb">Detections <span className="badge badge-blue">{boxes.length} objects</span></div>{boxes.length ? <div className="confidence-list">{boxes.map((box, index) => <div className="confidence-row" key={index}><span className="confidence-label winner">{box.label}</span><div className="confidence-track"><div className="confidence-fill winner-fill" style={{ width: `${box.confidence * 100}%` }} /></div><span className="confidence-number">{(box.confidence * 100).toFixed(1)}%</span></div>)}</div> : <div className="empty-visual">Run detection to populate results.</div>}</div></div>
}

function ComparisonView({ preview, result, chart }) {
  return <div className="stack"><div className="grid-2"><div className="card yolo-preview-card"><div className="card-title">Normal FP32 output</div><Frame preview={preview} result={result} boxes={result?.normal?.detections || []} /></div><div className="card yolo-preview-card"><div className="card-title">Quantized INT8 output</div><Frame preview={preview} result={result} boxes={result?.quantized?.detections || []} /></div></div><div className="card chart-card"><div className="card-title">Normal versus quantized</div>{chart ? <Bar data={chart} options={chartOptions} /> : <div className="empty-visual">Run comparison to plot differences.</div>}<div className="comparison-summary">{result && <><span><CheckCircle2 size={15} /> Count delta: {result.count_delta}</span><span>Matched classes: {result.matched_labels.join(', ') || 'none'}</span><span>Matched boxes: {result.matches.map(match => `${match.label} IoU ${match.iou}`).join(' | ') || 'none'}</span></>}</div></div></div>
}

function BenchmarkView({ result, chart }) {
  return <div className="stack">{result ? <><div className="grid-4"><div className="stat-card"><div className="stat-label">Images</div><div className="stat-value">{result.images}</div></div><div className="stat-card"><div className="stat-label">Normal average</div><div className="stat-value">{result.summary.normal_avg_ms}<span className="unit"> ms</span></div></div><div className="stat-card"><div className="stat-label">Quantized average</div><div className="stat-value">{result.summary.quantized_avg_ms}<span className="unit"> ms</span></div></div><div className="stat-card"><div className="stat-label">Quantized detections</div><div className="stat-value">{result.summary.quantized_total_detections}</div></div></div><div className="card chart-card large-chart"><div className="card-title">Inference latency per COCO8 image</div><Line data={chart} options={chartOptions} /></div><div className="card benchmark-table"><div className="card-title">Image-level comparison</div>{result.rows.map(row => <div className="benchmark-row" key={row.image}><span>{row.image}</span><span>{row.normal_ms} ms</span><span>{row.quantized_ms} ms</span><span>{row.normal_detections} / {row.quantized_detections} detections</span></div>)}</div></> : <div className="card empty-visual">Run the COCO8 benchmark to compare both models across the testing set.</div>}</div>
}

function Frame({ preview, result, boxes }) {
  if (!preview) return <div className="empty-visual">Choose an image to begin.</div>
  return <div className="yolo-frame"><img src={preview} alt="Detection input" />{result && boxes.map((box, index) => <div key={index} className="yolo-box" style={{ left: `${(box.x1 / result.width) * 100}%`, top: `${(box.y1 / result.height) * 100}%`, width: `${((box.x2 - box.x1) / result.width) * 100}%`, height: `${((box.y2 - box.y1) / result.height) * 100}%` }}><span>{box.label} {(box.confidence * 100).toFixed(0)}%</span></div>)}</div>
}

function MetricsView({ result, comparison, benchmark }) {
  const timings = result?.timings_ms || comparison?.quantized?.timings_ms
  return <div className="grid-4 yolo-metrics"><div className="stat-card"><div className="stat-label">Preprocess</div><div className="stat-value">{timings?.preprocess ?? '—'}<span className="unit"> ms</span></div></div><div className="stat-card"><div className="stat-label">Inference</div><div className="stat-value">{timings?.inference ?? '—'}<span className="unit"> ms</span></div></div><div className="stat-card"><div className="stat-label">Postprocess</div><div className="stat-value">{timings?.postprocess ?? '—'}<span className="unit"> ms</span></div></div><div className="stat-card"><div className="stat-label">Benchmark images</div><div className="stat-value">{benchmark?.images ?? '—'}</div></div></div>
}
