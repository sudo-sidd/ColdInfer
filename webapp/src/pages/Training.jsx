import { useState } from 'react'
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement,
  LineElement, Title, Tooltip, Legend
} from 'chart.js'
import { Line } from 'react-chartjs-2'
import { createSSE } from '../api'

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend)

export default function Training() {
  const [form, setForm] = useState({
    model: 'mnist', epochs: 1, hidden: 64, embed: 64, lr: 0.003, batch_size: 16,
  })
  const [logs, setLogs]       = useState([])
  const [running, setRunning] = useState(false)
  const [stopFn, setStopFn]   = useState(null)
  const [lossData, setLossData] = useState([])
  const [epoch, setEpoch]     = useState(0)

  function set(k, v) { setForm(f => ({ ...f, [k]: v })) }

  function parseLossLine(line) {
    // Match: "Epoch  5/60 | loss=1.2345 | ppl=3.4 | ..."
    const m = line.match(/Epoch\s+(\d+)\/\d+.*loss=([\d.]+)/)
    if (m) return { epoch: parseInt(m[1]), loss: parseFloat(m[2]) }
    return null
  }

  function start() {
    setLogs([])
    setLossData([])
    setEpoch(0)
    setRunning(true)
    const stop = createSSE('/api/train', JSON.stringify(form), {
      onLine: line => {
        setLogs(l => [...l.slice(-200), line])
        const parsed = parseLossLine(line)
        if (parsed) {
          setEpoch(parsed.epoch)
          setLossData(d => [...d, { x: parsed.epoch, y: parsed.loss }])
        }
      },
      onDone: ({ exit_code }) => {
        setEpoch(form.epochs)
        setLogs(l => [...l, exit_code === 0 ? '[done] training and export completed' : `[error] training exited with code ${exit_code}`])
        setRunning(false)
      },
      onError: e => { setLogs(l => [...l, `ERROR: ${e.message}`]); setRunning(false) },
    })
    setStopFn(() => stop)
  }

  function stop() { stopFn?.(); setRunning(false) }

  const chartData = {
    labels: lossData.map(d => d.x),
    datasets: [{
      label: 'Training Loss',
      data: lossData.map(d => d.y),
      borderColor: '#F97316',
      backgroundColor: 'rgba(249,115,22,.12)',
      tension: 0.3,
      pointRadius: 2,
    }],
  }

  const chartOptions = {
    responsive: true,
    animation: false,
    plugins: {
      legend: { labels: { color: '#F7F1E3' } },
      title: { display: false },
    },
    scales: {
      x: { ticks: { color: '#D8CFBF' }, grid: { color: 'rgba(74,120,148,.35)' }, title: { display: true, text: 'Epoch', color: '#D8CFBF' } },
      y: { ticks: { color: '#D8CFBF' }, grid: { color: 'rgba(74,120,148,.35)' }, title: { display: true, text: 'Loss', color: '#D8CFBF' } },
    },
  }

  const progress = form.epochs > 0 ? Math.round((epoch / form.epochs) * 100) : 0

  return (
    <div className="page-body stack">
      <div className="grid-2" style={{ alignItems: 'start' }}>
        {/* Config panel */}
        <div className="card stack">
          <div className="card-title">Training Configuration</div>
          <div className="form-group">
            <label className="form-label">Model</label>
            <select className="form-select" value={form.model} onChange={e => set('model', e.target.value)}>
              <option value="mnist">MNIST MLP</option>
              <option value="kws">KWS MLP</option>
            </select>
          </div>
          <div className="form-group">
            <label className="form-label">Epochs</label>
            <input className="form-input" type="number" value={form.epochs} onChange={e => set('epochs', parseInt(e.target.value))} />
          </div>

          <div className="row" style={{ justifyContent: 'flex-end', gap: 8 }}>
            {running && <button className="btn btn-ghost btn-sm" onClick={stop}>■ Stop</button>}
            <button className="btn btn-primary" onClick={start} disabled={running}>
              {running ? '⟳ Training...' : '▶ Start Training'}
            </button>
          </div>
        </div>

        {/* Status panel */}
        <div className="stack">
          <div className="card">
            <div className="card-title row-sb">
              Progress
              {running && <span className="badge badge-blue"><span className="dot dot-green dot-pulse" /> Running</span>}
            </div>
            <div style={{ marginBottom: 8 }}>
              <div className="row-sb" style={{ marginBottom: 6 }}>
                <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>Epoch {epoch} / {form.epochs}</span>
                <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--accent)' }}>{progress}%</span>
              </div>
              <div className="progress-bar-track">
                <div className="progress-bar-fill" style={{ width: `${progress}%` }} />
              </div>
            </div>
            {lossData.length > 0 && (
              <div className="grid-2" style={{ marginTop: 12 }}>
                <div className="stat-card" style={{ padding: '12px 16px' }}>
                  <div className="stat-label">Current Loss</div>
                  <div className="stat-value" style={{ fontSize: 18 }}>{lossData[lossData.length-1]?.y.toFixed(4)}</div>
                </div>
                <div className="stat-card" style={{ padding: '12px 16px' }}>
                  <div className="stat-label">Best Loss</div>
                  <div className="stat-value" style={{ fontSize: 18, color: 'var(--green)' }}>{Math.min(...lossData.map(d => d.y)).toFixed(4)}</div>
                </div>
              </div>
            )}
          </div>

          {lossData.length > 1 && (
            <div className="card">
              <div className="card-title">Loss Curve</div>
              <Line data={chartData} options={chartOptions} />
            </div>
          )}
        </div>
      </div>

      {/* Log output */}
      {logs.length > 0 && (
        <div className="card">
          <div className="card-title row-sb">
            Training Log
            <button className="btn btn-ghost btn-sm" onClick={() => setLogs([])}>Clear</button>
          </div>
          <div className="log-box">
            {logs.map((l, i) => (
              <div key={i} className={
                l.startsWith('ERROR') ? 'log-line-err' :
                l.includes('loss=') ? 'log-line-info' : ''
              }>{l || '\u00a0'}</div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
