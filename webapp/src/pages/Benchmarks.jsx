import { useState, useEffect } from 'react'
import {
  Chart as ChartJS, CategoryScale, LinearScale, BarElement,
  Title, Tooltip, Legend
} from 'chart.js'
import { Bar } from 'react-chartjs-2'
import { api, createSSE } from '../api'

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend)

const CHART_COLORS = {
  ColdInfer: '#F97316',
  PyTorch:   '#F7F1E3',
  TFLite:    '#67E8F9',
  TFLM:      '#C084FC',
}

const ENGINE_LABELS = {
  Custom_Infer_us: 'ColdInfer',
  PT_Infer_us:     'PyTorch',
  TFLite_Infer_us: 'TFLite',
  TFLM_Infer_us:   'TFLM',
}

const MEMORY_LABELS = {
  Custom_RSS_KB: 'ColdInfer',
  PT_RSS_KB:     'PyTorch',
  TFLite_RSS_KB: 'TFLite',
  TFLM_RSS_KB:   'TFLM',
}

export default function Benchmarks() {
  const [data, setData]         = useState([])
  const [model, setModel]       = useState('all')
  const [logs, setLogs]         = useState([])
  const [running, setRunning]   = useState(false)
  const [stopFn, setStopFn]     = useState(null)

  useEffect(() => {
    api.benchmarks().then(r => setData(r.data)).catch(() => {})
  }, [])

  const models = ['all', ...new Set(data.map(d => d.Model_Name).filter(Boolean))]
  const filtered = model === 'all' ? data : data.filter(d => d.Model_Name === model)

  // Build chart datasets — take last entry per model
  const latestByModel = {}
  for (const row of filtered) {
    latestByModel[row.Model_Name] = row
  }
  const chartLabels = Object.keys(latestByModel)

  function makeDataset(keyMap, title) {
    return Object.entries(keyMap).map(([key, label]) => ({
      label,
      data: chartLabels.map(m => latestByModel[m]?.[key] ?? 0),
      backgroundColor: CHART_COLORS[label],
      borderColor: CHART_COLORS[label],
      borderWidth: 1,
    }))
  }

  const latencyChart = {
    labels: chartLabels,
    datasets: makeDataset(ENGINE_LABELS, 'Latency (µs)'),
  }
  const memChart = {
    labels: chartLabels,
    datasets: makeDataset(MEMORY_LABELS, 'Memory (KB)'),
  }

  const chartOptions = (title, yLabel) => ({
    responsive: true,
    plugins: {
      legend: { labels: { color: '#F7F1E3', font: { size: 11 } } },
      title: { display: true, text: title, color: '#F7F1E3', font: { size: 13, weight: '600' } },
    },
    scales: {
      x: { ticks: { color: '#D8CFBF' }, grid: { color: 'rgba(74,120,148,.35)' } },
      y: { ticks: { color: '#D8CFBF' }, grid: { color: 'rgba(74,120,148,.35)' }, title: { display: true, text: yLabel, color: '#D8CFBF' } },
    },
  })

  function runBenchmark() {
    setLogs([])
    setRunning(true)
    const stop = createSSE('/api/benchmarks/run', '{}', {
      onLine: line => setLogs(l => [...l, line]),
      onDone: ({ exit_code }) => {
        setLogs(l => [...l, exit_code === 0 ? '[done] benchmark suite completed' : `[error] benchmark suite exited with code ${exit_code}`])
        setRunning(false)
        api.benchmarks().then(r => setData(r.data)).catch(() => {})
      },
      onError: e => { setLogs(l => [...l, `ERROR: ${e.message}`]); setRunning(false) },
    })
    setStopFn(() => stop)
  }

  return (
    <div className="page-body stack">
      {/* Controls */}
      <div className="row-sb">
        <div className="row">
          <label className="form-label" style={{ margin: 0 }}>Model:</label>
          <select className="form-select" style={{ width: 240 }} value={model} onChange={e => setModel(e.target.value)}>
            {models.map(m => <option key={m}>{m}</option>)}
          </select>
        </div>
        <div className="row">
          {running && (
            <button className="btn btn-ghost btn-sm" onClick={() => { stopFn?.(); setRunning(false) }}>
              ■ Stop
            </button>
          )}
          <button className="btn btn-primary btn-sm" onClick={runBenchmark} disabled={running}>
            {running ? '⟳ Running...' : '▶ Run Benchmark Suite'}
          </button>
        </div>
      </div>

      {/* Stats row */}
      {chartLabels.length > 0 && (
        <div className="grid-4">
          {Object.entries(ENGINE_LABELS).map(([key, label]) => {
            const vals = chartLabels.map(m => latestByModel[m]?.[key]).filter(Boolean)
            const avg = vals.length ? (vals.reduce((a, b) => a + b, 0) / vals.length) : 0
            return (
              <div className="stat-card" key={label}>
                <div className="stat-label">{label}</div>
                <div className="stat-value" style={{ fontSize: 20, color: CHART_COLORS[label] }}>
                  {avg.toFixed(0)} µs
                </div>
                <div className="stat-sub">avg latency</div>
              </div>
            )
          })}
        </div>
      )}

      {/* Charts */}
      {chartLabels.length > 0 && (
        <div className="grid-2">
          <div className="card">
            <Bar data={latencyChart} options={chartOptions('Inference Latency', 'µs')} />
          </div>
          <div className="card">
            <Bar data={memChart} options={chartOptions('Memory RSS', 'KB')} />
          </div>
        </div>
      )}

      {/* Raw data table */}
      {filtered.length > 0 && (
        <div className="card">
          <div className="card-title">Raw Benchmark Log</div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, fontFamily: 'var(--font-mono)' }}>
              <thead>
                <tr>
                  {['Date','Model','ColdInfer µs','PyTorch µs','TFLite µs','TFLM µs','ColdInfer KB','PyTorch KB'].map(h => (
                    <th key={h} style={{ padding: '6px 10px', textAlign: 'left', color: 'var(--text-muted)', borderBottom: '1px solid var(--border)' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filtered.slice(-20).reverse().map((row, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
                    <td style={{ padding: '5px 10px', color: 'var(--text-muted)' }}>{row.Date}</td>
                    <td style={{ padding: '5px 10px' }}>{row.Model_Name}</td>
                    <td style={{ padding: '5px 10px', color: 'var(--accent)' }}>{row.Custom_Infer_us}</td>
                    <td style={{ padding: '5px 10px' }}>{row.PT_Infer_us}</td>
                    <td style={{ padding: '5px 10px' }}>{row.TFLite_Infer_us}</td>
                    <td style={{ padding: '5px 10px', color: 'var(--green)' }}>{row.TFLM_Infer_us}</td>
                    <td style={{ padding: '5px 10px', color: 'var(--accent)' }}>{row.Custom_RSS_KB}</td>
                    <td style={{ padding: '5px 10px' }}>{row.PT_RSS_KB}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Log output */}
      {logs.length > 0 && (
        <div className="card">
          <div className="card-title row-sb">
            Benchmark Output
            <button className="btn btn-ghost btn-sm" onClick={() => setLogs([])}>Clear</button>
          </div>
          <div className="log-box">
            {logs.map((l, i) => (
              <div key={i} className={l.startsWith('ERROR') ? 'log-line-err' : ''}>{l || '\u00a0'}</div>
            ))}
          </div>
        </div>
      )}

      {data.length === 0 && !running && (
        <div className="card" style={{ textAlign: 'center', padding: '48px', color: 'var(--text-muted)' }}>
          No benchmark data yet. Click <strong>Run Benchmark Suite</strong> to generate results.
        </div>
      )}
    </div>
  )
}
