import { useState, useEffect } from 'react'
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement,
  LineElement, Title, Tooltip, Legend
} from 'chart.js'
import { Line } from 'react-chartjs-2'
import { api } from '../api'

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend)

const COLORS = ['#F97316', '#34D399', '#C084FC', '#67E8F9']
const ENGINES = ['Custom_Infer_us', 'TFLM_Infer_us', 'PT_Infer_us', 'TFLite_Infer_us']
const LABELS  = ['ColdInfer', 'TFLM', 'PyTorch', 'TFLite']

export default function PerformanceHistory() {
  const [data, setData]       = useState([])
  const [filter, setFilter]   = useState('all')
  const [metric, setMetric]   = useState('latency')

  useEffect(() => {
    api.benchmarks().then(r => setData(r.data)).catch(() => {})
  }, [])

  const models = ['all', ...new Set(data.map(d => d.Model_Name).filter(Boolean))]
  const filtered = filter === 'all' ? data : data.filter(d => d.Model_Name === filter)

  const keyMap = metric === 'latency'
    ? { Custom_Infer_us: 'ColdInfer', TFLM_Infer_us: 'TFLM', PT_Infer_us: 'PyTorch', TFLite_Infer_us: 'TFLite' }
    : { Custom_RSS_KB: 'ColdInfer', TFLM_RSS_KB: 'TFLM', PT_RSS_KB: 'PyTorch', TFLite_RSS_KB: 'TFLite' }

  const labels = filtered.map(d => d.Date?.split(' ')[0] || '')
  const chartData = {
    labels,
    datasets: Object.entries(keyMap).map(([key, label], idx) => ({
      label,
      data: filtered.map(d => d[key] ?? null),
      borderColor: COLORS[idx],
      backgroundColor: COLORS[idx] + '33',
      tension: 0.3,
      pointRadius: 3,
    })),
  }

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: { labels: { color: '#F7F1E3' } },
      title: {
        display: true,
        text: metric === 'latency' ? 'Inference Latency Over Time (µs)' : 'Memory RSS Over Time (KB)',
        color: '#F7F1E3',
      },
    },
    scales: {
      x: { ticks: { color: '#D8CFBF', maxRotation: 45 }, grid: { color: 'rgba(74,120,148,.35)' } },
      y: { ticks: { color: '#D8CFBF' }, grid: { color: 'rgba(74,120,148,.35)' } },
    },
  }

  // Summary stats across all filtered data
  const latestByEngine = {}
  if (filtered.length > 0) {
    const last = filtered[filtered.length - 1]
    Object.entries(keyMap).forEach(([k, label]) => { latestByEngine[label] = last[k] })
  }

  return (
    <div className="page-body stack">
      {/* Filters */}
      <div className="card row-sb">
        <div className="row">
          <div className="form-group" style={{ flexDirection: 'row', alignItems: 'center', gap: 8, margin: 0 }}>
            <label className="form-label" style={{ margin: 0, whiteSpace: 'nowrap' }}>Model:</label>
            <select className="form-select" style={{ width: 220 }} value={filter} onChange={e => setFilter(e.target.value)}>
              {models.map(m => <option key={m}>{m}</option>)}
            </select>
          </div>
          <div className="form-group" style={{ flexDirection: 'row', alignItems: 'center', gap: 8, margin: 0 }}>
            <label className="form-label" style={{ margin: 0, whiteSpace: 'nowrap' }}>Metric:</label>
            <select className="form-select" style={{ width: 160 }} value={metric} onChange={e => setMetric(e.target.value)}>
              <option value="latency">Latency (µs)</option>
              <option value="memory">Memory (KB)</option>
            </select>
          </div>
        </div>
        <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>{filtered.length} data points</span>
      </div>

      {/* Latest summary */}
      {Object.keys(latestByEngine).length > 0 && (
        <div className="grid-4">
          {Object.entries(latestByEngine).map(([label, val], i) => (
            <div className="stat-card" key={label}>
              <div className="stat-label">{label}</div>
              <div className="stat-value" style={{ color: COLORS[i], fontSize: 22 }}>
                {val ?? '—'} <span style={{ fontSize: 12, fontWeight: 400 }}>{metric === 'latency' ? 'µs' : 'KB'}</span>
              </div>
              <div className="stat-sub">latest run</div>
            </div>
          ))}
        </div>
      )}

      {/* Chart */}
      {filtered.length > 1 && (
        <div className="card">
          <Line data={chartData} options={chartOptions} />
        </div>
      )}

      {/* Table */}
      {filtered.length > 0 && (
        <div className="card">
          <div className="card-title">All Runs</div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, fontFamily: 'var(--font-mono)' }}>
              <thead>
                <tr>
                  {['Date', 'Model', 'ColdInfer µs', 'TFLM µs', 'PyTorch µs', 'TFLite µs', 'ColdInfer KB', 'TFLM KB'].map(h => (
                    <th key={h} style={{ padding: '6px 10px', textAlign: 'left', color: 'var(--text-muted)', borderBottom: '1px solid var(--border)' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {[...filtered].reverse().map((row, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
                    <td style={{ padding: '5px 10px', color: 'var(--text-muted)' }}>{row.Date}</td>
                    <td style={{ padding: '5px 10px' }}>{row.Model_Name}</td>
                    <td style={{ padding: '5px 10px', color: '#F97316' }}>{row.Custom_Infer_us}</td>
                    <td style={{ padding: '5px 10px', color: '#34D399' }}>{row.TFLM_Infer_us}</td>
                    <td style={{ padding: '5px 10px' }}>{row.PT_Infer_us}</td>
                    <td style={{ padding: '5px 10px' }}>{row.TFLite_Infer_us}</td>
                    <td style={{ padding: '5px 10px', color: '#F97316' }}>{row.Custom_RSS_KB}</td>
                    <td style={{ padding: '5px 10px', color: '#34D399' }}>{row.TFLM_RSS_KB}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {data.length === 0 && (
        <div className="card" style={{ textAlign: 'center', padding: '48px', color: 'var(--text-muted)' }}>
          No performance data yet. Run benchmarks to populate this view.
        </div>
      )}
    </div>
  )
}
