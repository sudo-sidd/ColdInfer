import { useState, useEffect } from 'react'
import { api, createSSE } from '../api'

export default function QuantizationWizard() {
  const [checkpoint, setCheckpoint] = useState('weights/chatbot_best.pt')
  const [prune, setPrune]           = useState(0.5)
  const [outDir, setOutDir]         = useState('weights')
  const [running, setRunning]       = useState(false)
  const [logs, setLogs]             = useState([])
  const [summary, setSummary]       = useState(null)
  const [stopFn, setStopFn]         = useState(null)

  useEffect(() => {
    api.quantizeSummary().then(setSummary).catch(() => {})
  }, [])

  function run() {
    setLogs([])
    setSummary(null)
    setRunning(true)
    const body = JSON.stringify({ checkpoint, prune, out_dir: outDir })
    const stop = createSSE('/api/quantize', body, {
      onLine: line => setLogs(l => [...l.slice(-200), line]),
      onDone: ev => { setSummary(ev.summary || null); setRunning(false) },
      onError: e => { setLogs(l => [...l, `ERROR: ${e.message}`]); setRunning(false) },
    })
    setStopFn(() => stop)
  }

  const compression = summary ? (summary.fp32_kb / summary.int8_kb).toFixed(1) : null

  return (
    <div className="page-body stack">
      {/* Config */}
      <div className="card stack">
        <div className="card-title">Quantization & Pruning Configuration</div>
        <div className="grid-3">
          <div className="form-group col-span-2">
            <label className="form-label">Checkpoint Path</label>
            <input className="form-input" value={checkpoint} onChange={e => setCheckpoint(e.target.value)} />
          </div>
          <div className="form-group">
            <label className="form-label">Output Directory</label>
            <input className="form-input" value={outDir} onChange={e => setOutDir(e.target.value)} />
          </div>
        </div>

        <div className="form-group">
          <label className="form-label">
            Magnitude Pruning Threshold: <strong style={{ color: 'var(--accent)' }}>{(prune * 100).toFixed(0)}%</strong>
            <span style={{ color: 'var(--text-muted)', marginLeft: 8, fontSize: 11 }}>
              (zeroes out the bottom {(prune*100).toFixed(0)}% of weights by absolute value)
            </span>
          </label>
          <input type="range" min={0} max={0.8} step={0.05} value={prune}
            onChange={e => setPrune(parseFloat(e.target.value))}
            style={{ width: '100%', accentColor: 'var(--accent)' }} />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>
            <span>0% (no pruning)</span>
            <span>40% recommended</span>
            <span>80% (aggressive)</span>
          </div>
        </div>

        {/* Preview */}
        <div className="card" style={{ background: 'var(--bg-input)' }}>
          <div className="card-title">Estimated Output</div>
          <div className="grid-3">
            <div>
              <div className="stat-label">Quantization</div>
              <div style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent)', marginTop: 4 }}>FP32 → INT8</div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>4× size reduction</div>
            </div>
            <div>
              <div className="stat-label">Pruning</div>
              <div style={{ fontFamily: 'var(--font-mono)', color: 'var(--purple)', marginTop: 4 }}>{(prune*100).toFixed(0)}% sparse</div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>skips zero multiplications</div>
            </div>
            <div>
              <div className="stat-label">Combined Est. Savings</div>
              <div style={{ fontFamily: 'var(--font-mono)', color: 'var(--green)', marginTop: 4 }}>~{(4 / (1 - prune * 0.6)).toFixed(1)}× smaller</div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>vs FP32 baseline</div>
            </div>
          </div>
        </div>

        <div className="row" style={{ justifyContent: 'flex-end', gap: 8 }}>
          {running && <button className="btn btn-ghost btn-sm" onClick={() => { stopFn?.(); setRunning(false) }}>■ Stop</button>}
          <button className="btn btn-primary" onClick={run} disabled={running}>
            {running ? '⟳ Running...' : '⚗ Export Quantized Model'}
          </button>
        </div>
      </div>

      {/* Results */}
      {summary && (
        <div className="card stack">
          <div className="card-title">Compression Results</div>
          <div className="grid-4">
            <div className="stat-card">
              <div className="stat-label">FP32 Size</div>
              <div className="stat-value" style={{ fontSize: 20 }}>{summary.fp32_kb}<span style={{ fontSize: 12 }}> KB</span></div>
            </div>
            <div className="stat-card">
              <div className="stat-label">INT8 Size</div>
              <div className="stat-value" style={{ fontSize: 20, color: 'var(--green)' }}>{summary.int8_kb}<span style={{ fontSize: 12 }}> KB</span></div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Compression Ratio</div>
              <div className="stat-value" style={{ fontSize: 20, color: 'var(--accent)' }}>{compression}×</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Avg Sparsity</div>
              <div className="stat-value" style={{ fontSize: 20, color: 'var(--purple)' }}>{(summary.avg_sparsity * 100).toFixed(0)}%</div>
            </div>
          </div>

          <div>
            <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-muted)', marginBottom: 8 }}>Per-Layer Details</div>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, fontFamily: 'var(--font-mono)' }}>
                <thead>
                  <tr>
                    {['Layer', 'Shape', 'Params', 'Sparsity', 'Scale', 'MAE'].map(h => (
                      <th key={h} style={{ padding: '6px 10px', textAlign: 'left', color: 'var(--text-muted)', borderBottom: '1px solid var(--border)' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {summary.layers?.map((l, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
                      <td style={{ padding: '5px 10px', color: 'var(--accent)' }}>{l.name}</td>
                      <td style={{ padding: '5px 10px' }}>{l.shape.join('×')}</td>
                      <td style={{ padding: '5px 10px' }}>{l.params.toLocaleString()}</td>
                      <td style={{ padding: '5px 10px', color: l.sparsity > 0.4 ? 'var(--green)' : 'var(--text-muted)' }}>{(l.sparsity * 100).toFixed(1)}%</td>
                      <td style={{ padding: '5px 10px' }}>{l.scale?.toExponential(3)}</td>
                      <td style={{ padding: '5px 10px', color: l.mae > 0.01 ? 'var(--yellow)' : 'var(--green)' }}>{l.mae?.toExponential(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div style={{ display: 'flex', gap: 8 }}>
            <span className="badge badge-green">✓ chatbot_weights.h written</span>
            <span className="badge badge-green">✓ chatbot_vocab.h written</span>
            <span className="badge badge-blue">Ready to compile</span>
          </div>
        </div>
      )}

      {/* Log */}
      {logs.length > 0 && (
        <div className="card">
          <div className="card-title row-sb">
            Export Output
            <button className="btn btn-ghost btn-sm" onClick={() => setLogs([])}>Clear</button>
          </div>
          <div className="log-box">
            {logs.map((l, i) => (
              <div key={i} className={l.startsWith('ERROR') ? 'log-line-err' : l.startsWith('[export]') ? 'log-line-info' : ''}>{l || '\u00a0'}</div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
