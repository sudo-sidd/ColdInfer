import { useState, useEffect } from 'react'
import { api } from '../api'

export default function ESP32Panel() {
  const [ip, setIp]           = useState('')
  const [port, setPort]       = useState(80)
  const [status, setStatus]   = useState(null)
  const [connected, setConn]  = useState(false)
  const [polling, setPolling] = useState(false)

  useEffect(() => {
    let interval
    if (connected && polling) {
      interval = setInterval(() => {
        api.esp32Health().then(s => setStatus(s)).catch(() => setStatus({ connected: false }))
      }, 3000)
    }
    return () => clearInterval(interval)
  }, [connected, polling])

  async function connect() {
    try {
      const s = await api.esp32Connect(ip, port)
      setStatus(s)
      setConn(s.connected !== false)
      setPolling(s.connected !== false)
    } catch (e) {
      setStatus({ connected: false, error: e.message })
    }
  }

  async function disconnect() {
    await api.esp32Disconnect().catch(() => {})
    setConn(false)
    setPolling(false)
    setStatus(null)
  }

  const heap = status?.free_heap_kb
  const heapPct = heap ? Math.min(100, (heap / 520) * 100) : 0

  return (
    <div className="page-body stack">
      {/* Connection */}
      <div className="card">
        <div className="card-title">ESP32 Connection</div>
        <div className="row" style={{ gap: 10 }}>
          <div className="form-group" style={{ flex: 1 }}>
            <label className="form-label">IP Address</label>
            <input className="form-input" placeholder="192.168.x.x" value={ip}
              onChange={e => setIp(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && connect()} />
          </div>
          <div className="form-group" style={{ width: 90 }}>
            <label className="form-label">Port</label>
            <input className="form-input" type="number" value={port} onChange={e => setPort(parseInt(e.target.value))} />
          </div>
          <div style={{ alignSelf: 'flex-end' }}>
            {connected
              ? <button className="btn btn-danger btn-sm" onClick={disconnect}>Disconnect</button>
              : <button className="btn btn-primary btn-sm" onClick={connect} disabled={!ip}>Connect</button>
            }
          </div>
        </div>

        {status && (
          <div style={{ marginTop: 12 }}>
            {status.connected === false
              ? <span className="badge badge-red">⚠ {status.error || 'Disconnected'}</span>
              : <span className="badge badge-green"><span className="dot dot-green dot-pulse" /> Connected</span>
            }
          </div>
        )}
      </div>

      {/* Board stats */}
      {connected && status?.connected !== false && (
        <div className="grid-4">
          <div className="stat-card">
            <div className="stat-label">Free Heap</div>
            <div className="stat-value" style={{ color: heap < 100 ? 'var(--red)' : 'var(--green)' }}>
              {heap ?? '—'} <span style={{ fontSize: 14, fontWeight: 400 }}>KB</span>
            </div>
            <div style={{ marginTop: 8 }}>
              <div className="progress-bar-track">
                <div className="progress-bar-fill" style={{ width: `${heapPct}%`, background: heap < 100 ? 'var(--red)' : 'var(--green)' }} />
              </div>
              <div className="stat-sub" style={{ marginTop: 4 }}>of 520 KB SRAM</div>
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Uptime</div>
            <div className="stat-value" style={{ fontSize: 20 }}>{status.uptime_s ?? '—'}<span style={{ fontSize: 13, fontWeight: 400 }}> s</span></div>
            <div className="stat-sub">since boot</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Ping</div>
            <div className="stat-value" style={{ fontSize: 20 }}>{status.ping_ms ?? '—'}<span style={{ fontSize: 13, fontWeight: 400 }}> ms</span></div>
            <div className="stat-sub">Wi-Fi round trip</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Status</div>
            <div className="stat-value" style={{ fontSize: 14, marginTop: 8 }}>
              <span className="badge badge-green">● Online</span>
            </div>
            <div className="stat-sub">target: native inference</div>
          </div>
        </div>
      )}

      {connected && status?.connected !== false && (
        <div className="card stack">
          <div className="card-title">Hardware telemetry</div>
          <div style={{ color: 'var(--text-muted)', fontSize: 13 }}>The board is connected and reporting live heap, uptime, and Wi-Fi round-trip data.</div>
        </div>
      )}

      {!connected && (
        <div className="card" style={{ textAlign: 'center', padding: '48px', color: 'var(--text-muted)' }}>
          Enter the ESP32 IP address and click <strong>Connect</strong> to start.
          <div style={{ marginTop: 8, fontSize: 12 }}>
            Find the IP in serial monitor output after flashing the firmware.
          </div>
        </div>
      )}
    </div>
  )
}
