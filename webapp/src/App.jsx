import { useState, useEffect } from 'react'
import SimulationLab from './pages/SimulationLab'
import YoloLab from './pages/YoloLab'
import Benchmarks from './pages/Benchmarks'
import Training from './pages/Training'
import ESP32Panel from './pages/ESP32Panel'
import PerformanceHistory from './pages/PerformanceHistory'
import { api } from './api'
import { Activity, BarChart3, Cpu, Eye, FlaskConical, Gauge, LineChart, Snowflake, Zap } from 'lucide-react'

const NAV_ITEMS = [
  { id: 'simulation', label: 'Simulation Lab', icon: FlaskConical, sub: 'Live MNIST & keyword inference' },
  { id: 'yolo', label: 'YOLO Detection', icon: Eye, sub: 'Embedded-style object detection' },
  { id: 'benchmarks', label: 'Benchmarks', icon: BarChart3, sub: 'Runtime comparison charts' },
  { id: 'training', label: 'Training Studio', icon: Zap, sub: 'Train MNIST and keyword models' },
  { id: 'esp32', label: 'ESP32 Device Panel', icon: Cpu, sub: 'Wi-Fi telemetry and hardware inference' },
  { id: 'history', label: 'Performance History', icon: LineChart, sub: 'Historical run tracker and CSV logs' },
]

export default function App() {
  const [activeTab, setActiveTab] = useState('simulation')
  const [backendStatus, setBackendStatus] = useState({ online: false, checking: true })
  const [esp32Status, setEsp32Status] = useState({ connected: false })

  useEffect(() => {
    // Check backend health
    api.health()
      .then(res => setBackendStatus({ online: true, checking: false, data: res }))
      .catch(() => setBackendStatus({ online: false, checking: false }))

    // Check ESP32 health
    api.esp32Health()
      .then(res => setEsp32Status(res))
      .catch(() => setEsp32Status({ connected: false }))
  }, [])

  const currentNav = NAV_ITEMS.find(item => item.id === activeTab) || NAV_ITEMS[0]

  return (
    <div className="app-shell">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-logo">
          <div className="logo-icon"><Snowflake size={18} strokeWidth={2.2} /></div>
          <div>
            <div className="logo-text">ColdInfer</div>
            <div className="logo-version">v0.2.0-esp32</div>
          </div>
        </div>

        <div className="nav-section-label">Navigation</div>

        <nav style={{ display: 'flex', flexDirection: 'column', gap: 2, flex: 1 }}>
          {NAV_ITEMS.map(item => (
            <button
              key={item.id}
              className={`nav-item ${activeTab === item.id ? 'active' : ''}`}
              onClick={() => setActiveTab(item.id)}
            >
              <span className="nav-icon"><item.icon size={17} strokeWidth={2} /></span>
              <div style={{ display: 'flex', flexDirection: 'column', textAlign: 'left' }}>
                <span style={{ fontSize: '13.5px', lineHeight: 1.2 }}>{item.label}</span>
              </div>
            </button>
          ))}
        </nav>

        {/* Sidebar Footer Status */}
        <div style={{ padding: '16px 20px', borderTop: '1px solid var(--border)', fontSize: '11px', display: 'flex', flexDirection: 'column', gap: 8 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span style={{ color: 'var(--text-muted)' }}>Backend:</span>
            <span style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 5,
              color: backendStatus.online ? 'var(--green)' : 'var(--red)',
              fontWeight: 600
            }}>
              <span style={{
                width: 6,
                height: 6,
                borderRadius: '50%',
                background: backendStatus.online ? 'var(--green)' : 'var(--red)'
              }} />
              {backendStatus.checking ? 'Checking...' : backendStatus.online ? 'Online' : 'Offline'}
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span style={{ color: 'var(--text-muted)' }}>ESP32 Target:</span>
            <span style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 5,
              color: esp32Status.connected ? 'var(--teal)' : 'var(--text-dim)',
              fontWeight: 600
            }}>
              <span style={{
                width: 6,
                height: 6,
                borderRadius: '50%',
                background: esp32Status.connected ? 'var(--teal)' : 'var(--text-dim)'
              }} />
              {esp32Status.connected ? 'Connected' : 'Disconnected'}
            </span>
          </div>

          <div style={{ color: 'var(--text-dim)', fontSize: '10px', marginTop: 4 }}>
            SRAM Constraint: 520 KB
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="main-content">
        <header className="page-header">
          <div>
            <div className="page-title">{currentNav.label}</div>
            <div className="page-subtitle">{currentNav.sub}</div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <span className="badge" style={{
              background: 'var(--accent-glow)',
              color: 'var(--accent)',
              border: '1px solid var(--accent-dim)',
              padding: '4px 8px',
              borderRadius: '4px',
              fontSize: '11px',
              fontFamily: 'var(--font-mono)'
            }}>
              Target: Xtensa LX6 / Sim-512K
            </span>
          </div>
        </header>

        {activeTab === 'simulation' && <SimulationLab />}
        {activeTab === 'yolo' && <YoloLab />}
        {activeTab === 'benchmarks' && <Benchmarks />}
        {activeTab === 'training' && <Training />}
        {activeTab === 'esp32' && <ESP32Panel />}
        {activeTab === 'history' && <PerformanceHistory />}
      </main>
    </div>
  )
}
