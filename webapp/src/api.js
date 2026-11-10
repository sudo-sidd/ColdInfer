// api.js — Centralized API client for the ColdInfer backend

const BASE = '/api'

async function req(path, options = {}) {
  const res = await fetch(BASE + path, options)
  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || `HTTP ${res.status}`)
  }
  return res.json()
}

export const api = {
  health:  () => req('/health'),

  simulationInfer: (model = 'mnist', sramCapKb = 520, inputVariant = 0, inputData = null) =>
    req('/simulation/infer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, sram_cap_kb: sramCapKb, input_variant: inputVariant, input_data: inputData }),
    }),
  yoloDetect: (file, confidence = 0.25, target = 'simulated-esp32', arenaKb = 520) => {
    const body = new FormData()
    body.append('image', file)
    return req(`/yolo/detect?confidence=${confidence}&target=${target}&arena_kb=${arenaKb}`, { method: 'POST', body })
  },
  yoloCompare: (file, confidence = 0.25) => {
    const body = new FormData()
    body.append('image', file)
    return req(`/yolo/compare?confidence=${confidence}`, { method: 'POST', body })
  },
  yoloBenchmark: (confidence = 0.25) =>
    req(`/yolo/benchmark?confidence=${confidence}`, { method: 'POST' }),

  // Benchmarks
  benchmarks: () => req('/benchmarks'),

  // Training — returns EventSource URL
  trainUrl: (params) => {
    const body = JSON.stringify(params)
    return { url: BASE + '/train', body }
  },

  // Models
  models: () => req('/models'),

  // Quantization summary
  quantizeSummary: () => req('/quantize/summary'),

  // Build
  build: (params) =>
    req('/build', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    }),

  // ESP32
  esp32Connect: (ip, port = 80) =>
    req('/esp32/connect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ip, port }),
    }),
  esp32Health: () => req('/esp32/health'),
  esp32Disconnect: () => req('/esp32/connect', { method: 'DELETE' }),
}

// SSE helper for streaming jobs (training, benchmarks, quantize)
export function createSSE(url, body, { onLine, onDone, onError } = {}) {
  const ctrl = new AbortController()

  ;(async () => {
    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body,
        signal: ctrl.signal,
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const reader = res.body.getReader()
      const dec = new TextDecoder()
      let buf = ''

      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        buf += dec.decode(value, { stream: true })
        const lines = buf.split('\n')
        buf = lines.pop()
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const parsed = JSON.parse(line.slice(6))
            if (parsed.event === 'done') {
              if (parsed.exit_code && parsed.exit_code !== 0) {
                onError?.(new Error(`Process exited with code ${parsed.exit_code}`))
              } else {
                onDone?.(parsed)
              }
            } else if (parsed.event === 'log') {
              onLine?.(parsed.line)
            }
          } catch {}
        }
      }
    } catch (e) {
      if (e.name !== 'AbortError') onError?.(e)
    }
  })()

  return () => ctrl.abort()
}
