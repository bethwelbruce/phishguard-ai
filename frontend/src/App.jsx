import { useState } from 'react'
import Scanner from './components/Scanner.jsx'
import Dashboard from './components/Dashboard.jsx'
import History from './components/History.jsx'

const TABS = ['Scanner', 'Dashboard', 'History']

export default function App() {
  const [tab, setTab] = useState('Scanner')
  // History is lifted here so a scan on the Scanner tab is
  // immediately visible on the History tab.
  const [history, setHistory] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem('pg_history') || '[]')
    } catch {
      return []
    }
  })

  const addToHistory = (entry) => {
    setHistory((prev) => {
      const next = [entry, ...prev].slice(0, 50)
      try {
        localStorage.setItem('pg_history', JSON.stringify(next))
      } catch {
        /* storage unavailable - history stays in memory only */
      }
      return next
    })
  }

  const clearHistory = () => {
    setHistory([])
    try {
      localStorage.removeItem('pg_history')
    } catch {
      /* ignore */
    }
  }

  return (
    <>
      <header className="header">
        <h1 className="brand">
          Phish<span className="accent">Guard</span> AI
        </h1>
        <span className="tagline">
          Decision Tree URL threat scanner with SHAP + LIME explanations
        </span>
      </header>

      <nav className="tabs" aria-label="Sections">
        {TABS.map((t) => (
          <button
            key={t}
            className={`tab ${tab === t ? 'active' : ''}`}
            onClick={() => setTab(t)}
          >
            {t}
          </button>
        ))}
      </nav>

      {tab === 'Scanner' && <Scanner onScan={addToHistory} />}
      {tab === 'Dashboard' && <Dashboard />}
      {tab === 'History' && (
        <History items={history} onClear={clearHistory} />
      )}
    </>
  )
}
