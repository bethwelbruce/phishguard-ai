import { useState } from 'react'
import { predictURL, explainURL } from '../api.js'

export default function Scanner({ onScan }) {
  const [url, setUrl] = useState('')
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState(null)
  const [explain, setExplain] = useState(null)
  const [error, setError] = useState('')

  const scan = async () => {
    const target = url.trim()
    if (!target) return
    setBusy(true)
    setError('')
    setResult(null)
    setExplain(null)
    try {
      const pred = await predictURL(target)
      setResult(pred)
      onScan({
        url: target,
        prediction: pred.prediction,
        confidence: pred.confidence,
        at: new Date().toISOString(),
      })
      // Explanations are slower (LIME sampling) - fetch after the verdict
      const exp = await explainURL(target)
      setExplain(exp)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  const isLegit = result?.prediction === 'legitimate'

  return (
    <>
      <section className="panel">
        <h3>Scan a URL</h3>
        <div className="scan-row">
          <input
            className="scan-input"
            placeholder="https://example.com/login"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && !busy && scan()}
            aria-label="URL to scan"
          />
          <button className="scan-btn" onClick={scan} disabled={busy}>
            {busy ? 'Scanning…' : 'Scan URL'}
          </button>
        </div>
        {error && <div className="error-box">{error}</div>}
      </section>

      {result && (
        <div className={`verdict ${isLegit ? 'legit' : 'phish'}`}>
          <div className="verdict-label">
            {isLegit ? 'Legitimate' : 'Phishing detected'}
          </div>
          <div className="verdict-url mono">{result.url}</div>
          <div className="gauge-track">
            <div
              className="gauge-fill"
              style={{ width: `${result.confidence * 100}%` }}
            />
          </div>
          <div className="gauge-caption">
            Model confidence {(result.confidence * 100).toFixed(1)}% · phishing{' '}
            {(result.probabilities.phishing * 100).toFixed(1)}% / legitimate{' '}
            {(result.probabilities.legitimate * 100).toFixed(1)}%
          </div>
        </div>
      )}

      {result && (
        <div className="xai-grid">
          <section className="panel">
            <h3>SHAP — feature contributions</h3>
            {explain ? (
              <>
                <p className="xai-note">{explain.shap.note}</p>
                {explain.shap.top_features.map((f) => (
                  <div className="xai-item" key={f.feature}>
                    <span className="xai-name mono">{f.feature}</span>
                    <span className={f.shap_value >= 0 ? 'w-pos' : 'w-neg'}>
                      {f.shap_value >= 0 ? '+' : ''}
                      {f.shap_value.toFixed(4)}
                    </span>
                  </div>
                ))}
              </>
            ) : (
              <p className="empty">Computing explanation…</p>
            )}
          </section>

          <section className="panel">
            <h3>LIME — local rules</h3>
            {explain ? (
              <>
                <p className="xai-note">{explain.lime.note}</p>
                {explain.lime.top_rules.map((r) => (
                  <div className="xai-item" key={r.rule}>
                    <span className="xai-name mono">{r.rule}</span>
                    <span className={r.weight >= 0 ? 'w-pos' : 'w-neg'}>
                      {r.weight >= 0 ? '+' : ''}
                      {r.weight.toFixed(4)}
                    </span>
                  </div>
                ))}
              </>
            ) : (
              <p className="empty">Computing explanation…</p>
            )}
          </section>
        </div>
      )}
    </>
  )
}
