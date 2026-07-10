import { useEffect, useState } from 'react'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Tooltip,
} from 'chart.js'
import { Bar } from 'react-chartjs-2'
import { getMetrics, getFeatureImportance } from '../api.js'

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip)

export default function Dashboard() {
  const [metrics, setMetrics] = useState(null)
  const [importance, setImportance] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    Promise.all([getMetrics(), getFeatureImportance()])
      .then(([m, fi]) => {
        setMetrics(m)
        setImportance(fi.features)
      })
      .catch((e) => setError(e.message))
  }, [])

  if (error) return <div className="error-box">{error}</div>
  if (!metrics) return <p className="empty">Loading model metrics…</p>

  const top = importance.slice(0, 15)
  const chartData = {
    labels: top.map((f) => f.name),
    datasets: [
      {
        data: top.map((f) => f.importance),
        backgroundColor: '#38bdf8',
        borderRadius: 4,
      },
    ],
  }
  const chartOpts = {
    indexAxis: 'y',
    responsive: true,
    plugins: { legend: { display: false } },
    scales: {
      x: { ticks: { color: '#8ca3c3' }, grid: { color: '#223350' } },
      y: {
        ticks: { color: '#e6eef9', font: { family: 'JetBrains Mono', size: 10 } },
        grid: { display: false },
      },
    },
  }

  return (
    <>
      <section className="panel">
        <h3>Model performance — held-out test set</h3>
        <div className="tiles">
          <div className="tile">
            <div className="v">{(metrics.accuracy * 100).toFixed(2)}%</div>
            <div className="k">Accuracy</div>
          </div>
          <div className="tile">
            <div className="v">{(metrics.f1_score * 100).toFixed(2)}%</div>
            <div className="k">F1-score</div>
          </div>
          <div className="tile">
            <div className="v">{metrics.n_train.toLocaleString()}</div>
            <div className="k">Training URLs</div>
          </div>
          <div className="tile">
            <div className="v">{metrics.n_test.toLocaleString()}</div>
            <div className="k">Test URLs</div>
          </div>
          <div className="tile">
            <div className="v">{metrics.n_features}</div>
            <div className="k">Features (URL-derived)</div>
          </div>
        </div>
        <p className="xai-note" style={{ marginTop: 12 }}>
          {metrics.model}
        </p>
      </section>

      <section className="panel">
        <h3>Feature importance — top 15</h3>
        <Bar data={chartData} options={chartOpts} />
      </section>
    </>
  )
}
