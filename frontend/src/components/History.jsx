export default function History({ items, onClear }) {
  if (!items.length)
    return (
      <section className="panel">
        <h3>Scan history</h3>
        <p className="empty">
          No scans yet. Run a URL through the Scanner tab and it will be
          logged here.
        </p>
      </section>
    )

  return (
    <section className="panel">
      <h3>
        Scan history ({items.length}){' '}
        <button className="link-btn" onClick={onClear}>
          Clear
        </button>
      </h3>
      <table className="hist">
        <thead>
          <tr>
            <th>Time</th>
            <th>URL</th>
            <th>Verdict</th>
            <th>Confidence</th>
          </tr>
        </thead>
        <tbody>
          {items.map((it, i) => (
            <tr key={`${it.at}-${i}`}>
              <td>{new Date(it.at).toLocaleString()}</td>
              <td className="mono">{it.url}</td>
              <td>
                <span
                  className={`badge ${
                    it.prediction === 'legitimate' ? 'legit' : 'phish'
                  }`}
                >
                  {it.prediction}
                </span>
              </td>
              <td>{(it.confidence * 100).toFixed(1)}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  )
}
