import ResultCard from './ResultCard'

const MODEL_NAMES = {
  logreg: 'Logistic Regression',
  rf: 'Random Forest',
  mlp: 'MLP Neural Network',
}

function modelTitle(prefix, result) {
  const name = MODEL_NAMES[result.model_name] || result.model_label || result.model_name
  return `${prefix} — ${name}`
}

function formatScore(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) return '—'
  return `${Math.round(value * 100)} / 100`
}

function formatMetric(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) return '—'
  return value.toFixed(3)
}

export default function ComparisonResult({ result }) {
  if (!result) return null
  const ml = result.ml || result
  const dl = result.dl
  if (!dl) return <ResultCard result={ml} />

  const comparison = result.comparison || {}
  return (
    <div className="comparison-result" data-testid="comparison-result">
      <div className="comparison-grid">
        <ResultCard title={modelTitle('Machine Learning', ml)} result={ml} />
        <ResultCard
          title={modelTitle('Deep Learning', dl)}
          result={dl}
        />
      </div>
      <div className="card comparison-summary" data-testid="comparison-summary">
        <h3>ML vs DL comparison</h3>
        <p>ML: {ml.risk_level}, {formatScore(ml.risk_score)}</p>
        <p>DL: {dl.risk_level}, {formatScore(dl.risk_score)}</p>
        <p>Score difference: {formatScore(Math.abs(comparison.score_difference || 0))}</p>
        {comparison.interpretation && <p>{comparison.interpretation}</p>}
        <p className="small">
          Test ROC-AUC: ML {formatMetric(comparison.ml_test_roc_auc)}; DL{' '}
          {formatMetric(comparison.dl_test_roc_auc)}
        </p>
        {comparison.note && <p className="small">{comparison.note}</p>}
        {(comparison.disclaimer || result.disclaimer) && (
          <p className="small">{comparison.disclaimer || result.disclaimer}</p>
        )}
      </div>
    </div>
  )
}
