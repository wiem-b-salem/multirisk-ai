import { labelFor } from '../labels'
import { rangeMessage } from '../uiCopy'

const warningStyle = { color: '#b45309' }

export default function ResultCard({ title = 'Result', state = 'success', result, message, errors, violation }) {
  if (state === 'error') {
    return (
      <div className="card result" data-testid="result-card">
        <h2>{title}</h2>
        {violation && (
          <>
            <p className="error">Not assessed</p>
            <p>{rangeMessage({ valid_min: violation.min, valid_max: violation.max })}</p>
          </>
        )}
        {message && <p className="error">{message}</p>}
        {errors &&
          Object.entries(errors).map(([field, text]) => (
            <p className="error" key={field}>
              {field}: {text}
            </p>
          ))}
      </div>
    )
  }

  return (
    <div className="card result" data-testid="result-card">
      <h2>{title}</h2>
      <span className={`badge badge-${result.risk_level}`}>{result.risk_level}</span>
      <p className="small">Risk category: {result.risk_level}</p>
      <p>Model score: {Math.round(result.risk_score * 100)} / 100</p>
      {result.score_note && <p className="small">{result.score_note}</p>}
      {result.top_factors && (
        <div>
          <h3>Top factors</h3>
          <ul>
            {result.top_factors.map((factor) => (
              <li key={factor.feature}>
                {labelFor(factor.feature)}:{' '}
                {factor.effect === 'above_average' ? 'above average' : 'below average'}
              </li>
            ))}
          </ul>
        </div>
      )}
      {result.factors_note && <p className="small">{result.factors_note}</p>}
      {result.model_note && <p className="small">{result.model_note}</p>}
      {result.range_warning && (
        <p className="warning" style={warningStyle}>
          {result.range_warning}{' '}
          {Array.isArray(result.out_of_training_range) &&
            `Fields: ${result.out_of_training_range.map((item) => item.field).join(', ')}`}
        </p>
      )}
      {result.disclaimer && <p className="small">{result.disclaimer}</p>}
    </div>
  )
}
