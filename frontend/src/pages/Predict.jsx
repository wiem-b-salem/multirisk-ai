import { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { API_URL, clearToken, getToken } from '../api'
import { GRAMS_PER_CIGARETTE, bmiFromWeightHeight, cumulativeTobaccoKg } from '../derived'
import { labelFor } from '../labels'

const DISEASE_NAMES = {
  cvd: 'Cardiovascular',
  diabetes: 'Type 2 Diabetes',
  alzheimers: "Alzheimer's disease",
  liver: 'Liver disease',
}
const FAMILY_HISTORY = {
  cvd: 'famhist',
  diabetes: 'FamilyHistoryDiabetes',
  alzheimers: 'FamilyHistoryAlzheimers',
  liver: 'GeneticRisk',
}
const DEFAULT_BOX_TITLE = 'Family history (required)'

function normalizeSchema(disease, data) {
  if (Array.isArray(data)) {
    return {
      fields: data,
      familyHistoryField: FAMILY_HISTORY[disease],
      boxTitle: DEFAULT_BOX_TITLE,
      populationNote: '',
    }
  }
  return {
    fields: data.raw_fields,
    familyHistoryField: data.family_history_field,
    boxTitle: data.family_history_box_title || DEFAULT_BOX_TITLE,
    populationNote: data.population_note || '',
  }
}

function fieldLabel(field) {
  return field.label || labelFor(field.name)
}

function isOutsideTrainRange(field, value) {
  if (value === '' || value === null || value === undefined) return false
  if (field.train_min === null || field.train_min === undefined) return false
  const numeric = Number(value)
  return numeric < field.train_min || numeric > field.train_max
}

function isBinary(field) {
  const values = field.allowed_values
  return Array.isArray(values) && values.length === 2 && values[0] === 0 && values[1] === 1
}

function isBmiField(field) {
  return field.name === 'BMI' || field.name === 'obesity'
}

function isTobaccoField(field) {
  return field.name === 'tobacco'
}

function optionLabel(field, value) {
  if (isBinary(field)) {
    if (field.name === 'Gender') return value === 0 ? 'Gender code 0' : 'Gender code 1'
    return value === 0 ? 'No' : 'Yes'
  }
  return String(value)
}

function mapServerErrors(detail) {
  const errors = {}
  if (Array.isArray(detail)) {
    for (const item of detail) {
      const loc = Array.isArray(item.loc) ? item.loc : []
      const name = [...loc]
        .reverse()
        .find((part) => typeof part === 'string' && part !== 'body' && part !== 'features')
      if (name) errors[name] = item.msg || 'Invalid value'
    }
  }
  return errors
}

export default function Predict() {
  const { disease } = useParams()
  const navigate = useNavigate()
  const [schema, setSchema] = useState(null)
  const [values, setValues] = useState({})
  const [errors, setErrors] = useState({})
  const [formError, setFormError] = useState('')
  const [result, setResult] = useState(null)

  useEffect(() => {
    let active = true
    async function load() {
      const response = await fetch(`${API_URL}/schema/${disease}`, {
        headers: { Authorization: `Bearer ${getToken()}` },
      })
      if (!active) return
      if (response.status === 401) {
        clearToken()
        navigate('/login')
        return
      }
      const normalized = normalizeSchema(disease, await response.json())
      const initial = {}
      for (const field of normalized.fields) initial[field.name] = ''
      setSchema(normalized)
      setValues(initial)
    }
    load()
    return () => {
      active = false
    }
  }, [disease, navigate])

  function update(name, value) {
    setValues((current) => ({ ...current, [name]: value }))
  }

  function bmiValue() {
    const weight = values.weight
    const height = values.height
    if (weight === '' || height === '' || weight == null || height == null) return null
    return bmiFromWeightHeight(Number(weight), Number(height))
  }

  function tobaccoValue() {
    const status = values.smokingStatus
    if (status === 'never') return 0
    const cigarettes = values.cigarettesPerDay
    const years = values.yearsSmoked
    if (
      status === '' ||
      status == null ||
      cigarettes === '' ||
      years === '' ||
      cigarettes == null ||
      years == null
    ) {
      return null
    }
    return cumulativeTobaccoKg({
      status,
      cigarettesPerDay: Number(cigarettes),
      yearsSmoked: Number(years),
    })
  }

  function renderControl(field) {
    if (field.allowed_values) {
      return (
        <select
          id={field.name}
          name={field.name}
          value={values[field.name]}
          onChange={(event) => update(field.name, event.target.value)}
        >
          <option value="">Select...</option>
          {field.allowed_values.map((value) => (
            <option key={String(value)} value={String(value)}>
              {optionLabel(field, value)}
            </option>
          ))}
        </select>
      )
    }
    return (
      <input
        id={field.name}
        name={field.name}
        type="number"
        min={field.valid_min}
        max={field.valid_max}
        step={field.type === 'int' ? 1 : 'any'}
        value={values[field.name]}
        onChange={(event) => update(field.name, event.target.value)}
      />
    )
  }

  function renderBmiField(field) {
    const bmi = bmiValue()
    const invalid = bmi !== null && (bmi < field.valid_min || bmi > field.valid_max)
    return (
      <div className="field" key={field.name}>
        <span className="field-title">{fieldLabel(field)}</span>
        <div
          className="input-row"
          style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}
        >
          <label htmlFor="weight">Weight (kg)</label>
          <input
            id="weight"
            name="weight"
            type="number"
            min={25}
            max={300}
            step="any"
            value={values.weight ?? ''}
            onChange={(event) => update('weight', event.target.value)}
          />
          <label htmlFor="height">Height (cm)</label>
          <input
            id="height"
            name="height"
            type="number"
            min={100}
            max={230}
            step="any"
            value={values.height ?? ''}
            onChange={(event) => update('height', event.target.value)}
          />
        </div>
        <span className="helper">Calculated from your weight and height.</span>
        {bmi !== null && <span className="helper">BMI used by the model: {bmi}</span>}
        {field.train_min != null && (
          <span className="helper">
            Typical range in the training data: {field.train_min} to {field.train_max}
          </span>
        )}
        {field.train_min != null && bmi !== null && (bmi < field.train_min || bmi > field.train_max) && (
          <span className="warning" style={{ color: '#b45309' }}>
            Outside the training range: the estimate may be unreliable.
          </span>
        )}
        {invalid && (
          <span className="error">
            BMI must be between {field.valid_min} and {field.valid_max}.
          </span>
        )}
        {errors[field.name] && <span className="error">{errors[field.name]}</span>}
      </div>
    )
  }

  function renderTobaccoField(field) {
    const status = values.smokingStatus ?? ''
    const tobacco = tobaccoValue()
    const showAmount = status !== '' && status !== 'never'
    return (
      <div className="field" key={field.name}>
        <label htmlFor="smokingStatus">{fieldLabel(field)}</label>
        <div
          className="input-row"
          style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}
        >
          <select
            id="smokingStatus"
            name="smokingStatus"
            value={status}
            onChange={(event) => update('smokingStatus', event.target.value)}
          >
            <option value="">Select...</option>
            <option value="never">Never smoked</option>
            <option value="former">Former smoker</option>
            <option value="current">Current smoker</option>
          </select>
        </div>
        {showAmount && (
          <>
            <div
              className="input-row"
              style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}
            >
              <label htmlFor="cigarettesPerDay">
                {status === 'former' ? 'Cigarettes per day while smoking' : 'Cigarettes per day'}
              </label>
              <input
                id="cigarettesPerDay"
                name="cigarettesPerDay"
                type="number"
                min={0}
                max={100}
                step="any"
                value={values.cigarettesPerDay ?? ''}
                onChange={(event) => update('cigarettesPerDay', event.target.value)}
              />
            </div>
            <div
              className="input-row"
              style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}
            >
              <label htmlFor="yearsSmoked">Years smoked</label>
              <input
                id="yearsSmoked"
                name="yearsSmoked"
                type="number"
                min={0}
                max={80}
                step={1}
                value={values.yearsSmoked ?? ''}
                onChange={(event) => update('yearsSmoked', event.target.value)}
              />
            </div>
          </>
        )}
        {tobacco !== null && (
          <span className="helper">
            Estimated lifetime tobacco: {tobacco} kg (assuming about {GRAMS_PER_CIGARETTE} g of
            tobacco per cigarette)
          </span>
        )}
        {field.train_min != null && (
          <span className="helper">
            Typical range in the training data: {field.train_min} to {field.train_max}
          </span>
        )}
        {field.train_min != null && tobacco !== null && (tobacco < field.train_min || tobacco > field.train_max) && (
          <span className="warning" style={{ color: '#b45309' }}>
            Outside the training range: the estimate may be unreliable.
          </span>
        )}
        {errors[field.name] && <span className="error">{errors[field.name]}</span>}
      </div>
    )
  }

  function renderField(field) {
    if (isBmiField(field)) return renderBmiField(field)
    if (isTobaccoField(field)) return renderTobaccoField(field)
    return (
      <div className="field" key={field.name}>
        <label htmlFor={field.name}>{fieldLabel(field)}</label>
        <div
          className="input-row"
          style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}
        >
          {renderControl(field)}
          {field.unit && <span className="unit">{field.unit}</span>}
        </div>
        {field.help && <span className="helper">{field.help}</span>}
        {field.type !== 'category' && !field.allowed_values && field.train_min != null && (
          <span className="helper">
            Typical range in the training data: {field.train_min} to {field.train_max}
          </span>
        )}
        {isOutsideTrainRange(field, values[field.name]) && (
          <span className="warning" style={{ color: '#b45309' }}>
            Outside the training range: the estimate may be unreliable.
          </span>
        )}
        {errors[field.name] && <span className="error">{errors[field.name]}</span>}
      </div>
    )
  }

  async function handleSubmit(event) {
    event.preventDefault()
    setFormError('')
    const missing = {}
    const features = {}
    for (const field of schema.fields) {
      if (isBmiField(field)) {
        const bmi = bmiValue()
        if (bmi === null) {
          missing[field.name] = 'This field is required.'
        } else if (bmi < field.valid_min || bmi > field.valid_max) {
          missing[field.name] = `BMI must be between ${field.valid_min} and ${field.valid_max}.`
        } else {
          features[field.name] = bmi
        }
        continue
      }
      if (isTobaccoField(field)) {
        const status = values.smokingStatus
        const tobacco = tobaccoValue()
        if (status === '' || status == null || tobacco === null) {
          missing[field.name] = 'This field is required.'
        } else {
          features[field.name] = tobacco
        }
        continue
      }
      const value = values[field.name]
      if (value === '' || value === null || value === undefined) {
        missing[field.name] = 'This field is required.'
      } else if (field.type === 'int') {
        features[field.name] = Number.parseInt(value, 10)
      } else if (field.type === 'float') {
        features[field.name] = Number.parseFloat(value)
      } else {
        features[field.name] = value
      }
    }
    if (Object.keys(missing).length > 0) {
      setErrors(missing)
      return
    }
    setErrors({})
    const response = await fetch(`${API_URL}/predict/${disease}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${getToken()}`,
      },
      body: JSON.stringify({ features }),
    })
    if (response.status === 401) {
      clearToken()
      navigate('/login')
      return
    }
    if (response.status === 422) {
      const data = await response.json()
      setErrors(mapServerErrors(data.detail))
      return
    }
    if (response.status === 503) {
      setFormError('Prediction service unavailable')
      return
    }
    if (!response.ok) {
      setFormError('Something went wrong')
      return
    }
    setResult(await response.json())
  }

  if (!schema) return <div className="page">Loading...</div>

  const familyField = schema.fields.find((field) => field.name === schema.familyHistoryField)

  return (
    <div className="page">
      <h1>{DISEASE_NAMES[disease] || disease}</h1>
      {schema.populationNote && (
        <div
          className="info-box"
          style={{
            border: '1px solid #93c5fd',
            background: '#eff6ff',
            padding: '0.6rem',
            borderRadius: '8px',
            marginBottom: '1rem',
          }}
        >
          {schema.populationNote}
        </div>
      )}
      <form data-testid="predict-form" onSubmit={handleSubmit}>
        {familyField && (
          <div className="family-box" data-testid="family-history-box">
            <h3>{schema.boxTitle}</h3>
            {renderField(familyField)}
          </div>
        )}
        {schema.fields.filter((field) => field !== familyField).map(renderField)}
        {formError && <p className="error">{formError}</p>}
        <button type="submit">Predict</button>
      </form>

      {result && (
        <div className="card result">
          <h2>Result</h2>
          <span className={`badge badge-${result.risk_level}`}>{result.risk_level}</span>
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
            <p className="warning" style={{ color: '#b45309' }}>
              {result.range_warning}{' '}
              {Array.isArray(result.out_of_training_range) &&
                `Fields: ${result.out_of_training_range
                  .map((item) => item.field)
                  .join(', ')}`}
            </p>
          )}
          {result.disclaimer && <p className="small">{result.disclaimer}</p>}
        </div>
      )}
    </div>
  )
}
