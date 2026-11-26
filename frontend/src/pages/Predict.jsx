import { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { API_URL, clearToken, getToken } from '../api'
import { FieldRenderer } from '../components/FieldInput'
import ComparisonResult from '../components/ComparisonResult'
import { bmiFromWeightHeight, cumulativeTobaccoKg } from '../derived'
import {
  DISEASE_NAMES,
  buildFeatures,
  isBmiField,
  isTobaccoField,
  mapServerErrors,
  normalizeSchema,
} from '../sharedFields'
import { fieldCopy, groupedFields, rangeMessage } from '../uiCopy'

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

  function outsideLimits(field, value) {
    if (value === '' || value === null || value === undefined) return false
    const numeric = Number(value)
    if (Number.isNaN(numeric)) return false
    return (
      (field.valid_min != null && numeric < field.valid_min) ||
      (field.valid_max != null && numeric > field.valid_max)
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
        } else if (outsideLimits(field, bmi)) {
          missing[field.name] = rangeMessage(field)
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
        } else if (outsideLimits(field, tobacco)) {
          missing[field.name] = rangeMessage(field)
        } else {
          features[field.name] = tobacco
        }
        continue
      }
      const value = values[field.name]
      if (value === '' || value === null || value === undefined) {
        missing[field.name] = 'This field is required.'
      } else if (outsideLimits(field, value)) {
        missing[field.name] = rangeMessage(field)
      } else {
        Object.assign(
          features,
          buildFeatures([field], () => value),
        )
      }
    }
    if (Object.keys(missing).length > 0) {
      setErrors(missing)
      return
    }
    setErrors({})
    const response = await fetch(`${API_URL}/compare/${disease}`, {
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
  const sections = groupedFields(
    disease,
    schema.fields.filter((field) => field !== familyField),
  )

  return (
    <div className="page">
      <h1>{DISEASE_NAMES[disease] || disease}</h1>
      <form data-testid="predict-form" noValidate onSubmit={handleSubmit}>
        {familyField && (
          <div className="family-box" data-testid="family-history-box">
            <h3>{schema.boxTitle}</h3>
            <div className="form-grid">
              <FieldRenderer
                key={familyField.name}
                field={familyField}
                copy={fieldCopy(disease, familyField.name)}
                values={values}
                errors={errors}
                onValue={update}
              />
            </div>
          </div>
        )}
        {sections.map((section) => (
          <section className="form-section" key={section.group}>
            <h3 className="group-title">{section.group}</h3>
            <div className="form-grid">
              {section.fields.map((field) => (
                <FieldRenderer
                  key={field.name}
                  field={field}
                  copy={fieldCopy(disease, field.name)}
                  values={values}
                  errors={errors}
                  onValue={update}
                />
              ))}
            </div>
          </section>
        ))}
        {formError && <p className="error">{formError}</p>}
        <button type="submit">Predict</button>
      </form>

      {result && <ComparisonResult result={result} />}
    </div>
  )
}
