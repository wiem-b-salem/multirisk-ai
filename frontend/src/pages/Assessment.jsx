import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { API_URL, clearToken, getToken } from '../api'
import { FieldRenderer } from '../components/FieldInput'
import ComparisonResult from '../components/ComparisonResult'
import ResultCard from '../components/ResultCard'
import { bmiFromWeightHeight, cumulativeTobaccoKg } from '../derived'
import {
  DISEASES,
  DISEASE_NAMES,
  SHARED_FIELDS,
  convertValue,
  mapServerErrors,
  normalizeSchema,
  outsideTrainingDiseases,
  sharedKeyForField,
  validLimitsViolation,
  widestValidRange,
} from '../sharedFields'
import { fieldCopy, groupedFields, outsideRangeNotice, sharedCopy } from '../uiCopy'

const SHARED_FIELD_DEFINITIONS = {
  age: { name: 'shared-age', type: 'int' },
  bmi: { name: 'BMI', type: 'float' },
  systolic_bp: { name: 'shared-systolic_bp', type: 'int' },
  hypertension: { name: 'shared-hypertension', type: 'int' },
  diabetes_diagnosis: { name: 'shared-diabetes_diagnosis', type: 'int' },
}

const INPUT_NAMES = {
  age: 'shared-age',
  bmi: 'BMI',
  systolic_bp: 'shared-systolic_bp',
  hypertension: 'shared-hypertension',
  diabetes_diagnosis: 'shared-diabetes_diagnosis',
}

const INITIAL_SHARED = {
  age: '',
  systolic_bp: '',
  hypertension: '',
  diabetes_diagnosis: '',
  family: '',
  weight: '',
  height: '',
  smokingStatus: '',
  cigarettesPerDay: '',
  yearsSmoked: '',
}

const INITIAL_SELECTION = { cvd: false, diabetes: false, alzheimers: false, liver: false }

export default function Assessment() {
  const navigate = useNavigate()
  const [schemas, setSchemas] = useState({})
  const [selected, setSelected] = useState(INITIAL_SELECTION)
  const [shared, setShared] = useState(INITIAL_SHARED)
  const [specific, setSpecific] = useState({})
  const [errors, setErrors] = useState({})
  const [formError, setFormError] = useState('')
  const [results, setResults] = useState({})

  useEffect(() => {
    let active = true
    async function load() {
      const entries = await Promise.allSettled(
        DISEASES.map(async (disease) => {
          const response = await fetch(`${API_URL}/schema/${disease}`, {
            headers: { Authorization: `Bearer ${getToken()}` },
          })
          if (response.status === 401) {
            clearToken()
            navigate('/login')
            return [disease, null]
          }
          if (!response.ok) return [disease, null]
          return [disease, normalizeSchema(disease, await response.json())]
        }),
      )
      if (!active) return
      const loaded = {}
      for (const entry of entries) {
        if (entry.status === 'fulfilled' && entry.value[1]) loaded[entry.value[0]] = entry.value[1]
      }
      setSchemas(loaded)
    }
    load()
    return () => {
      active = false
    }
  }, [navigate])

  const selectedDiseases = DISEASES.filter((disease) => selected[disease])
  const firstDisease = selectedDiseases[0] || 'cvd'
  const labelDiseases = selectedDiseases.length > 0 ? selectedDiseases : DISEASES
  const ready = DISEASES.every((disease) => schemas[disease])

  function updateShared(name, value) {
    setShared((current) => ({ ...current, [name]: value }))
  }

  function updateSpecific(disease, name, value) {
    setSpecific((current) => ({ ...current, [`${disease}:${name}`]: value }))
  }

  function toggleDisease(disease) {
    setSelected((current) => {
      const next = { ...current, [disease]: !current[disease] }
      if (!DISEASES.some((item) => next[item])) return current
      setResults({})
      return next
    })
  }

  function bmiValue() {
    if (shared.weight === '' || shared.height === '') return null
    return bmiFromWeightHeight(Number(shared.weight), Number(shared.height))
  }

  function tobaccoValue() {
    if (shared.smokingStatus === 'never') return 0
    if (
      shared.smokingStatus === '' ||
      shared.cigarettesPerDay === '' ||
      shared.yearsSmoked === ''
    ) {
      return null
    }
    return cumulativeTobaccoKg({
      status: shared.smokingStatus,
      cigarettesPerDay: Number(shared.cigarettesPerDay),
      yearsSmoked: Number(shared.yearsSmoked),
    })
  }

  function errorKey(disease, field) {
    const sharedKey = sharedKeyForField(disease, field.name)
    if (sharedKey) return `shared:${sharedKey}`
    if (disease === firstDisease && field.name === schemas[disease].familyHistoryField) {
      return 'shared:family'
    }
    return `${disease}:${field.name}`
  }

  function rawValue(disease, field) {
    const sharedKey = sharedKeyForField(disease, field.name)
    if (sharedKey === 'bmi') return bmiValue()
    if (sharedKey) return shared[sharedKey]
    if (disease === firstDisease && field.name === schemas[disease].familyHistoryField) {
      return shared.family
    }
    if (field.name === 'tobacco') return tobaccoValue()
    return specific[`${disease}:${field.name}`] ?? ''
  }

  function prepareDisease(disease) {
    const missing = {}
    const features = {}
    let violation = null
    for (const field of schemas[disease].fields) {
      const value = rawValue(disease, field)
      const key = errorKey(disease, field)
      if (value === '' || value === null || value === undefined) {
        missing[key] = 'This field is required.'
        continue
      }
      if (!violation) {
        violation = validLimitsViolation(field, value)
        if (violation) violation.disease = disease
      }
      features[field.name] = convertValue(field, value)
    }
    return { features, missing, violation }
  }

  function diseaseValues(disease) {
    const values = {}
    for (const field of schemas[disease].fields) {
      if (sharedKeyForField(disease, field.name)) continue
      values[field.name] = specific[`${disease}:${field.name}`] ?? ''
    }
    values.smokingStatus = shared.smokingStatus
    values.cigarettesPerDay = shared.cigarettesPerDay
    values.yearsSmoked = shared.yearsSmoked
    return values
  }

  function diseaseErrors(disease) {
    const fieldErrors = {}
    for (const field of schemas[disease].fields) {
      const message = errors[`${disease}:${field.name}`]
      if (message) fieldErrors[field.name] = message
    }
    return fieldErrors
  }

  function sharedField(key) {
    const definition = SHARED_FIELD_DEFINITIONS[key]
    const limits = widestValidRange(key, labelDiseases, schemas)
    return {
      ...definition,
      name: INPUT_NAMES[key],
      valid_min: limits.min,
      valid_max: limits.max,
      train_min: null,
      train_max: null,
    }
  }

  function sharedWarning(key, value) {
    const names = outsideTrainingDiseases(key, value, labelDiseases, schemas)
    if (names.length === 0) return ''
    return outsideRangeNotice(names)
  }

  function diseaseWarning(disease, field, value) {
    if (value === '' || value === null || value === undefined) return ''
    if (field.train_min == null || field.train_max == null) return ''
    const numeric = Number(value)
    if (Number.isNaN(numeric)) return ''
    if (numeric < field.train_min || numeric > field.train_max) {
      return outsideRangeNotice([DISEASE_NAMES[disease]])
    }
    return ''
  }

  function sharedValues() {
    return {
      'shared-age': shared.age,
      BMI: bmiValue() ?? '',
      'shared-systolic_bp': shared.systolic_bp,
      'shared-hypertension': shared.hypertension,
      'shared-diabetes_diagnosis': shared.diabetes_diagnosis,
      weight: shared.weight,
      height: shared.height,
    }
  }

  function sharedFieldErrors(key) {
    const message = errors[`shared:${key}`]
    return message ? { [INPUT_NAMES[key]]: message } : {}
  }

  function sharedValueFor(key) {
    if (key === 'bmi') return bmiValue()
    return shared[key]
  }

  async function handleSubmit(event) {
    event.preventDefault()
    setFormError('')
    if (selectedDiseases.length === 0) {
      setFormError('Select at least one disease.')
      return
    }
    const prepared = {}
    const missing = {}
    for (const disease of selectedDiseases) {
      prepared[disease] = prepareDisease(disease)
      Object.assign(missing, prepared[disease].missing)
    }
    if (Object.keys(missing).length > 0) {
      setErrors(missing)
      return
    }
    setErrors({})
    setResults({})
    const requests = selectedDiseases.filter((disease) => !prepared[disease].violation)
    const outcomes = await Promise.allSettled(
      requests.map(async (disease) => {
        try {
          const response = await fetch(`${API_URL}/compare/${disease}`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              Authorization: `Bearer ${getToken()}`,
            },
            body: JSON.stringify({ features: prepared[disease].features }),
          })
          if (response.status === 401) {
            clearToken()
            navigate('/login')
            return { disease, state: 'error', kind: 'unauthorized' }
          }
          if (response.status === 422) {
            const data = await response.json()
            const mapped = {}
            for (const [name, message] of Object.entries(mapServerErrors(data.detail))) {
              const field = schemas[disease].fields.find((item) => item.name === name)
              mapped[field ? errorKey(disease, field) : name] = message
            }
            return { disease, state: 'error', kind: 'validation', errors: mapped }
          }
          if (response.status === 503) {
            return { disease, state: 'error', kind: 'service', message: 'Prediction service unavailable' }
          }
          if (!response.ok) {
            return { disease, state: 'error', kind: 'service', message: 'Something went wrong' }
          }
          return { disease, state: 'success', result: await response.json() }
        } catch {
          return { disease, state: 'error', kind: 'service', message: 'Prediction service unavailable' }
        }
      }),
    )
    const nextResults = {}
    for (const disease of selectedDiseases) {
      const violation = prepared[disease].violation
      if (violation) nextResults[disease] = { state: 'error', kind: 'violation', violation }
    }
    outcomes.forEach((outcome, index) => {
      if (outcome.status === 'rejected') {
        const disease = requests[index]
        nextResults[disease] = {
          state: 'error',
          kind: 'service',
          message: 'Prediction service unavailable',
        }
        return
      }
      const item = outcome.value
      if (item.kind !== 'unauthorized') nextResults[item.disease] = item
    })
    setResults(nextResults)
  }

  function diseaseFields(disease) {
    return schemas[disease].fields.filter((field) => {
      if (sharedKeyForField(disease, field.name)) return false
      if (disease === firstDisease && field.name === schemas[disease].familyHistoryField) return false
      return true
    })
  }

  if (!ready) return <div className="page">Loading...</div>

  const familyField = schemas[firstDisease].fields.find(
    (field) => field.name === schemas[firstDisease].familyHistoryField,
  )
  const familyValues = { [familyField.name]: shared.family }
  const familyErrors = errors['shared:family'] ? { [familyField.name]: errors['shared:family'] } : {}

  return (
    <div className="page">
      <h1>Full assessment (all diseases)</h1>
      <form className="card" data-testid="assessment-form" noValidate onSubmit={handleSubmit}>
        <fieldset data-testid="disease-selection">
          <legend>Diseases to assess</legend>
          {DISEASES.map((disease) => (
            <label key={disease}>
              <input
                type="checkbox"
                name={disease}
                checked={selected[disease]}
                onChange={() => toggleDisease(disease)}
              />
              {DISEASE_NAMES[disease]}
            </label>
          ))}
        </fieldset>
        {formError && <p className="error">{formError}</p>}

        {familyField && (
          <div className="family-box" data-testid="family-history-box">
            <h3>{schemas[firstDisease].boxTitle}</h3>
            <div className="form-grid">
              <FieldRenderer
                field={familyField}
                copy={fieldCopy(firstDisease, familyField.name)}
                values={familyValues}
                errors={familyErrors}
                onValue={(name, value) => updateShared('family', value)}
              />
            </div>
          </div>
        )}

        <div className="card">
          <h2>Common information</h2>
          <div className="form-grid">
            {SHARED_FIELDS.map((key) => {
              const value = sharedValueFor(key)
              return (
                <FieldRenderer
                  key={key}
                  field={sharedField(key)}
                  copy={sharedCopy(key)}
                  values={sharedValues()}
                  errors={sharedFieldErrors(key)}
                  onValue={(name, nextValue) =>
                    updateShared(name === 'weight' || name === 'height' ? name : key, nextValue)
                  }
                  warningText={sharedWarning(key, value)}
                />
              )
            })}
          </div>
        </div>

        {selectedDiseases.map((disease) => (
          <div className="card" key={disease} data-testid={`disease-fields-${disease}`}>
            <h2>{DISEASE_NAMES[disease]}</h2>
            {groupedFields(disease, diseaseFields(disease)).map((section) => (
              <section className="form-section" key={`${disease}:${section.group}`}>
                <h3 className="group-title">{section.group}</h3>
                <div className="form-grid">
                  {section.fields.map((field) => (
                    <FieldRenderer
                      key={`${disease}:${field.name}`}
                      field={field}
                      copy={fieldCopy(disease, field.name)}
                      values={diseaseValues(disease)}
                      errors={diseaseErrors(disease)}
                      onValue={(name, value) => {
                        if (name === 'smokingStatus' || name === 'cigarettesPerDay' || name === 'yearsSmoked') {
                          updateShared(name, value)
                        } else {
                          updateSpecific(disease, name, value)
                        }
                      }}
                      warningText={diseaseWarning(disease, field, rawValue(disease, field))}
                    />
                  ))}
                </div>
              </section>
            ))}
          </div>
        ))}

        <button type="submit">Assess selected diseases</button>
      </form>

      <div data-testid="assessment-results">
        {selectedDiseases.map((disease) => {
          const item = results[disease]
          if (!item) return null
          return item.state === 'error' ? (
            <ResultCard
              key={disease}
              title={DISEASE_NAMES[disease]}
              state={item.state}
              result={item.result}
              message={item.message}
              errors={item.errors}
              violation={item.violation}
            />
          ) : (
            <div key={disease}>
              <h2>{DISEASE_NAMES[disease]}</h2>
              <ComparisonResult result={item.result} />
            </div>
          )
        })}
      </div>
    </div>
  )
}
