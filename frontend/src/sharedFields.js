import { labelFor } from './labels'

export const DISEASES = ['cvd', 'diabetes', 'alzheimers', 'liver']

export const DISEASE_NAMES = {
  cvd: 'Cardiovascular',
  diabetes: 'Type 2 Diabetes',
  alzheimers: "Alzheimer's disease",
  liver: 'Liver disease',
}

export const FAMILY_HISTORY = {
  cvd: 'famhist',
  diabetes: 'FamilyHistoryDiabetes',
  alzheimers: 'FamilyHistoryAlzheimers',
  liver: 'GeneticRisk',
}

export const DEFAULT_BOX_TITLE = 'Family history (required)'

export const SHARED_FIELDS = ['age', 'bmi', 'systolic_bp', 'hypertension', 'diabetes_diagnosis']

export const SHARED_FIELD_MAP = {
  age: { cvd: 'age', diabetes: 'Age', alzheimers: 'Age', liver: 'Age' },
  bmi: { cvd: 'obesity', diabetes: 'BMI', alzheimers: 'BMI', liver: 'BMI' },
  systolic_bp: { cvd: 'sbp', diabetes: 'SystolicBP' },
  hypertension: { diabetes: 'Hypertension', alzheimers: 'Hypertension', liver: 'Hypertension' },
  diabetes_diagnosis: { alzheimers: 'Diabetes', liver: 'Diabetes' },
}

export function normalizeSchema(disease, data) {
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
    modelName: data.model_name || '',
  }
}

export function fieldLabel(field) {
  return field.label || labelFor(field.name)
}

export function isOutsideTrainRange(field, value) {
  if (value === '' || value === null || value === undefined) return false
  if (field.train_min === null || field.train_min === undefined) return false
  const numeric = Number(value)
  return numeric < field.train_min || numeric > field.train_max
}

export function isBinary(field) {
  const values = field.allowed_values
  return Array.isArray(values) && values.length === 2 && values[0] === 0 && values[1] === 1
}

export function isBmiField(field) {
  return field.name === 'BMI' || field.name === 'obesity'
}

export function isTobaccoField(field) {
  return field.name === 'tobacco'
}

export function optionLabel(field, value) {
  if (isBinary(field)) {
    if (field.name === 'Gender') return value === 0 ? 'Gender code 0' : 'Gender code 1'
    return value === 0 ? 'No' : 'Yes'
  }
  return String(value)
}

export function mapServerErrors(detail) {
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

export function convertValue(field, value) {
  if (field.type === 'int') return Number.parseInt(value, 10)
  if (field.type === 'float') return Number.parseFloat(value)
  return value
}

export function buildFeatures(fields, getValue) {
  const features = {}
  for (const field of fields) {
    const value = getValue(field)
    if (value === '' || value === null || value === undefined) continue
    features[field.name] = convertValue(field, value)
  }
  return features
}

export function sharedKeyForField(disease, fieldName) {
  return SHARED_FIELDS.find((key) => SHARED_FIELD_MAP[key][disease] === fieldName) || null
}

export function findSharedField(key, disease, schemas) {
  const name = SHARED_FIELD_MAP[key]?.[disease]
  if (!name || !schemas[disease]) return null
  return schemas[disease].fields.find((field) => field.name === name) || null
}

export function sharedFieldDefinition(key, diseases, schemas, fallback) {
  for (const disease of diseases) {
    const field = findSharedField(key, disease, schemas)
    if (field) return field
  }
  return fallback
}

export function widestValidRange(key, diseases, schemas) {
  const limits = diseases
    .map((disease) => findSharedField(key, disease, schemas))
    .filter((field) => field && field.valid_min != null && field.valid_max != null)
  if (limits.length === 0) return { min: null, max: null }
  return {
    min: Math.min(...limits.map((field) => field.valid_min)),
    max: Math.max(...limits.map((field) => field.valid_max)),
  }
}

export function trainingRangeInfo(key, diseases, schemas) {
  return diseases
    .map((disease) => {
      const field = findSharedField(key, disease, schemas)
      if (!field || field.train_min == null || field.train_max == null) return null
      return { disease, min: field.train_min, max: field.train_max }
    })
    .filter(Boolean)
}

export function outsideTrainingDiseases(key, value, diseases, schemas) {
  const numeric = Number(value)
  if (value === '' || value === null || value === undefined || Number.isNaN(numeric)) return []
  return trainingRangeInfo(key, diseases, schemas)
    .filter((info) => numeric < info.min || numeric > info.max)
    .map((info) => DISEASE_NAMES[info.disease])
}

export function validLimitsViolation(field, value) {
  if (value === '' || value === null || value === undefined) return null
  const numeric = Number(value)
  if (Number.isNaN(numeric)) return null
  const below = field.valid_min != null && numeric < field.valid_min
  const above = field.valid_max != null && numeric > field.valid_max
  if (!below && !above) return null
  return {
    field: fieldLabel(field),
    value: numeric,
    min: field.valid_min,
    max: field.valid_max,
  }
}
