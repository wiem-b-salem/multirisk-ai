import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'
import {
  DISEASES,
  SHARED_FIELD_MAP,
  normalizeSchema,
  outsideTrainingDiseases,
  widestValidRange,
} from './sharedFields'

const SCHEMA_PATHS = {
  cvd: '../../models/cardiovascular/schema_cvd.json',
  diabetes: '../../models/diabetes/schema_diabetes_compact.json',
  alzheimers: '../../models/alzheimers/schema_alzheimers_compact.json',
  liver: '../../models/liver/schema_liver_full.json',
}

function loadSchemas() {
  const schemas = {}
  for (const disease of DISEASES) {
    const path = fileURLToPath(new URL(SCHEMA_PATHS[disease], import.meta.url))
    schemas[disease] = normalizeSchema(disease, JSON.parse(readFileSync(path, 'utf8')))
  }
  return schemas
}

describe('shared fields', () => {
  const schemas = loadSchemas()

  it('maps shared inputs onto fields that exist in each schema', () => {
    expect(SHARED_FIELD_MAP).toEqual({
      age: { cvd: 'age', diabetes: 'Age', alzheimers: 'Age', liver: 'Age' },
      bmi: { cvd: 'obesity', diabetes: 'BMI', alzheimers: 'BMI', liver: 'BMI' },
      systolic_bp: { cvd: 'sbp', diabetes: 'SystolicBP' },
      hypertension: { diabetes: 'Hypertension', alzheimers: 'Hypertension', liver: 'Hypertension' },
      diabetes_diagnosis: { alzheimers: 'Diabetes', liver: 'Diabetes' },
    })

    for (const [key, mapping] of Object.entries(SHARED_FIELD_MAP)) {
      for (const [disease, name] of Object.entries(mapping)) {
        expect(schemas[disease].fields.some((field) => field.name === name), `${key}/${disease}`).toBe(
          true,
        )
      }
    }
  })

  it('uses the widest valid limits of the selected schemas', () => {
    expect(widestValidRange('age', DISEASES, schemas)).toEqual({ min: 15, max: 110 })
    expect(widestValidRange('bmi', DISEASES, schemas)).toEqual({ min: 10, max: 70 })
    expect(widestValidRange('systolic_bp', DISEASES, schemas)).toEqual({ min: 60, max: 280 })
    expect(widestValidRange('age', ['cvd'], schemas)).toEqual({ min: 15, max: 100 })
    expect(widestValidRange('age', ['alzheimers'], schemas)).toEqual({ min: 40, max: 110 })
  })

  it('names schemas whose training range excludes a shared value', () => {
    expect(outsideTrainingDiseases('age', 35, ['cvd', 'alzheimers'], schemas)).toEqual([
      "Alzheimer's disease",
    ])
    expect(outsideTrainingDiseases('age', 70, ['cvd', 'alzheimers'], schemas)).toEqual([
      'Cardiovascular',
    ])
    expect(outsideTrainingDiseases('age', 70, ['alzheimers'], schemas)).toEqual([])
  })

  it('loads the real family-history and genetic-risk fields', () => {
    expect(schemas.cvd.familyHistoryField).toBe('famhist')
    expect(schemas.diabetes.familyHistoryField).toBe('FamilyHistoryDiabetes')
    expect(schemas.alzheimers.familyHistoryField).toBe('FamilyHistoryAlzheimers')
    expect(schemas.liver.familyHistoryField).toBe('GeneticRisk')
    expect(schemas.liver.boxTitle).toBe('Genetic risk (required)')
  })
})
