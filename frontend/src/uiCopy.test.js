import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'
import { DISEASES } from './sharedFields'
import { GROUPS, SHARED_COPY, UI_COPY } from './uiCopy'

const SCHEMA_PATHS = {
  cvd: '../../models/cardiovascular/schema_cvd.json',
  diabetes: '../../models/diabetes/schema_diabetes_compact.json',
  alzheimers: '../../models/alzheimers/schema_alzheimers_compact.json',
  liver: '../../models/liver/schema_liver_full.json',
}

const FORBIDDEN = ['dataset', 'training', 'code', 'verified', 'coding', 'typical']

function loadSchemas() {
  const schemas = {}
  for (const disease of DISEASES) {
    const path = fileURLToPath(new URL(SCHEMA_PATHS[disease], import.meta.url))
    schemas[disease] = JSON.parse(readFileSync(path, 'utf8'))
  }
  return schemas
}

function copyTexts(copy) {
  return [copy.question, copy.hint, ...(copy.options ?? []).map((option) => option.label)].filter(
    Boolean,
  )
}

describe('ui copy', () => {
  const schemas = loadSchemas()

  it('covers every raw schema field with a question and a valid group', () => {
    for (const disease of DISEASES) {
      for (const field of schemas[disease].raw_fields) {
        const copy = UI_COPY[disease]?.[field.name]
        expect(copy, `${disease}.${field.name}`).toBeTruthy()
        expect(copy.question.length).toBeGreaterThan(0)
        expect(GROUPS).toContain(copy.group)
      }
    }
  })

  it('uses the schema option values for every categorical field', () => {
    for (const disease of DISEASES) {
      for (const field of schemas[disease].raw_fields) {
        if (!field.allowed_values) continue
        const copy = UI_COPY[disease][field.name]
        expect(copy.options, `${disease}.${field.name}`).toBeTruthy()
        expect(copy.options.map((option) => String(option.value)).sort()).toEqual(
          field.allowed_values.map(String).sort(),
        )
      }
    }
  })

  it('keeps developer wording out of questions, hints, and option labels', () => {
    const copies = []
    for (const disease of DISEASES) {
      for (const copy of Object.values(UI_COPY[disease])) copies.push(copy)
    }
    copies.push(...Object.values(SHARED_COPY))

    for (const copy of copies) {
      for (const text of copyTexts(copy)) {
        for (const word of FORBIDDEN) {
          expect(text.toLowerCase().includes(word), `${text} / ${word}`).toBe(false)
        }
      }
    }
  })
})
