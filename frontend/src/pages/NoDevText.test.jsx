import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import Assessment from './Assessment'
import Predict from './Predict'
import { DISEASES } from '../sharedFields'

const SCHEMA_PATHS = {
  cvd: '../../../models/cardiovascular/schema_cvd.json',
  diabetes: '../../../models/diabetes/schema_diabetes_compact.json',
  alzheimers: '../../../models/alzheimers/schema_alzheimers_compact.json',
  liver: '../../../models/liver/schema_liver_full.json',
}

const FORBIDDEN_PHRASES = [
  'dataset',
  'training',
  'gender code',
  'to be verified',
  'typical range',
  'allowed range',
  'not state',
  'coding',
]

const schemas = {}
for (const disease of DISEASES) {
  const path = fileURLToPath(new URL(SCHEMA_PATHS[disease], import.meta.url))
  schemas[disease] = JSON.parse(readFileSync(path, 'utf8'))
}

function mockFetch() {
  vi.stubGlobal(
    'fetch',
    vi.fn((url) => {
      const disease = String(url).split('/').pop()
      return Promise.resolve({
        ok: true,
        status: 200,
        json: async () => schemas[disease],
      })
    }),
  )
}

function expectNoDeveloperText() {
  const text = document.body.textContent.toLowerCase()
  for (const phrase of FORBIDDEN_PHRASES) {
    expect(text.includes(phrase), phrase).toBe(false)
  }
}

function renderPredict(disease) {
  return render(
    <MemoryRouter initialEntries={[`/predict/${disease}`]}>
      <Routes>
        <Route path="/predict/:disease" element={<Predict />} />
      </Routes>
    </MemoryRouter>,
  )
}

function renderAssessment() {
  return render(
    <MemoryRouter initialEntries={['/assessment']}>
      <Routes>
        <Route path="/assessment" element={<Assessment />} />
      </Routes>
    </MemoryRouter>,
  )
}

beforeEach(() => {
  sessionStorage.setItem('token', 'test-token')
})

afterEach(() => {
  vi.unstubAllGlobals()
  sessionStorage.clear()
})

describe('form wording', () => {
  it('keeps developer wording out of every single-disease form', async () => {
    mockFetch()
    for (const disease of DISEASES) {
      const view = renderPredict(disease)
      await screen.findByTestId('predict-form')
      expectNoDeveloperText()
      view.unmount()
    }
  })

  it('keeps developer wording out of the full assessment form', async () => {
    mockFetch()
    renderAssessment()
    await screen.findByTestId('assessment-form')
    expectNoDeveloperText()
  })
})
