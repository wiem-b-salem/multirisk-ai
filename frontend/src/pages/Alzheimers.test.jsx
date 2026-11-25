import { render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import Home from './Home'
import Predict from './Predict'

const schema = {
  raw_fields: [
    {
      name: 'Age',
      type: 'int',
      train_min: 60,
      train_max: 90,
      valid_min: 40,
      valid_max: 110,
      label: 'Age',
      unit: 'years',
      help: 'Age in years.',
    },
    {
      name: 'FamilyHistoryAlzheimers',
      type: 'int',
      allowed_values: [0, 1],
      train_min: 0,
      train_max: 1,
      valid_min: null,
      valid_max: null,
      label: "Family history of Alzheimer's disease",
      unit: '',
      help: "Whether a close relative has or had Alzheimer's disease",
    },
    {
      name: 'EducationLevel',
      type: 'int',
      allowed_values: [0, 1, 2, 3],
      train_min: 0,
      train_max: 3,
      valid_min: null,
      valid_max: null,
      label: 'Education level',
      unit: '',
      help: '0 to 3 as coded in the dataset; meaning to be verified on the source page',
    },
    {
      name: 'BMI',
      type: 'float',
      train_min: 15.008851181631059,
      train_max: 39.99276746402374,
      valid_min: 10,
      valid_max: 70,
      label: 'Body mass index (BMI)',
      unit: 'kg/m2',
      help: 'Weight in kg divided by height in metres squared.',
    },
    {
      name: 'MMSE',
      type: 'float',
      train_min: 0.005312146441700527,
      train_max: 29.991380560529063,
      valid_min: 0,
      valid_max: 30,
      label: 'MMSE score',
      unit: 'points',
      help: 'Mini-Mental State Examination score, 0 to 30, higher is better; usually administered by a clinician',
    },
    {
      name: 'MemoryComplaints',
      type: 'int',
      allowed_values: [0, 1],
      train_min: 0,
      train_max: 1,
      valid_min: null,
      valid_max: null,
      label: 'Memory complaints',
      unit: '',
      help: 'Whether the person or their family reports memory problems',
    },
  ],
  family_history_field: 'FamilyHistoryAlzheimers',
  population_note:
    'Trained on a synthetic dataset of 2,149 people aged 60-90; not validated on real patients.',
  model_name: 'logreg',
}

function jsonResponse(body) {
  return Promise.resolve({ ok: true, status: 200, json: async () => body })
}

beforeEach(() => {
  sessionStorage.setItem('token', 'test-token')
})

afterEach(() => {
  vi.unstubAllGlobals()
  sessionStorage.clear()
})

describe('Alzheimer\'s pages', () => {
  it('offers an Alzheimer\'s disease card on the home page', () => {
    render(
      <MemoryRouter>
        <Home />
      </MemoryRouter>,
    )
    const card = screen.getByText("Alzheimer's disease").closest('a')
    expect(card.getAttribute('href')).toBe('/predict/alzheimers')
  })

  it('renders the Alzheimer\'s form with raw education codes and BMI inputs', async () => {
    vi.stubGlobal('fetch', vi.fn(() => jsonResponse(schema)))
    render(
      <MemoryRouter initialEntries={['/predict/alzheimers']}>
        <Routes>
          <Route path="/predict/:disease" element={<Predict />} />
        </Routes>
      </MemoryRouter>,
    )

    await screen.findByTestId('predict-form')
    expect(screen.getByTestId('family-history-box')).toBeTruthy()
    expect(screen.getByLabelText('Weight (kg)')).toBeTruthy()
    expect(screen.getByLabelText('Height (cm)')).toBeTruthy()
    const education = screen.getByLabelText('Education level')
    expect([...education.options].map((option) => option.value)).toEqual(['', '0', '1', '2', '3'])
    expect(screen.getByText(/0 to 3 as coded in the dataset/)).toBeTruthy()
    expect(screen.getByText(/synthetic dataset of 2,149 people/)).toBeTruthy()
  })
})
