import { render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import Home from './Home'
import Predict from './Predict'

const liverSchema = {
  raw_fields: [
    {
      name: 'Age',
      type: 'int',
      train_min: 20,
      train_max: 80,
      valid_min: 18,
      valid_max: 110,
      label: 'Age',
      unit: 'years',
      help: 'Age in years.',
    },
    {
      name: 'BMI',
      type: 'float',
      train_min: 15,
      train_max: 40,
      valid_min: 10,
      valid_max: 70,
      label: 'Body mass index (BMI)',
      unit: 'kg/m2',
      help: 'Weight in kg divided by height in metres squared.',
    },
    {
      name: 'GeneticRisk',
      type: 'int',
      train_min: 0,
      train_max: 2,
      valid_min: null,
      valid_max: null,
      allowed_values: [0, 1, 2],
      label: 'Genetic risk level',
      unit: '',
      help: 'Genetic risk category from 0 to 2 as coded in the dataset.',
    },
  ],
  family_history_field: 'GeneticRisk',
  family_history_box_title: 'Genetic risk (required)',
  population_note: 'Trained on a synthetic dataset of 1,700 people aged 20-80.',
  model_name: 'rf',
}

const alzheimersSchema = {
  raw_fields: [
    {
      name: 'FamilyHistoryAlzheimers',
      type: 'int',
      train_min: 0,
      train_max: 1,
      valid_min: null,
      valid_max: null,
      allowed_values: [0, 1],
      label: "Family history of Alzheimer's disease",
      unit: '',
      help: "Whether a close relative has or had Alzheimer's disease",
    },
  ],
  family_history_field: 'FamilyHistoryAlzheimers',
  population_note: 'Trained on a synthetic dataset.',
  model_name: 'logreg',
}

function jsonResponse(body) {
  return Promise.resolve({ ok: true, status: 200, json: async () => body })
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

beforeEach(() => {
  sessionStorage.setItem('token', 'test-token')
})

afterEach(() => {
  vi.unstubAllGlobals()
  sessionStorage.clear()
})

describe('Liver disease pages', () => {
  it('offers a liver disease card on the home page', () => {
    render(
      <MemoryRouter>
        <Home />
      </MemoryRouter>,
    )
    const card = screen.getByText('Liver disease').closest('a')
    expect(card.getAttribute('href')).toBe('/predict/liver')
  })

  it('renders the liver form with genetic risk level codes and BMI inputs', async () => {
    vi.stubGlobal('fetch', vi.fn(() => jsonResponse(liverSchema)))
    renderPredict('liver')

    await screen.findByTestId('predict-form')
    expect(screen.getByText('Genetic risk (required)')).toBeTruthy()
    expect(screen.queryByText('Family history (required)')).toBeNull()
    const geneticRisk = screen.getByLabelText('Genetic risk level')
    expect([...geneticRisk.options].map((option) => option.value)).toEqual(['', '0', '1', '2'])
    expect(screen.getAllByText(/Genetic risk category from 0 to 2/).length).toBeGreaterThan(0)
    expect(screen.getByLabelText('Weight (kg)')).toBeTruthy()
    expect(screen.getByLabelText('Height (cm)')).toBeTruthy()
  })

  it('keeps the default box title for other diseases', async () => {
    vi.stubGlobal('fetch', vi.fn(() => jsonResponse(alzheimersSchema)))
    renderPredict('alzheimers')

    await screen.findByTestId('predict-form')
    expect(screen.getByText('Family history (required)')).toBeTruthy()
    expect(screen.queryByText('Genetic risk (required)')).toBeNull()
  })
})
