import { fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import Predict from './Predict'

const schema = {
  raw_fields: [
    {
      name: 'age',
      type: 'int',
      train_min: 15,
      train_max: 64,
      valid_min: 15,
      valid_max: 100,
      label: 'Age',
      unit: 'years',
      help: 'Age in years.',
    },
    {
      name: 'famhist',
      type: 'category',
      allowed_values: ['Absent', 'Present'],
      train_min: null,
      train_max: null,
      valid_min: null,
      valid_max: null,
      label: 'Family history of heart disease',
      unit: '',
      help: 'Whether a close relative had ischaemic heart disease.',
    },
    {
      name: 'obesity',
      type: 'float',
      train_min: 14.7,
      train_max: 46.58,
      valid_min: 10,
      valid_max: 70,
      label: 'Body mass index (BMI)',
      unit: 'kg/m2',
      help: 'Weight in kg divided by height in metres squared.',
    },
    {
      name: 'tobacco',
      type: 'float',
      train_min: 0,
      train_max: 31.2,
      valid_min: 0,
      valid_max: 300,
      label: 'Cumulative tobacco use',
      unit: 'kg',
      help: 'Total tobacco used over a lifetime, not per year.',
    },
  ],
  family_history_field: 'famhist',
  population_note: 'Trained on 462 men aged 15-64 from a high-risk region.',
  model_name: 'logreg',
}

function jsonResponse(body, status = 200) {
  return Promise.resolve({
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  })
}

function renderPredict() {
  return render(
    <MemoryRouter initialEntries={['/predict/cvd']}>
      <Routes>
        <Route path="/predict/:disease" element={<Predict />} />
      </Routes>
    </MemoryRouter>,
  )
}

function fillNonFamilyFields(status = 'never') {
  fireEvent.change(screen.getByLabelText('Age'), { target: { value: '40' } })
  fireEvent.change(screen.getByLabelText('Weight (kg)'), { target: { value: '70' } })
  fireEvent.change(screen.getByLabelText('Height (cm)'), { target: { value: '175' } })
  fireEvent.change(screen.getByLabelText('Cumulative tobacco use'), {
    target: { value: status },
  })
}

beforeEach(() => {
  sessionStorage.setItem('token', 'test-token')
})

afterEach(() => {
  vi.unstubAllGlobals()
  sessionStorage.clear()
})

describe('Predict form', () => {
  it('renders the family history box and the derived inputs', async () => {
    vi.stubGlobal('fetch', vi.fn(() => jsonResponse(schema)))
    renderPredict()

    const form = await screen.findByTestId('predict-form')
    expect(screen.getByTestId('family-history-box')).toBeTruthy()
    expect(form.querySelector('input, select').getAttribute('name')).toBe('famhist')
    expect(screen.getByLabelText('Weight (kg)')).toBeTruthy()
    expect(screen.getByLabelText('Height (cm)')).toBeTruthy()
    expect(screen.getByLabelText('Cumulative tobacco use')).toBeTruthy()
  })

  it('shows the computed BMI from weight and height', async () => {
    vi.stubGlobal('fetch', vi.fn(() => jsonResponse(schema)))
    renderPredict()

    await screen.findByTestId('predict-form')
    fireEvent.change(screen.getByLabelText('Weight (kg)'), { target: { value: '70' } })
    fireEvent.change(screen.getByLabelText('Height (cm)'), { target: { value: '175' } })

    expect(await screen.findByText(/BMI used by the model: 22\.9/)).toBeTruthy()
  })

  it('hides the cigarette inputs for never smokers', async () => {
    vi.stubGlobal('fetch', vi.fn(() => jsonResponse(schema)))
    renderPredict()

    await screen.findByTestId('predict-form')
    fireEvent.change(screen.getByLabelText('Cumulative tobacco use'), {
      target: { value: 'current' },
    })
    expect(screen.getByLabelText('Cigarettes per day')).toBeTruthy()
    expect(screen.getByLabelText('Years smoked')).toBeTruthy()

    fireEvent.change(screen.getByLabelText('Cumulative tobacco use'), {
      target: { value: 'never' },
    })
    expect(screen.queryByLabelText('Cigarettes per day')).toBeNull()
    expect(screen.queryByLabelText('Years smoked')).toBeNull()
  })

  it('blocks submit when weight and height are missing and sends no request', async () => {
    const fetchMock = vi.fn(() => jsonResponse(schema))
    vi.stubGlobal('fetch', fetchMock)
    renderPredict()

    await screen.findByTestId('predict-form')
    fireEvent.change(screen.getByLabelText('Age'), { target: { value: '40' } })
    fireEvent.change(screen.getByLabelText('Family history of heart disease'), {
      target: { value: 'Present' },
    })
    fireEvent.change(screen.getByLabelText('Cumulative tobacco use'), {
      target: { value: 'never' },
    })
    fireEvent.click(screen.getByRole('button', { name: 'Predict' }))

    await screen.findByText('This field is required.')
    const postCalls = fetchMock.mock.calls.filter(
      ([, options]) => options && options.method === 'POST',
    )
    expect(postCalls).toHaveLength(0)
  })

  it('sends computed obesity and tobacco and omits the helper keys', async () => {
    const result = { disease: 'cvd', risk_score: 0.95, risk_level: 'high' }
    const fetchMock = vi.fn((url, options = {}) =>
      options.method === 'POST' ? jsonResponse(result) : jsonResponse(schema),
    )
    vi.stubGlobal('fetch', fetchMock)
    renderPredict()

    await screen.findByTestId('predict-form')
    fillNonFamilyFields('current')
    fireEvent.change(screen.getByLabelText('Family history of heart disease'), {
      target: { value: 'Present' },
    })
    fireEvent.change(screen.getByLabelText('Cigarettes per day'), {
      target: { value: '20' },
    })
    fireEvent.change(screen.getByLabelText('Years smoked'), { target: { value: '10' } })
    fireEvent.click(screen.getByRole('button', { name: 'Predict' }))

    await screen.findByText('high')
    const post = fetchMock.mock.calls.find(
      ([, options]) => options && options.method === 'POST',
    )
    const body = JSON.parse(post[1].body)
    expect(body.features.obesity).toBe(22.9)
    expect(body.features.tobacco).toBe(73)
    expect(body.features).not.toHaveProperty('weight')
    expect(body.features).not.toHaveProperty('height')
    expect(body.features).not.toHaveProperty('smokingStatus')
    expect(body.features).not.toHaveProperty('cigarettesPerDay')
    expect(body.features).not.toHaveProperty('yearsSmoked')
  })

  it('shows the risk badge and disclaimer and hides factors when top_factors is null', async () => {
    const result = {
      disease: 'cvd',
      model_name: 'rf',
      risk_score: 0.95,
      risk_level: 'high',
      score_note: 'Model score note.',
      top_factors: null,
      disclaimer: 'Academic estimate only.',
      model_note: 'Model note.',
    }
    vi.stubGlobal(
      'fetch',
      vi.fn((url, options = {}) =>
        options.method === 'POST' ? jsonResponse(result) : jsonResponse(schema),
      ),
    )
    renderPredict()

    await screen.findByTestId('predict-form')
    fillNonFamilyFields('never')
    fireEvent.change(screen.getByLabelText('Family history of heart disease'), {
      target: { value: 'Present' },
    })
    fireEvent.click(screen.getByRole('button', { name: 'Predict' }))

    expect(await screen.findByText('high')).toBeTruthy()
    expect(screen.getByText('Academic estimate only.')).toBeTruthy()
    expect(screen.queryByText('Top factors')).toBeNull()
  })

  it('shows an amber notice when a value is outside the training range', async () => {
    vi.stubGlobal('fetch', vi.fn(() => jsonResponse(schema)))
    renderPredict()

    await screen.findByTestId('predict-form')
    fireEvent.change(screen.getByLabelText('Age'), { target: { value: '80' } })

    expect(
      await screen.findByText(
        'Outside the training range: the estimate may be unreliable.',
      ),
    ).toBeTruthy()
  })
})
