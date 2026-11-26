import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import Assessment from './Assessment'
import { DISEASES } from '../sharedFields'

const SCHEMA_PATHS = {
  cvd: '../../../models/cardiovascular/schema_cvd.json',
  diabetes: '../../../models/diabetes/schema_diabetes_compact.json',
  alzheimers: '../../../models/alzheimers/schema_alzheimers_compact.json',
  liver: '../../../models/liver/schema_liver_full.json',
}

const schemas = {}
for (const disease of DISEASES) {
  const path = fileURLToPath(new URL(SCHEMA_PATHS[disease], import.meta.url))
  schemas[disease] = JSON.parse(readFileSync(path, 'utf8'))
}

function jsonResponse(body, status = 200) {
  return Promise.resolve({
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  })
}

function resultFor(disease) {
  const common = {
    disease,
    risk_score: 0.95,
    risk_level: 'high',
    score_note: 'Model score note.',
    top_factors: null,
    model_note: 'Model note.',
    range_warning: 'Outside training range.',
    out_of_training_range: [{ field: 'Age' }],
    disclaimer: 'Academic estimate only.',
  }
  return {
    disease,
    ml: {
      ...common,
      model_name: 'rf',
      top_factors: [{ feature: 'age', effect: 'above_average' }],
      factors_note: 'Factors note.',
    },
    dl: {
      ...common,
      model_name: 'mlp',
      model_label: 'Neural Network (MLP)',
      score_note: 'DL score note.',
    },
    comparison: {
      score_difference: 0,
      risk_levels_match: true,
      interpretation: 'Both models classify this input as high risk.',
      note: 'Comparison note.',
      ml_test_roc_auc: 0.9,
      dl_test_roc_auc: 0.8,
      ml_test_accuracy: 0.85,
      dl_test_accuracy: 0.75,
    },
    disclaimer: 'Academic estimate only.',
  }
}

function mockFetch(postHandler) {
  const fetchMock = vi.fn((url, options = {}) => {
    const disease = String(url).split('/').pop()
    if (options.method === 'POST') {
      return postHandler ? postHandler(disease) : jsonResponse(resultFor(disease))
    }
    return jsonResponse(schemas[disease])
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

function renderAssessment({ withLogin = false } = {}) {
  return render(
    <MemoryRouter initialEntries={['/assessment']}>
      <Routes>
        <Route path="/assessment" element={<Assessment />} />
        {withLogin && <Route path="/login" element={<div>Login page</div>} />}
      </Routes>
    </MemoryRouter>,
  )
}

function fillAll({ age = '40', weight = '70', height = '175' } = {}) {
  const changes = {
    'Has a close family member had heart disease?': 'Present',
    'How old are you?': age,
    Weight: weight,
    Height: height,
    'Systolic blood pressure': '120',
    'Have you been diagnosed with high blood pressure?': '1',
    'Have you been diagnosed with diabetes?': '0',
    'LDL cholesterol': '2.5',
    Adiposity: '20',
    'Type A behavior score': '50',
    'How much alcohol do you drink?': '10',
    'What is your sex?': '0',
    'What is your highest level of education?': '2',
    "Has a close relative had Alzheimer's disease?": '1',
    'Have you been diagnosed with heart or blood vessel disease?': '0',
    'Have you been diagnosed with depression?': '0',
    'Have you had a head injury?': '0',
    'MMSE score': '28',
    'Do you have complaints about your memory?': '0',
    'How much alcohol do you drink per week?': '10',
    'How much physical activity do you do per week?': '5',
    'Genetic risk level': '1',
    'Liver function test result': '50',
  }
  for (const [label, value] of Object.entries(changes)) {
    const control = screen.queryByLabelText(label)
    if (control) fireEvent.change(control, { target: { value } })
  }
  const smoking = screen.queryByLabelText('Do you smoke?')
  if (smoking) {
    const value = [...smoking.options].some((option) => option.value === 'never') ? 'never' : '0'
    fireEvent.change(smoking, { target: { value } })
  }
}

async function submit() {
  fireEvent.click(screen.getByRole('button', { name: 'Assess selected diseases' }))
  await waitFor(() => expect(screen.queryByText('Select at least one disease.')).toBeNull())
}

beforeEach(() => {
  sessionStorage.setItem('token', 'test-token')
})

afterEach(() => {
  vi.unstubAllGlobals()
  sessionStorage.clear()
})

describe('Full assessment', () => {
  it('shows four unchecked checkboxes and does not let the last one be unchecked', async () => {
    mockFetch()
    renderAssessment()
    await screen.findByTestId('assessment-form')

    const checkboxes = DISEASES.map((disease) => screen.getByLabelText(disease === 'cvd' ? 'Cardiovascular' : disease === 'diabetes' ? 'Type 2 Diabetes' : disease === 'alzheimers' ? "Alzheimer's disease" : 'Liver disease'))
    expect(checkboxes.every((checkbox) => checkbox.checked === false)).toBe(true)

    for (const checkbox of checkboxes) fireEvent.click(checkbox)
    expect(checkboxes.every((checkbox) => checkbox.checked === true)).toBe(true)

    for (const checkbox of checkboxes.slice(0, 3)) fireEvent.click(checkbox)
    expect(checkboxes.slice(0, 3).every((checkbox) => checkbox.checked === false)).toBe(true)
    expect(checkboxes[3].checked).toBe(true)
  })

  it('shows shared labels and the widest valid limits of the selected diseases', async () => {
    mockFetch()
    renderAssessment()
    await screen.findByTestId('assessment-form')
    fireEvent.click(screen.getByLabelText('Cardiovascular'))
    fireEvent.click(screen.getByLabelText("Alzheimer's disease"))

    expect(screen.getByLabelText('How old are you?').getAttribute('min')).toBe('15')
    expect(screen.getByLabelText('How old are you?').getAttribute('max')).toBe('110')
    expect(screen.getByLabelText('Systolic blood pressure').getAttribute('min')).toBe('60')
    expect(screen.getByLabelText('Systolic blood pressure').getAttribute('max')).toBe('280')
    expect(screen.getByText('BMI')).toBeTruthy()
    expect(screen.getByLabelText('Have you been diagnosed with high blood pressure?')).toBeTruthy()
    expect(screen.getByLabelText('Have you been diagnosed with diabetes?')).toBeTruthy()
    expect(
      [...screen.getByLabelText('Has a close family member had heart disease?').options].map(
        (option) => option.value,
      ),
    ).toEqual(['', 'Present', 'Absent'])
  })

  it('sends exact cvd and alzheimers request bodies when those checkboxes are checked', async () => {
    const fetchMock = mockFetch()
    renderAssessment()
    await screen.findByTestId('assessment-form')
    fireEvent.click(screen.getByLabelText('Cardiovascular'))
    fireEvent.click(screen.getByLabelText("Alzheimer's disease"))
    fillAll()
    await submit()
    await waitFor(() => expect(screen.getAllByTestId('result-card')).toHaveLength(4))

    const posts = fetchMock.mock.calls.filter(([, options]) => options && options.method === 'POST')
    expect(posts).toHaveLength(2)
    const bodies = Object.fromEntries(
      posts.map(([url, options]) => [String(url).split('/').pop(), JSON.parse(options.body)]),
    )
    expect(bodies.cvd).toEqual({
      features: {
        sbp: 120,
        tobacco: 0,
        ldl: 2.5,
        adiposity: 20,
        famhist: 'Present',
        typea: 50,
        obesity: 22.9,
        alcohol: 10,
        age: 40,
      },
    })
    expect(bodies.alzheimers).toEqual({
      features: {
        Age: 40,
        Gender: 0,
        EducationLevel: 2,
        BMI: 22.9,
        FamilyHistoryAlzheimers: 1,
        CardiovascularDisease: 0,
        Diabetes: 0,
        Depression: 0,
        HeadInjury: 0,
        Hypertension: 1,
        MMSE: 28,
        MemoryComplaints: 0,
      },
    })
  })

  it('sends the exact liver request body when liver is checked', async () => {
    const fetchMock = mockFetch()
    renderAssessment()
    await screen.findByTestId('assessment-form')
    fireEvent.click(screen.getByLabelText('Liver disease'))
    fillAll()
    await submit()
    await waitFor(() => expect(screen.getAllByTestId('result-card')).toHaveLength(2))

    const posts = fetchMock.mock.calls.filter(([, options]) => options && options.method === 'POST')
    expect(posts).toHaveLength(1)
    const body = JSON.parse(posts[0][1].body)
    expect(body).toEqual({
      features: {
        Age: 40,
        Gender: 0,
        BMI: 22.9,
        AlcoholConsumption: 10,
        Smoking: 0,
        GeneticRisk: 1,
        PhysicalActivity: 5,
        Diabetes: 0,
        Hypertension: 1,
        LiverFunctionTest: 50,
      },
    })
  })

  it('skips prediction requests for schemas whose valid limits are violated', async () => {
    const fetchMock = mockFetch()
    renderAssessment()
    await screen.findByTestId('assessment-form')
    fireEvent.click(screen.getByLabelText('Cardiovascular'))
    fireEvent.click(screen.getByLabelText("Alzheimer's disease"))
    fillAll({ age: '120' })
    await submit()

    expect(await screen.findAllByText('Not assessed')).toHaveLength(2)
    expect(screen.getByText('Please enter a value between 15 and 100.')).toBeTruthy()
    expect(screen.getByText('Please enter a value between 40 and 110.')).toBeTruthy()
    expect(fetchMock.mock.calls.some(([, options]) => options && options.method === 'POST')).toBe(false)
  })

  it('shows an amber BMI notice naming the affected diseases', async () => {
    mockFetch()
    renderAssessment()
    await screen.findByTestId('assessment-form')
    fireEvent.click(screen.getByLabelText('Cardiovascular'))
    fireEvent.click(screen.getByLabelText("Alzheimer's disease"))
    fillAll({ weight: '150', height: '175' })

    expect(await screen.findByText('Your BMI: 49')).toBeTruthy()
    expect(
      screen.getByText(
        "This value is outside the range the model was built for, so the result may be less reliable. Less reliable for: Cardiovascular, Alzheimer's disease.",
      ),
    ).toBeTruthy()
  })

  it('shows a per-disease service unavailable card for 503 responses', async () => {
    mockFetch(() => jsonResponse({}, 503))
    renderAssessment()
    await screen.findByTestId('assessment-form')
    fireEvent.click(screen.getByLabelText('Cardiovascular'))
    fillAll()
    await submit()

    expect(await screen.findByText('Prediction service unavailable')).toBeTruthy()
    expect(screen.getAllByTestId('result-card')).toHaveLength(1)
  })

  it('shows separate result details with no combined score', async () => {
    mockFetch()
    renderAssessment()
    await screen.findByTestId('assessment-form')
    fireEvent.click(screen.getByLabelText('Cardiovascular'))
    fillAll()
    await submit()

    expect(await screen.findAllByText('Model score: 95 / 100')).toHaveLength(2)
    expect(screen.getByText('Model score note.')).toBeTruthy()
    expect(screen.getByText('DL score note.')).toBeTruthy()
    expect(screen.getByText('Top factors')).toBeTruthy()
    expect(screen.getByText('Age: above average')).toBeTruthy()
    expect(screen.getByText('Factors note.')).toBeTruthy()
    expect(screen.getAllByText('Model note.')).toHaveLength(2)
    expect(screen.getAllByText(/Outside training range\. Fields: Age/)).toHaveLength(2)
    expect(screen.getAllByText('Academic estimate only.')).toHaveLength(3)
    expect(screen.getByText('ML vs DL comparison')).toBeTruthy()
    expect(screen.getByText('Both models classify this input as high risk.')).toBeTruthy()
    expect(screen.queryByText(/combined|overall|average score/i)).toBeNull()
  })

  it('clears the token and navigates to login when a schema request returns 401', async () => {
    vi.stubGlobal('fetch', vi.fn(() => jsonResponse({}, 401)))
    renderAssessment({ withLogin: true })

    expect(await screen.findByText('Login page')).toBeTruthy()
    expect(sessionStorage.getItem('token')).toBeNull()
  })
})
