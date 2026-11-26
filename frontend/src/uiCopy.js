export const GROUPS = [
  'About you',
  'Medical history',
  'Symptoms',
  'Test results',
  'Clinical assessments',
]

export const GENDER_OPTIONS = [
  { value: 0, label: 'Male' },
  { value: 1, label: 'Female' },
]

export const YES_NO = [
  { value: 1, label: 'Yes' },
  { value: 0, label: 'No' },
]

const bloodPressureHint =
  'The top number of a blood pressure reading, for example 120 in 120/80.'

export const UI_COPY = {
  cvd: {
    age: { question: 'How old are you?', group: 'About you', unit: 'years' },
    obesity: { question: 'BMI', group: 'About you' },
    tobacco: {
      question: 'Do you smoke?',
      group: 'About you',
      options: [
        { value: 'never', label: 'Never smoked' },
        { value: 'former', label: 'Used to smoke' },
        { value: 'current', label: 'Currently smoke' },
      ],
    },
    alcohol: {
      question: 'How much alcohol do you drink?',
      group: 'About you',
      hint: 'Your current alcohol consumption.',
    },
    famhist: {
      question: 'Has a close family member had heart disease?',
      group: 'Medical history',
      hint: 'For example a parent, brother or sister.',
      options: [
        { value: 'Present', label: 'Yes' },
        { value: 'Absent', label: 'No' },
      ],
    },
    sbp: {
      question: 'Systolic blood pressure',
      group: 'Test results',
      unit: 'mm Hg',
      hint: bloodPressureHint,
    },
    ldl: {
      question: 'LDL cholesterol',
      group: 'Test results',
      hint: 'From a blood test. Enter the value as shown on your report.',
    },
    adiposity: {
      question: 'Adiposity',
      group: 'Clinical assessments',
      hint: 'A body-fat measure from a clinical assessment.',
    },
    typea: {
      question: 'Type A behavior score',
      group: 'Clinical assessments',
      unit: 'points',
      hint: 'Result of a Type A behavior questionnaire.',
    },
  },
  diabetes: {
    Age: { question: 'How old are you?', group: 'About you', unit: 'years' },
    Gender: { question: 'What is your sex?', group: 'About you', options: GENDER_OPTIONS },
    BMI: { question: 'BMI', group: 'About you' },
    FamilyHistoryDiabetes: {
      question: 'Does a close relative have diabetes?',
      group: 'Medical history',
      hint: 'For example a parent, brother or sister.',
      options: YES_NO,
    },
    Hypertension: {
      question: 'Have you been diagnosed with high blood pressure?',
      group: 'Medical history',
      options: YES_NO,
    },
    PreviousPreDiabetes: {
      question: 'Have you ever been told you have prediabetes?',
      group: 'Medical history',
      options: YES_NO,
    },
    FrequentUrination: {
      question: 'Do you urinate more often than usual?',
      group: 'Symptoms',
      options: YES_NO,
    },
    ExcessiveThirst: {
      question: 'Do you often feel very thirsty?',
      group: 'Symptoms',
      options: YES_NO,
    },
    FastingBloodSugar: {
      question: 'Fasting blood sugar',
      group: 'Test results',
      unit: 'mg/dL',
      hint: 'From a blood test taken after fasting.',
    },
    HbA1c: {
      question: 'HbA1c',
      group: 'Test results',
      unit: '%',
      hint: 'From a blood test; it reflects your average blood sugar over the past 2 to 3 months.',
    },
    SystolicBP: {
      question: 'Systolic blood pressure',
      group: 'Test results',
      unit: 'mm Hg',
      hint: bloodPressureHint,
    },
    DiastolicBP: {
      question: 'Diastolic blood pressure',
      group: 'Test results',
      unit: 'mm Hg',
      hint: 'The bottom number of a blood pressure reading, for example 80 in 120/80.',
    },
  },
  alzheimers: {
    Age: { question: 'How old are you?', group: 'About you', unit: 'years' },
    Gender: { question: 'What is your sex?', group: 'About you', options: GENDER_OPTIONS },
    EducationLevel: {
      question: 'What is your highest level of education?',
      group: 'About you',
      options: [
        { value: 0, label: 'None' },
        { value: 1, label: 'High school' },
        { value: 2, label: "Bachelor's degree" },
        { value: 3, label: 'Higher degree' },
      ],
    },
    BMI: { question: 'BMI', group: 'About you' },
    FamilyHistoryAlzheimers: {
      question: "Has a close relative had Alzheimer's disease?",
      group: 'Medical history',
      hint: 'For example a parent, brother or sister.',
      options: YES_NO,
    },
    CardiovascularDisease: {
      question: 'Have you been diagnosed with heart or blood vessel disease?',
      group: 'Medical history',
      options: YES_NO,
    },
    Diabetes: {
      question: 'Have you been diagnosed with diabetes?',
      group: 'Medical history',
      options: YES_NO,
    },
    Depression: {
      question: 'Have you been diagnosed with depression?',
      group: 'Medical history',
      options: YES_NO,
    },
    HeadInjury: {
      question: 'Have you had a head injury?',
      group: 'Medical history',
      options: YES_NO,
    },
    Hypertension: {
      question: 'Have you been diagnosed with high blood pressure?',
      group: 'Medical history',
      options: YES_NO,
    },
    MMSE: {
      question: 'MMSE score',
      group: 'Clinical assessments',
      unit: 'points',
      hint: 'Mini-Mental State Examination, scored from 0 to 30 (higher is better). Usually done by a clinician.',
    },
    MemoryComplaints: {
      question: 'Do you have complaints about your memory?',
      group: 'Symptoms',
      options: YES_NO,
    },
  },
  liver: {
    Age: { question: 'How old are you?', group: 'About you', unit: 'years' },
    Gender: { question: 'What is your sex?', group: 'About you', options: GENDER_OPTIONS },
    BMI: { question: 'BMI', group: 'About you' },
    AlcoholConsumption: {
      question: 'How much alcohol do you drink per week?',
      group: 'About you',
      unit: 'units per week',
    },
    Smoking: { question: 'Do you smoke?', group: 'About you', options: YES_NO },
    PhysicalActivity: {
      question: 'How much physical activity do you do per week?',
      group: 'About you',
      unit: 'hours per week',
    },
    GeneticRisk: {
      question: 'Genetic risk level',
      group: 'Medical history',
      hint: 'Usually determined by a clinician or a genetic test.',
      options: [
        { value: 0, label: 'Low' },
        { value: 1, label: 'Medium' },
        { value: 2, label: 'High' },
      ],
    },
    Diabetes: {
      question: 'Have you been diagnosed with diabetes?',
      group: 'Medical history',
      options: YES_NO,
    },
    Hypertension: {
      question: 'Have you been diagnosed with high blood pressure?',
      group: 'Medical history',
      options: YES_NO,
    },
    LiverFunctionTest: {
      question: 'Liver function test result',
      group: 'Test results',
      hint: 'From a blood test. Enter the value as shown on your report.',
    },
  },
}

export const SHARED_COPY = {
  age: { question: 'How old are you?', group: 'About you', unit: 'years' },
  bmi: { question: 'BMI', group: 'About you' },
  systolic_bp: {
    question: 'Systolic blood pressure',
    group: 'Test results',
    unit: 'mm Hg',
    hint: bloodPressureHint,
  },
  hypertension: {
    question: 'Have you been diagnosed with high blood pressure?',
    group: 'Medical history',
    options: YES_NO,
  },
  diabetes_diagnosis: {
    question: 'Have you been diagnosed with diabetes?',
    group: 'Medical history',
    options: YES_NO,
  },
}

export const OUT_OF_RANGE_NOTICE =
  'This value is outside the range the model was built for, so the result may be less reliable.'

export function fieldCopy(disease, name) {
  const copy = UI_COPY[disease]?.[name]
  if (!copy) throw new Error(`Missing UI copy for ${disease}.${name}`)
  return copy
}

export function sharedCopy(key) {
  const copy = SHARED_COPY[key]
  if (!copy) throw new Error(`Missing shared UI copy for ${key}`)
  return copy
}

export function groupedFields(disease, fields) {
  const byGroup = new Map(GROUPS.map((group) => [group, []]))
  for (const field of fields) {
    byGroup.get(fieldCopy(disease, field.name).group).push(field)
  }
  return GROUPS.map((group) => ({ group, fields: byGroup.get(group) })).filter(
    (section) => section.fields.length > 0,
  )
}

export function rangeMessage(field) {
  return `Please enter a value between ${field.valid_min} and ${field.valid_max}.`
}

export function outsideRangeNotice(diseaseNames = []) {
  if (diseaseNames.length === 0) return OUT_OF_RANGE_NOTICE
  return `${OUT_OF_RANGE_NOTICE} Less reliable for: ${diseaseNames.join(', ')}.`
}
