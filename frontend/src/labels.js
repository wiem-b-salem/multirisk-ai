export const LABELS = {
  sbp: 'Systolic blood pressure',
  tobacco: 'Tobacco use',
  ldl: 'LDL cholesterol',
  adiposity: 'Adiposity',
  famhist: 'Family history of heart disease',
  typea: 'Type A behavior',
  obesity: 'Obesity',
  alcohol: 'Alcohol consumption',
  age: 'Age',
  Age: 'Age',
  Gender: 'Gender',
  BMI: 'BMI',
  FamilyHistoryDiabetes: 'Family history of diabetes',
  FastingBloodSugar: 'Fasting blood sugar',
  HbA1c: 'HbA1c',
  SystolicBP: 'Systolic blood pressure',
  DiastolicBP: 'Diastolic blood pressure',
  Hypertension: 'Hypertension',
  PreviousPreDiabetes: 'Previous prediabetes',
  FrequentUrination: 'Frequent urination',
  ExcessiveThirst: 'Excessive thirst',
}

export function labelFor(name) {
  return LABELS[name] || name
}
