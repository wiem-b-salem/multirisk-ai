import { GRAMS_PER_CIGARETTE, bmiFromWeightHeight, cumulativeTobaccoKg } from '../derived'
import { isBmiField, isOutsideTrainRange, isTobaccoField } from '../sharedFields'
import { outsideRangeNotice, rangeMessage } from '../uiCopy'

function limitError(field, value) {
  if (value === '' || value === null || value === undefined) return ''
  const numeric = Number(value)
  if (Number.isNaN(numeric)) return ''
  const below = field.valid_min != null && numeric < field.valid_min
  const above = field.valid_max != null && numeric > field.valid_max
  return below || above ? rangeMessage(field) : ''
}

function Warning({ text }) {
  if (!text) return null
  return <span className="warning">{text}</span>
}

export function FieldInput({ field, copy, value, error, onChange, warningText }) {
  if (!copy) throw new Error(`Missing UI copy for field ${field.name}`)
  const options = copy.options ?? []
  const shownError = error || limitError(field, value)
  const fallbackWarning = isOutsideTrainRange(field, value) ? outsideRangeNotice() : ''

  return (
    <div className="field">
      <label className="field-question required" htmlFor={field.name}>
        {copy.question}
      </label>
      {copy.hint && <span className="hint">{copy.hint}</span>}
      <div className="input-row">
        {options.length > 0 ? (
          <select
            id={field.name}
            name={field.name}
            value={value}
            onChange={(event) => onChange(event.target.value)}
          >
            <option value="">Select...</option>
            {options.map((option) => (
              <option key={String(option.value)} value={String(option.value)}>
                {option.label}
              </option>
            ))}
          </select>
        ) : (
          <input
            id={field.name}
            name={field.name}
            type="number"
            min={field.valid_min ?? undefined}
            max={field.valid_max ?? undefined}
            step={field.type === 'int' ? 1 : 'any'}
            value={value}
            onChange={(event) => onChange(event.target.value)}
          />
        )}
        {copy.unit && <span className="unit">| {copy.unit}</span>}
      </div>
      <Warning text={warningText || fallbackWarning} />
      {shownError && <span className="error">{shownError}</span>}
    </div>
  )
}

export function BmiInput({ field, copy, values, onValue, error, warningText }) {
  if (!copy) throw new Error(`Missing UI copy for field ${field.name}`)
  const weight = values.weight ?? ''
  const height = values.height ?? ''
  const bmi = weight === '' || height === '' ? null : bmiFromWeightHeight(Number(weight), Number(height))
  const invalid = bmi !== null && (bmi < field.valid_min || bmi > field.valid_max)
  const fallbackWarning =
    bmi !== null && field.train_min != null && (bmi < field.train_min || bmi > field.train_max)
      ? outsideRangeNotice()
      : ''
  const shownError = error || (invalid ? rangeMessage(field) : '')

  return (
    <div className="field">
      <label className="field-question required">{copy.question}</label>
      <div className="input-row">
        <label htmlFor="weight">Weight</label>
        <input
          id="weight"
          name="weight"
          type="number"
          min={25}
          max={300}
          step="any"
          value={weight}
          onChange={(event) => onValue('weight', event.target.value)}
        />
        <span className="unit">| kg</span>
        <label htmlFor="height">Height</label>
        <input
          id="height"
          name="height"
          type="number"
          min={100}
          max={230}
          step="any"
          value={height}
          onChange={(event) => onValue('height', event.target.value)}
        />
        <span className="unit">| cm</span>
      </div>
      {bmi !== null && <span className="helper">Your BMI: {bmi}</span>}
      <Warning text={warningText || fallbackWarning} />
      {shownError && <span className="error">{shownError}</span>}
    </div>
  )
}

export function TobaccoInput({ field, copy, values, onValue, error, warningText }) {
  if (!copy) throw new Error(`Missing UI copy for field ${field.name}`)
  const status = values.smokingStatus ?? ''
  const cigarettes = values.cigarettesPerDay ?? ''
  const years = values.yearsSmoked ?? ''
  const tobacco =
    status === 'never'
      ? 0
      : status === '' || cigarettes === '' || years === ''
        ? null
        : cumulativeTobaccoKg({
            status,
            cigarettesPerDay: Number(cigarettes),
            yearsSmoked: Number(years),
          })
  const showAmount = status !== '' && status !== 'never'
  const cigarettesQuestion =
    status === 'former'
      ? 'How many cigarettes per day did you smoke?'
      : 'How many cigarettes per day?'
  const fallbackWarning =
    field.train_min != null && tobacco !== null && (tobacco < field.train_min || tobacco > field.train_max)
      ? outsideRangeNotice()
      : ''

  return (
    <div className="field">
      <label className="field-question required" htmlFor="smokingStatus">
        {copy.question}
      </label>
      <div className="input-row">
        <select
          id="smokingStatus"
          name="smokingStatus"
          value={status}
          onChange={(event) => onValue('smokingStatus', event.target.value)}
        >
          <option value="">Select...</option>
          {copy.options.map((option) => (
            <option key={String(option.value)} value={String(option.value)}>
              {option.label}
            </option>
          ))}
        </select>
      </div>
      {showAmount && (
        <>
          <div className="input-row">
            <label htmlFor="cigarettesPerDay">{cigarettesQuestion}</label>
            <input
              id="cigarettesPerDay"
              name="cigarettesPerDay"
              type="number"
              min={0}
              max={100}
              step="any"
              value={cigarettes}
              onChange={(event) => onValue('cigarettesPerDay', event.target.value)}
            />
          </div>
          <div className="input-row">
            <label htmlFor="yearsSmoked">For how many years?</label>
            <input
              id="yearsSmoked"
              name="yearsSmoked"
              type="number"
              min={0}
              max={80}
              step={1}
              value={years}
              onChange={(event) => onValue('yearsSmoked', event.target.value)}
            />
          </div>
        </>
      )}
      {tobacco !== null && (
        <span className="helper">
          Estimated lifetime tobacco: {tobacco} kg (about {GRAMS_PER_CIGARETTE} g per cigarette)
        </span>
      )}
      <Warning text={warningText || fallbackWarning} />
      {error && <span className="error">{error}</span>}
    </div>
  )
}

export function FieldRenderer({ field, copy, values, errors, onValue, warningText }) {
  const error = errors[field.name]
  if (isBmiField(field)) {
    return (
      <BmiInput
        field={field}
        copy={copy}
        values={values}
        onValue={onValue}
        error={error}
        warningText={warningText}
      />
    )
  }
  if (isTobaccoField(field)) {
    return (
      <TobaccoInput
        field={field}
        copy={copy}
        values={values}
        onValue={onValue}
        error={error}
        warningText={warningText}
      />
    )
  }
  return (
    <FieldInput
      field={field}
      copy={copy}
      value={values[field.name]}
      error={error}
      onChange={(value) => onValue(field.name, value)}
      warningText={warningText}
    />
  )
}
