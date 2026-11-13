import { describe, expect, it } from 'vitest'
import { GRAMS_PER_CIGARETTE, bmiFromWeightHeight, cumulativeTobaccoKg } from './derived'

describe('bmiFromWeightHeight', () => {
  it('computes BMI from weight and height rounded to one decimal', () => {
    expect(bmiFromWeightHeight(70, 175)).toBe(22.9)
  })
})

describe('cumulativeTobaccoKg', () => {
  it('is zero for never smokers', () => {
    expect(
      cumulativeTobaccoKg({ status: 'never', cigarettesPerDay: 20, yearsSmoked: 10 }),
    ).toBe(0)
  })

  it('computes lifetime tobacco for current smokers', () => {
    expect(
      cumulativeTobaccoKg({ status: 'current', cigarettesPerDay: 20, yearsSmoked: 10 }),
    ).toBe(73)
  })

  it('computes lifetime tobacco for former smokers', () => {
    expect(
      cumulativeTobaccoKg({ status: 'former', cigarettesPerDay: 5, yearsSmoked: 5 }),
    ).toBe(9.1)
  })

  it('treats former and current smokers the same for identical inputs', () => {
    const former = cumulativeTobaccoKg({
      status: 'former',
      cigarettesPerDay: 15,
      yearsSmoked: 20,
    })
    const current = cumulativeTobaccoKg({
      status: 'current',
      cigarettesPerDay: 15,
      yearsSmoked: 20,
    })
    expect(former).toBe(current)
  })

  it('exposes the grams-per-cigarette assumption', () => {
    expect(GRAMS_PER_CIGARETTE).toBe(1)
  })
})
