export const GRAMS_PER_CIGARETTE = 1

export function bmiFromWeightHeight(weightKg, heightCm) {
  return Math.round((weightKg / (heightCm / 100) ** 2) * 10) / 10
}

export function cumulativeTobaccoKg({ status, cigarettesPerDay, yearsSmoked }) {
  if (status === 'never') return 0
  const kg = (cigarettesPerDay * 365 * yearsSmoked * GRAMS_PER_CIGARETTE) / 1000
  return Math.round(kg * 10) / 10
}
