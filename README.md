# MultiRisk AI

Intelligent Multi-Disease Risk Assessment System Based on Deep Learning

## Overview

MultiRisk AI predicts **current disease risk** based on family history, clinical markers, and demographic factors.

Given a patient's:
- Family history (parent/sibling had disease)
- Clinical measurements (blood pressure, cholesterol, BMI, etc.)
- Demographic info (age, gender, lifestyle)

The system estimates their **risk level** for:
1. **Type 2 Diabetes**
2. **Cardiovascular Disease**
3. ~~Alzheimer's Disease~~ (pending dataset)

## Key Features

- Family history is **mandatory input**
- Disease-specific risk models
- Clear risk stratification (low/medium/high)
- Explains which factors drive the risk prediction

## Note

This MVP uses cross-sectional data to assess **current disease risk**. Future versions will integrate longitudinal cohorts for true future-risk prediction.

## Project Structure

- `data/` — Datasets (diabetes, cardiovascular)
- `src/` — Source code
- `notebooks/` — Analysis and research
- `models/` — Trained models
- `docs/` — Documentation