# Phase 1 — Dataset Research

Research summary for all three diseases.

## Disease 1: Type 2 Diabetes

### Candidate Datasets
- [x] Diabetes Health Dataset Analysis
- [ ] ARIC Study
- [ ] Other: ___________

### Selected Dataset: Diabetes Health Dataset Analysis

#### 1. Access & Licensing
- **Where found:** Kaggle
- **Access requirements:** Free (no institutional approval needed)
- **Free/Cost:** Free
- **Academic use permitted:** YES (Kaggle CC BY 4.0 license)

#### 2. Data Dictionary
- **Link to documentation:** https://www.kaggle.com/datasets/rabieelkharoua/diabetes-health-dataset-analysis
- **Total variables:** 46
- **Sample size:** 1,879 records
- **Note:** Cross-sectional dataset (current diagnosis status, not future prediction)

#### 3. Family History Variables
- **Variables found:**
  - [x] `FamilyHistoryDiabetes` (0=No, 1=Yes)
  - Distribution: No=1,431 (76%), Yes=448 (24%)

#### 4. Future Outcome
- **Target variable:** `Diagnosis` (0=No diabetes, 1=Diabetes)
- **Is it a future outcome?** NO — Cross-sectional (current status at single time point)
- **Outcome distribution:** No=1,127 (60%), Yes=752 (40%)
- **Class balance:** Moderate (40% positive class)

#### 5. Data Quality
- **Missing values:** 0 (100% complete dataset)
- **Class balance:** Acceptable (60-40 split)
- **Age range:** 20-90 years
- **Additional features:** 45 variables including BMI, blood pressure, cholesterol, lifestyle factors, symptoms

#### 6. Decision
- **Feasible for MVP:** YES
- **Limitation:** Cross-sectional, not longitudinal future-risk prediction
- **Mitigation:** Clearly documented in academic report as MVP limitation
- **Next steps:** Data preprocessing, model training, evaluation

---

## Disease 2: Cardiovascular Disease

### Candidate Datasets
- [x] Coronary Heart Disease (CHDdata)
- [ ] Framingham Heart Study
- [ ] Other: ___________

### Selected Dataset: Coronary Heart Disease

#### 1. Access & Licensing
- **Where found:** Kaggle
- **Access requirements:** Free (no institutional approval needed)
- **Free/Cost:** Free
- **Academic use permitted:** YES

#### 2. Data Dictionary
- **Link to documentation:** https://www.kaggle.com/datasets/billbasener/coronary-heart-disease
- **Total variables:** 10
- **Sample size:** 462 records
- **Note:** Cross-sectional dataset (current diagnosis status)

#### 3. Family History Variables
- **Variables found:**
  - [x] `famhist` (Absent/Present)
  - Distribution: Absent=270 (58%), Present=192 (42%)

#### 4. Future Outcome
- **Target variable:** `chd` (0=No CHD, 1=CHD present)
- **Is it a future outcome?** NO — Cross-sectional (current status at single time point)
- **Outcome distribution:** No=302 (65%), Yes=160 (35%)
- **Class balance:** Moderate (35% positive class)

#### 5. Data Quality
- **Missing values:** 0 (100% complete dataset)
- **Class balance:** Acceptable (65-35 split)
- **Age range:** 15-64 years
- **Variables included:** systolic BP, tobacco use, LDL, adiposity, type A personality, obesity, alcohol, age, family history

#### 6. Decision
- **Feasible for MVP:** YES
- **Limitation:** Cross-sectional, not longitudinal future-risk prediction
- **Mitigation:** Clearly documented in academic report as MVP limitation
- **Next steps:** Data preprocessing, model training, evaluation

---

## Disease 3: Alzheimer's / Dementia

### Candidate Datasets
- [x] Kaggle Alzheimer's Dataset
- [ ] PREVENT-AD
- [ ] WRAP (Wisconsin Registry)
- [ ] Other: ___________

### Selected Dataset: Alzheimer's Disease Dataset (Kaggle)

#### 1. Access & Licensing
- **Where found:** Kaggle
- **Access requirements:** Free (no institutional approval needed)
- **Free/Cost:** Free
- **Academic use permitted:** To be verified on the source page
- **Licence and provenance:** To be verified on the source page

#### 2. Data Dictionary
- **Link to documentation:** Kaggle Alzheimer's Disease Dataset page
- **Total variables:** 35 (34 predictors and `Diagnosis`)
- **Sample size:** 2,149 records
- **Note:** Cross-sectional dataset (current diagnosis status, not future prediction)

#### 3. Family History Variables
- **Variables found:**
  - [x] `FamilyHistoryAlzheimers` (0=No, 1=Yes)
  - Distribution: No=1,607 (75%), Yes=542 (25%)

#### 4. Future Outcome
- **Target variable:** `Diagnosis` (0=No Alzheimer's disease, 1=Alzheimer's disease)
- **Is it a future outcome?** NO — Cross-sectional (current status at a single time point)
- **Outcome distribution:** No=1,389 (65%), Yes=760 (35%)
- **Class balance:** Moderate (35% positive class)

#### 5. Data Quality
- **Missing values:** 0 in every column
- **Class balance:** Acceptable (65-35 split)
- **Age range:** 60-90 years
- **Education level:** 0-3 (numeric codes; meaning to be verified on the source page)
- **Ethnicity:** 0-3 (categorical codes; one-hot encoded in the models)
- **Identifiers and constants:** `PatientID` is unique per row; `DoctorInCharge` is constant (`XXXConfid`) and is dropped
- **Other columns:** gender, lifestyle, cardiovascular and comorbidity indicators, cholesterol, MMSE, and cognitive/functional scores
- **Synthetic-data indicators:** MMSE is nearly evenly spread from 0 to 30 (six bins contain 326-390 rows each), and many continuous columns have one unique value per row. The dataset appears synthetic.

#### 6. Decision
- **Feasible for this project?** YES
- **Why/Why not:** It has a diagnosis target and a family-history variable, but it is cross-sectional and appears synthetic, so results do not describe real patients
- **Next steps:** Preprocessing, baseline models, MLP, family-history analysis and honest reporting of the limitations

---

## Summary

| Disease | Dataset | Access | Family History | Size | Outcome Type | Status |
|---------|---------|--------|-----------------|------|--------------|--------|
| Diabetes | Diabetes Health Dataset | Free ✅ | Yes ✅ | 1,879 | Cross-sectional | Ready ✅ |
| CVD | Coronary Heart Disease | Free ✅ | Yes ✅ | 462 | Cross-sectional | Ready ✅ |
| Alzheimer's | TBD | - | - | - | - | Pending |

---

## Phase 1 Outcome

**Two datasets selected and validated for MVP:**
- ✅ Diabetes dataset: 1,879 records, 46 features, family history present
- ✅ CVD dataset: 462 records, 10 features, family history present

**MVP Limitations (documented):**
- Cross-sectional data (current diagnosis, not future risk prediction)
- No longitudinal follow-up outcomes
- These datasets are suitable for demonstrating deep learning capability
- Future versions should integrate true prospective cohorts (Framingham, ARIC, PREVENT-AD)

**Ready to proceed to Phase 2: Scientific Design**
## Data caveats
- The Diabetes Health Dataset is synthetic: its Kaggle description says it was generated for educational purposes. Results do not reflect real patients.
- The CHD dataset should be checked for provenance on its Kaggle page before making claims about it.
- Both datasets are cross-sectional (current diagnosis), so models estimate current risk, not future risk.
