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
- [ ] PREVENT-AD
- [ ] WRAP (Wisconsin Registry)
- [ ] Kaggle Alzheimer's Dataset
- [ ] Other: ___________

### Selected Dataset: TBD (Not yet downloaded)

#### 1. Access & Licensing
- **Where found:** 
- **Access requirements:** 
- **Free/Cost:** 
- **Academic use permitted:** 

#### 2. Data Dictionary
- **Link to documentation:** 
- **Total variables:** 
- **Sample size:** 
- **Follow-up duration:** 

#### 3. Family History Variables
- **Variables found:** TBD

#### 4. Future Outcome
- **Target variable:** TBD
- **Is it a future outcome?** TBD
- **Prediction horizon:** TBD

#### 5. Data Quality
- **Missing values:** TBD
- **Class balance:** TBD
- **Baseline population characteristics:** TBD

#### 6. Decision
- **Feasible for this project?** TBD
- **Why/Why not:** TBD
- **Next steps:** TBD

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