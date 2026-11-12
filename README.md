# 🎓 University Student Dropout Early Warning System
> **Machine Learning Zoomcamp 2026 — Midterm Project**  
> *An End-to-End Machine Learning Solution for Proactive Student Retention and Dropout Risk Mitigation*

---

## 📌 1. Project Overview & Problem Statement

### 1.1. Project Summary
Student retention is a fundamental benchmark of institutional quality, social mobility, and financial sustainability in higher education. Each year, universities experience significant student attrition, where students drop out before completing their degrees due to a combination of academic friction, demographic hurdles, and financial distress.

This project delivers an **end-to-end Machine Learning Early Warning System** that evaluates first-year student data to forecast the likelihood of student attrition (`Dropout = 1` vs `Graduate = 0`). By accurately identifying at-risk students within their first two semesters, the system enables academic advisors and university administrators to execute early, targeted interventions—such as personalized tutoring and emergency tuition aid—before permanent withdrawal occurs.

### 1.2. Problem Formulation
- **Objective:** Predict binary student outcome (`1` for Dropout, `0` for Graduate) using early academic evaluations and socio-demographic features.
- **Actionable Business Impact:** Minimize false negatives (failing to identify a student who will drop out) while avoiding excessive false alarms to optimize counselor workload and financial aid distribution.
- **Primary Metric:** **ROC-AUC** is selected as the primary optimization and benchmarking metric to evaluate probability discrimination across all decision thresholds, complemented by **Recall** and **F1-Score** to monitor sensitivity on at-risk students.

---

## 🏗️ 2. Architecture & Workflow Diagram

```text
+-------------------+      +-------------------------+      +------------------------+
|   Raw Data        | ---> | Data Cleaning & EDA     | ---> | Validation & Tuning    |
|   (UCI data.csv)  |      | Filter 'Enrolled' (60/40)|      | (Stratified 60/20/20   |
+-------------------+      | MI & Corr Analysis      |      |  5-Fold GridSearchCV)  |
+-------------------+      +-------------------------+      +------------------------+
                                                                         |
                                                                         v
+-------------------+      +-------------------------+      +------------------------+
|  Docker / Cloud   | <--- | Web Service API         | <--- | Final Model Artifact   |
|  (Render.com API) |      | (FastAPI `predict.py`)  |      | (`models/model.bin`)   |
+-------------------+      +-------------------------+      +------------------------+
```

---

## 📊 3. Dataset Description

The project utilizes the **Predict Students' Dropout and Academic Success** dataset from the **UCI Machine Learning Repository**:
* **Source:** [UCI Machine Learning Repository (Dataset ID: 697)](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
* **Original Dimensions:** 4,424 observations across 37 attributes (socio-economic, demographic, academic records, and macro indicators).
* **Target Filtering:** To address the clear dichotomy between attrition and graduation, active (`Enrolled`) students are excluded, yielding **3,630 clean observations**:
  * `Graduate (0)`: 2,209 students (60.85%)
  * `Dropout (1)`: 1,421 students (39.15%)
  * *Class Ratio:* ~1.55 : 1 (mild imbalance; preserved via stratified splitting).

The clean dataset is committed directly to the repository at [`data/data.csv`](data/data.csv), ensuring immediate, out-of-the-box reproducibility without any external network dependency.

---

## 🔍 4. Exploratory Data Analysis (EDA) Summary

Key empirical findings from the exploratory analysis in [`notebooks/notebook.ipynb`](notebooks/notebook.ipynb):

1. **Financial Arrears as the Primary Early Warning Signal:**
   - `tuition_fees_up_to_date`: Students behind on tuition payments suffer an overwhelming **94.0% dropout rate** (vs. 30.7% for students whose fees are paid). This variable exhibits the highest Mutual Information score (**0.105**).
   - `debtor`: Students with outstanding institutional debt exhibit a **75.5% dropout rate** (vs. 34.5% for non-debtors).
   - `scholarship_holder`: Scholarship recipients demonstrate high persistence, dropping out at only **13.8%** (vs. 48.4% for non-recipients).
2. **First-Year Academic Performance:**
   - 1st and 2nd semester approved units (`curricular_units_1st_sem_approved`, `curricular_units_2nd_sem_approved`) show the strongest negative correlations with dropout ($r = -0.55$ to $-0.65$).
   - Graduating students pass an average of **6.2 units** in their first semester with a grade average of **12.6**, compared to only **2.5 units** and a grade average of **7.3** for dropouts.
3. **Demographic & Contextual Dynamics:**
   - `age_at_enrollment`: Positive correlation ($r = +0.27$). Dropouts enroll at an older average age (**26.1 years**) than graduates (**21.8 years**), reflecting work and family obligations among mature students.
   - Male students display higher attrition rates (**56.1%**) compared to female students (**30.2%**).
4. **Data Hygiene & Outlier Handling:**
   - **Missing Values & Duplicates:** Verified zero missing cells and zero duplicate rows.
   - **Outlier Retention:** Extreme values (e.g. students aged up to 70 and 0 approved credits) are intentionally preserved without truncation. These represent genuine, highly vulnerable student segments rather than measurement errors; trimming them would introduce survivorship bias.

---

## 🤖 5. Modeling Approach & Evaluation Metrics

### 5.1. Model Exploration & 5-Fold GridSearch Tuning
Three distinct model families were tuned on the training partition ($N = 2,178$) using 5-Fold Cross-Validation optimizing for ROC-AUC:

| Model Family | Algorithm | Tuned Hyperparameters | Validation ROC-AUC |
|---|---|---|:---:|
| **Linear Baseline** | Logistic Regression (`liblinear`) | `C: 0.1` | **0.9455** |
| **Tree Ensemble (Bagging)** | Random Forest Classifier | `max_depth: 15, n_estimators: 150` | **0.9450** |
| **Tree Ensemble (Boosting)** | **XGBoost Classifier (Champion)** | `learning_rate: 0.05, max_depth: 5, n_estimators: 100` | **0.9436 (CV: 0.9492)** |

### 5.2. Final Test Set Evaluation
The champion **XGBoost** model was retrained on the complete training set (`Train + Validation`, $N = 2,904$) and evaluated on the untouched Test set ($N = 726$):

| Evaluation Metric | Test Set Score | Practical Business Interpretation |
|---|:---:|---|
| **ROC-AUC Score** | **0.9744** | Outstanding discriminative ability across all probability cutoffs |
| **Accuracy** | **92.29%** | Correctly predicts student status for >92% of the test cohort |
| **Precision (Dropout)**| **91.30%** | When flagging a student as at-risk, the model is correct 91% of the time |
| **Recall (Sensitivity)**| **88.73%** | **Successfully flags ~89% of all actual dropouts for intervention** |
| **F1-Score** | **0.9000** | Exceptional harmonic balance between precision and recall |

---

## 💻 6. How to Run Locally

### 6.1. Environment Setup
```bash
# 1. Clone repository
git clone https://github.com/BagusAri-17/ml-zoomcamp-midterm-project-2026-by-datatalks.club.git
cd ml-zoomcamp-midterm-project-2026-by-datatalks.club

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate       # On Windows: .\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 6.2. Train Model Script (`train.py`)
Run the training script to retrain the champion model and serialize artifacts:
```bash
python train.py
```
*(Template script will generate `models/model.bin`)*

### 6.3. Launch Web Service Locally (`predict.py`)
Run the API web service using Uvicorn:
```bash
uvicorn predict:app --host 0.0.0.0 --port 9696 --reload
```
The API documentation will be available at `http://localhost:9696/docs`.

---

## 🐳 7. How to Run via Docker

To build and run the service inside a container:

```bash
# 1. Build Docker image
docker build -t student-dropout-service:v1 .

# 2. Run container
docker run -it --rm -p 9696:9696 student-dropout-service:v1
```

---

## 🧪 8. API Usage Example & Testing

### 8.1. Testing via Python Script
In a separate terminal, test the service with:
```bash
python test_service.py
```

### 8.2. Testing via cURL
Send a sample inference request:
```bash
curl -X POST "http://localhost:9696/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "marital_status": 1,
       "application_mode": 1,
       "course": 9254,
       "daytime_evening_attendance": 1,
       "previous_qualification": 1,
       "previous_qualification_grade": 120.0,
       "nacionality": 1,
       "mother_s_qualification": 1,
       "father_s_qualification": 1,
       "mother_s_occupation": 5,
       "father_s_occupation": 5,
       "admission_grade": 115.0,
       "displaced": 0,
       "educational_special_needs": 0,
       "debtor": 1,
       "tuition_fees_up_to_date": 0,
       "gender": 1,
       "scholarship_holder": 0,
       "age_at_enrollment": 27,
       "international": 0,
       "curricular_units_1st_sem_credited": 0,
       "curricular_units_1st_sem_enrolled": 6,
       "curricular_units_1st_sem_evaluations": 6,
       "curricular_units_1st_sem_approved": 1,
       "curricular_units_1st_sem_grade": 8.0,
       "curricular_units_1st_sem_without_evaluations": 0,
       "curricular_units_2nd_sem_credited": 0,
       "curricular_units_2nd_sem_enrolled": 6,
       "curricular_units_2nd_sem_evaluations": 6,
       "curricular_units_2nd_sem_approved": 0,
       "curricular_units_2nd_sem_grade": 0.0,
       "curricular_units_2nd_sem_without_evaluations": 0,
       "unemployment_rate": 10.8,
       "inflation_rate": 1.4,
       "gdp": 1.74
     }'
```

**Example Output:**
```json
{
  "dropout_probability": 0.9421,
  "dropout_prediction": true,
  "risk_level": "High Risk",
  "recommendation": "Priority academic counselor outreach and emergency tuition aid review recommended."
}
```

---

## ☁️ 9. Cloud Deployment

The containerized service will be deployed to a cloud platform (e.g., Render.com / Fly.io).

* **Live Base URL:** `https://<your-service-name>.onrender.com`
* **Health Check:** `GET /health`
* **Inference Endpoint:** `POST /predict`

### Verification Proof (Screenshot):
*(A screenshot of the successful prediction test against the live cloud endpoint will be added below)*:

![Cloud Deployment Test Proof](docs/cloud_test_proof.png)

---

## ⚠️ 10. Known Limitations & Next Steps

1. **Current Limitations:**
   - **Static Historical Snapshot:** The model currently consumes fixed Semester 1 & 2 aggregations; it does not capture dynamic week-by-week LMS engagement (e.g. Moodle login frequency or assignment submission delays).
   - **Threshold Dependency:** Default operational decision boundaries use $P = 0.5$. Institutional deployment should calibrate this cutoff based on counseling department capacity.
2. **Future Enhancements:**
   - Integrate time-series course interaction telemetry to detect disengagement mid-semester.
   - Build a lightweight frontend dashboard (Streamlit / React) for academic advisors to inspect batch student cohorts.

