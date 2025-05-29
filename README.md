# 🧠 Mental Health Risk Prediction Using Machine Learning

This project focuses on predicting mental health risks using machine learning. By analyzing key lifestyle, demographic, and psychological factors, it helps identify individuals who might be at risk of developing mental health issues.

---

## 📌 Project Overview

Mental health has become a critical global concern. This project aims to build a predictive model that classifies individuals based on their risk of mental health issues using features related to their behavior, emotional state, and support systems.

---

## 🗂️ Dataset Features

The dataset includes the following 14 features:

- `age`: Age of the individual
- `gender`: Gender identity
- `employment_status`: Current employment status
- `work_environment`: Quality and type of work environment
- `mental_health_history`: Whether the person has a history of mental illness
- `seeks_treatment`: Willingness or current status of seeking treatment
- `stress_level`: Self-reported stress level
- `sleep_hours`: Average sleep per night
- `physical_activity_days`: Number of days involved in physical activity per week
- `depression_score`: Score from depression screening
- `anxiety_score`: Score from anxiety screening
- `social_support_score`: Level of social support received
- `productivity_score`: Self-rated productivity
- `mental_health_risk`: Target label (e.g., Low, Medium, High risk)

---

## 🧰 Technologies & Libraries

- **Python 3**
- **Label Encoding** – for handling categorical features
- **Pipeline** – to streamline preprocessing and modeling
- **StandardScaler** – to normalize numerical features
- **RandomForestClassifier** – the main model used
- **GridSearchCV** – for exhaustive hyperparameter tuning
- **Optuna** – for efficient, automated hyperparameter optimization
- **K-Nearest Neighbors (KNN)** – used for performance comparison

---

## 🧠 Model Architecture

### Pipeline Setup

```python
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('random', RandomForestClassifier())
])
