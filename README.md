<p align="center">
  <img src="https://raw.githubusercontent.com/mylethidiem/Heart-Sentinel/refs/heads/main/static/heart_sentinel_background.png" alt="Heart Sentinel Banner" width="100%">
</p>


### *Early Detection. Smarter Health Decisions.*

Heart Sentinel is an intelligent health-monitoring and early-warning system designed to **analyze cardiovascular signals**, **predict health risks**, and **provide personalized lifestyle guidance**.
The system integrates **machine learning**, **risk prediction**, **chatbot health coaching**, and is designed to **extend into real-time wearable data** for continuous health monitoring.

---
### 🎬 Demo
- [You can run the Hugging Face demo here](https://elizabethmyn-heart-sentinel.hf.space/)

# 📌 **Key Features**

### 🔎 **1. Heart Disease Diagnosis**

* Predicts the likelihood of heart-related conditions using clinical and demographic features.
* Supports explainability (XAI) to show *why* the model makes a prediction.

### 🩸 **2. Cholesterol Level Prediction**

* Regression model forecasting cholesterol based on health metrics and lifestyle indicators.

### ⚠️ **3. Stroke Risk Assessment**

* Identifies high-risk patterns early using medical datasets and statistical indicators.
* Designed to alert users before symptoms escalate.

### 🔔 **4. Smart Health Alerts**

* Instant warnings for abnormal metrics (e.g., elevated heart rate, risk spikes).
* Ideal for future IoT sensor and wearable integration.

### 🤖 **5. Health Advisory Chatbot**

* Provides recommendations on:

  * diet
  * exercise
  * lifestyle habits
  * early preventive care
* Tailored guidance based on the user’s health profile.

### ⌚ **6. Wearable Data Support (Future Extension)**

* Real-time tracking from smartwatches or fitness bands (heart rate, SPO2, sleep).
* Event-driven alerts when irregular patterns appear.

---

# 🧠 **Technical Overview**

### **📊 Machine Learning Models**

| Module                  | Model Used                                     | Goal           |
| ----------------------- | ---------------------------------------------- | -------------- |
| Heart Disease Diagnosis | Logistic Regression / Random Forest / XGBoost  | Classification |
| Cholesterol Prediction  | Linear Regression / XGBoost Regressor          | Regression     |
| Stroke Warning          | Random Forest / Gradient Boosting              | Classification |
| Advisory Chatbot        | Retrieval-based system / RAG (optional future) | Guidance       |

---

### **📈 Explainable AI (XAI)**

Heart Sentinel incorporates XAI features such as:

* **SHAP values**
* **Feature importance**
* **Decision path visualization**

This helps users and healthcare professionals understand how predictions were made.

---

# 🧩 **System Architecture (High-Level)**

```
                 ┌───────────────────┐
                 │   User Input /    │
                 │   Wearable Data   │
                 └─────────┬─────────┘
                           │
                    Data Preprocessing
                           │
               ┌───────────┴───────────┐
               │ ML Risk Prediction     │
               │ (Heart, Cholesterol,   │
               │  Stroke Models)        │
               └───────────┬───────────┘
                           │
                 Smart Alerts Engine
                           │
               ┌───────────┴───────────┐
               │  Health Advisory Chatbot│
               └───────────┬───────────┘
                           │
                      Recommendations
```

---

# 🛠️ **Technology Stack**

### **Languages**

* Python
* (Future) Kotlin/Swift for mobile app
* (Future) JavaScript for web dashboard

### **Libraries**

* NumPy, Pandas
* Scikit-learn
* XGBoost, LightGBM
* Matplotlib, Seaborn
* SHAP / LIME for XAI
* FastAPI (for backend, optional)

### **Tools**

* Git & GitHub
* Jupyter Notebook
* Kaggle Datasets
* DVC (optional)

---

# 📂 **Project Structure**

```
Heart-Sentinel/
│
├── data/                # Datasets
├── notebooks/           # ML experiments & EDA
├── models/              # Trained models
├── src/
│   ├── preprocessing/   # Data cleaning & feature engineering
│   ├── prediction/      # ML model scripts
│   ├── alerts/          # Rules & anomaly detection
│   ├── chatbot/         # Health recommendation engine
│   └── api/             # (optional) FastAPI endpoints
│
├── xai/                 # SHAP or LIME explanations
├── README.md
└── requirements.txt
```

---

# 📈 **Planned Enhancements**

* 🩺 Integration with Fitbit/Garmin/Apple Watch
* 🧬 Multi-sensor fusion (HRV, ECG, sleep cycles)
* 📱 Mobile app with real-time monitoring
* 🧠 RAG-powered Health Coaching chatbot
* 🔐 Privacy-preserving ML (Federated, DP-SGD)
* 🏥 Deployment-ready Clinical Dashboard

---

# 🧪 **How to Run**

```bash
# 1. Clone repo
git clone https://github.com/yourusername/heart-sentinel.git
cd heart-sentinel

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run notebooks or ML modules
jupyter notebook
```

---

# 📃 **Relate work**
- My mini project about [Heart Disease Diagnosis](https://github.com/mylethidiem/data-science-artificial-intelligence-projects/tree/main/Heart-Disease-Diagnosis)
- [My HuggingFace Space for Heart Sentinel](https://huggingface.co/spaces/elizabethmyn/Intelligent-Retail-Decision-Making-System)
- [My Note for this project](https://www.notion.so/Heart-Disease-Diagnosis-2a40730a967380689b87eeb26a447b72)
  
---

# 👩‍⚕️ **About the Author**

**Lê Thị Diễm My**
AI & Data Science Learner

* Specializing in Machine Learning, Time Series, and Explainable AI
* Interested in AI-for-Health and Human-Centered ML

**👩‍💻 Author:** [Lê Thị Diễm My](https://github.com/mylethidiem)
📧 **Email:** lethidiemmy961996@gmail.com
🔗 **LinkedIn:** [Thi-Diem-My Le](https://www.linkedin.com/in/mylethidiem/)

______________________________________________________________________

> _"Learning, Building, and Growing in Data & AI."_ 🌍

