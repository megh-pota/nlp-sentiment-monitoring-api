# 🚀 NLP Sentiment Monitoring API

A production-style NLP Sentiment Monitoring system built with FastAPI that performs real-time inference, logging, drift detection, and automated retraining simulation.

This project demonstrates how machine learning models can be deployed, monitored, and continuously improved in a production-like environment.

---

## 📌 Key Features

* ✅ FastAPI-based inference API
* ✅ Real-time latency and confidence logging
* ✅ Rolling-window drift detection
* ✅ Automated retraining trigger (async)
* ✅ Model versioning and registration
* ✅ Payload validation and API hardening
* ✅ Production-style monitoring architecture

---

## 🏗️ System Architecture

Client
→ FastAPI Inference API
→ Text Preprocessing
→ Vectorizer + ML Model
→ Prediction + Confidence
→ Metrics Logging
→ Drift Detection
→ Automated Retraining
→ Model Registry

(Architecture diagram included in repository.)

---

## ⚙️ Tech Stack

* Python
* FastAPI
* scikit-learn
* NLTK
* Joblib
* Threading
* Logging

---

## 📓 Training Notebook
* Model training and preprocessing are documented in /notebooks/sentiment_training.ipynb.

---
## 📂 Project Structure

```
.
├── api/
│   ├── main.py
│   ├── schemas.py
│   └── logger.py
├── model/
│   ├── sentiment_model.pkl
│   └── vectorizer.pkl
├── monitoring/
│   └── drift.py
├── retraining/
│   └── retrain.py
└── README.md
```

---

## ▶️ Running Locally

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Start the API

```bash
uvicorn api.main:app --reload
```

3. Open Swagger UI

```
http://localhost:8000/docs
```

---

## 🧪 Example Request

```json
{
  "text": "I really love this product. It works perfectly."
}
```

---

## 📤 Example Response

```json
{
  "sentiment": "positive",
  "confidence": 0.93,
  "drift": {
    "drift": false
  }
}
```

---

## 📊 Drift Monitoring Logic

* Tracks rolling statistics of:

  * Input text length
  * Prediction confidence
* Compares live distribution against baseline window
* Flags drift when deviation crosses threshold
* Automatically triggers retraining pipeline

---

## 🔁 Automated Retraining

When drift is detected:

* A background thread triggers retraining
* A new model version is trained
* Model is registered and versioned
* System continues serving traffic

Concurrency protection prevents duplicate retraining jobs.

---

## 🔒 Production Considerations

* JSON payload validation
* Size limits for text input
* Logging for observability
* Thread-safe retraining execution
* Modular design for scalability

---

## 🌱 Future Improvements

* Prometheus metrics integration
* Model performance dashboards
* Canary deployments
* Shadow testing
* Cloud deployment
* CI/CD pipeline automation

---

## 👨‍💻 Author

Built as a portfolio project to demonstrate real-world MLOps and backend ML engineering skills.
