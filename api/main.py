from fastapi import FastAPI
from api.schemas import SentimentRequest, SentimentResponse
import joblib
import re
import nltk
from nltk.corpus import stopwords
import time
from api.logger import logger
from monitoring.drift import update_metrics, detect_drift
import threading
from retraining.retrain import retrain_model


# Download if missing
nltk.download("stopwords")

app = FastAPI(title="Sentiment Monitoring API")

# Load artifacts
model = joblib.load("model/sentiment_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

stop_words = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(tokens)


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=SentimentResponse)
def predict(request: SentimentRequest):
    start_time = time.time()

    try:
        clean = clean_text(request.text)
        vec = vectorizer.transform([clean])

        probs = model.predict_proba(vec)[0]
        pred = probs.argmax()
        confidence = float(probs[pred])
        sentiment = "positive" if pred == 1 else "negative"

        update_metrics(len(request.text), confidence)
        drift_status = detect_drift()
        # 🔁 Trigger retraining asynchronously if drift detected
        if drift_status.get("drift"):
            threading.Thread(
                target=retrain_model,
                args=("drift_detected",),
                daemon=True
            ).start()

        latency = round(time.time() - start_time, 4)

        logger.info(
            f"text_len={len(request.text)} | "
            f"sentiment={sentiment} | "
            f"confidence={confidence:.3f} | "
            f"latency={latency}s"
        )

        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 3),
            "drift": drift_status
        }

    except Exception as e:
        logger.error(f"prediction_failed | error={str(e)}")
        raise




"""✅ Correct Way to Test the API
✔️ Option 1 — Swagger UI (Recommended)

Open this in browser:

http://127.0.0.1:8000/docs


Then:

Click POST /predict

Click Try it out

Enter:

{
  "text": "This movie was amazing and wonderful"
}


Click Execute

You should get response like:{
  "sentiment": "positive",
  "confidence": 0.91
}

When demoing your project (interview / GitHub):

Show:
✅ Positive case
✅ Negative case
✅ Edge case

This demonstrates robustness 💪

This log output is exactly what a production ML API should generate.

You’ve successfully implemented:

✅ Request logging
✅ Latency tracking
✅ Prediction traceability
✅ Confidence monitoring
✅ Persistent storage

This is real MLOps behavior 👏

Example from your log:

text_len=219 | sentiment=negative | confidence=0.938 | latency=0.001s


That means you can analyze:

How long predictions take

What kind of inputs users send

Prediction confidence distribution

Traffic patterns

I implemented rolling-window drift detection based on input statistics and confidence degradation.

"""