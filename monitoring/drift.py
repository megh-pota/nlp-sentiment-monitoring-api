import numpy as np
from collections import deque

WINDOW_SIZE = 200

text_lengths = deque(maxlen=WINDOW_SIZE)
confidences = deque(maxlen=WINDOW_SIZE)

BASELINE_TEXT_LEN = 120
BASELINE_CONFIDENCE = 0.85


def update_metrics(text_len: int, confidence: float):
    text_lengths.append(text_len)
    confidences.append(confidence)


def detect_drift():
    if len(text_lengths) < 1:   # show immediately for debugging
        return {"drift": False, "samples": len(text_lengths)}

    avg_len = np.mean(text_lengths)
    avg_conf = np.mean(confidences)

    drift_detected = False
    reasons = []

    if abs(avg_len - BASELINE_TEXT_LEN) > 40:
        drift_detected = True
        reasons.append("text_length_shift")

    if avg_conf < BASELINE_CONFIDENCE - 0.1:
        drift_detected = True
        reasons.append("confidence_drop")

    return {
        "drift": drift_detected,
        "samples": len(text_lengths),
        "avg_text_length": round(avg_len, 2),
        "avg_confidence": round(avg_conf, 3),
        "reasons": reasons
    }
