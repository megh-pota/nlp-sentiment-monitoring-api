import time
from datetime import datetime


def retrain_model(reason: str):
    print("🚀 Retraining triggered!")
    print(f"Reason: {reason}")

    # Simulate heavy training job
    for i in range(5):
        print(f"Training step {i+1}/5 ...")
        time.sleep(1)

    print("✅ Model retraining completed")
    print("📦 New model version registered")
    print(f"🕒 Timestamp: {datetime.now()}")


if __name__ == "__main__":
    retrain_model("manual_test")
