import requests
import random
import string
import time

URL = "http://localhost:8000/predict"

def random_text(n=800):
    return "".join(random.choices(string.ascii_letters + " !!! ??? ### ", k=n))

for i in range(40):
    payload = {"text": random_text(1000 if i % 2 == 0 else 20)}
    r = requests.post(URL, json=payload)
    print(i, r.json()["drift"])
    time.sleep(0.2)
