import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
with open("variants_200.json") as f:
    data = json.load(f)

# --- Feature extractor ---
def features(mutation):
    # Example: N215S
    if len(mutation) < 3:
        return [0,0,0]

    start = mutation[0]
    end = mutation[-1]

    # extract position
    pos = "".join([c for c in mutation if c.isdigit()])
    pos = int(pos) if pos else 0

    # encode letters numerically
    return [
        pos,
        ord(start),
        ord(end)
    ]

# --- Prepare training data ---
X=[]
y=[]

for m,info in data.items():
    X.append(features(m))
    y.append(1 if info["amenable"] else 0)

X=np.array(X)
y=np.array(y)

# --- Train model ---
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X,y)

# --- Save model ---
joblib.dump(model,"mutation_model.pkl")

print("Model trained and saved.")