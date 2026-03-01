import json
import subprocess
import sys
import joblib
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.calibration import CalibratedClassifierCV

# ---------------- BASELINE ENERGY ----------------
BASELINE_ENERGY = -1.0499999998620173  # WT baseline

# ---------------- LOAD DATA ----------------
with open("variants_200.json") as f:
    variants = json.load(f)

X_classical = []
X_hybrid = []
y = []

# ---------------- QUANTUM FEATURE EXTRACTION ----------------
def extract_quantum_features(mutation):
    proc = subprocess.run(
        [sys.executable, "quantum/simple_vqe.py", mutation],
        capture_output=True,
        text=True
    )

    quantum_data = json.loads(proc.stdout)

    final_energy = quantum_data["final_energy"]
    min_energy = quantum_data["min_energy"]
    iterations = quantum_data["iterations"]
    variance = quantum_data["variance"]

    return final_energy, min_energy, iterations, variance

# ---------------- BUILD DATASET ----------------
for mutation, data in variants.items():

    if "amenable" not in data:
        continue

    label = 1 if data["amenable"] else 0

    start = mutation[0]
    end = mutation[-1]
    pos = int("".join([c for c in mutation if c.isdigit()]) or 0)

    # Classical features
    classical_features = [
        pos,
        ord(start),
        ord(end)
    ]

    # Quantum features
    q_energy, q_min, q_iter, q_var = extract_quantum_features(mutation)

    delta_energy = q_energy - BASELINE_ENERGY
    stability_index = abs(delta_energy) * q_var
    interaction = q_var * pos
    convergence_rate = abs(q_energy - q_min) / q_iter if q_iter != 0 else 0

    hybrid_features = [
        pos,
        ord(start),
        ord(end),
        q_energy,
        q_min,
        q_iter,
        q_var,
        delta_energy,
        stability_index,
        interaction,
        convergence_rate
    ]

    X_classical.append(classical_features)
    X_hybrid.append(hybrid_features)
    y.append(label)

# ---------------- TRAIN/TEST SPLIT ----------------
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_classical, y, test_size=0.2, random_state=42
)

Xh_train, Xh_test, yh_train, yh_test = train_test_split(
    X_hybrid, y, test_size=0.2, random_state=42
)

# ---------------- CLASSICAL MODEL ----------------
clf_classical = RandomForestClassifier(n_estimators=300, random_state=42)
clf_classical.fit(Xc_train, yc_train)

yc_pred = clf_classical.predict(Xc_test)

# ---------------- HYBRID MODEL ----------------
clf_hybrid = RandomForestClassifier(n_estimators=300, random_state=42)

# Probability calibration
calibrated_model = CalibratedClassifierCV(clf_hybrid, method="sigmoid", cv=3)
calibrated_model.fit(Xh_train, yh_train)

yh_pred = calibrated_model.predict(Xh_test)

# ---------------- METRICS ----------------
print("\n=== Classical Model Performance ===")
print("Accuracy:", accuracy_score(yc_test, yc_pred))
print("Precision:", precision_score(yc_test, yc_pred))
print("Recall:", recall_score(yc_test, yc_pred))
print("F1:", f1_score(yc_test, yc_pred))

print("\n=== Hybrid Model Performance ===")
print("Accuracy:", accuracy_score(yh_test, yh_pred))
print("Precision:", precision_score(yh_test, yh_pred))
print("Recall:", recall_score(yh_test, yh_pred))
print("F1:", f1_score(yh_test, yh_pred))

# ---------------- FEATURE IMPORTANCE ----------------
rf_model = calibrated_model.calibrated_classifiers_[0].estimator
importances = rf_model.feature_importances_

labels = [
    "Position",
    "Start AA",
    "End AA",
    "Q Final Energy",
    "Q Min Energy",
    "Q Iterations",
    "Q Variance",
    "Delta Energy",
    "Stability Index",
    "Variance-Position Interaction",
    "Convergence Rate"
]

plt.figure(figsize=(10,6))
plt.bar(labels, importances)
plt.xticks(rotation=60)
plt.ylabel("Importance Score")
plt.title("Hybrid Model Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# ---------------- SHAP ANALYSIS ----------------
Xh_test_np = np.array(Xh_test)

explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(Xh_test_np)

shap.summary_plot(shap_values, Xh_test_np, feature_names=labels)

# ---------------- SAVE FINAL MODEL ----------------
joblib.dump(calibrated_model, "mutation_model.pkl")

print("\nHybrid calibrated model trained and saved successfully.")