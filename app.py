import subprocess
import sys
import joblib

from flask import Flask, render_template, request, redirect, url_for
import json

app = Flask(__name__)

# Load dataset once at startup
with open("variants_200.json") as f:
    variants = json.load(f)

model = joblib.load("mutation_model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():

    # ---------------- POST ----------------
    if request.method == "POST":
        mutation_input = request.form["mutation"].strip().upper()
        return redirect(url_for("home", mutation=mutation_input))

    # ---------------- GET ----------------
    mutation_input = request.args.get("mutation")
    result = None

    if mutation_input:

        # ---------- DATABASE MATCH ----------
        if mutation_input in variants:
            result = variants[mutation_input].copy()
            result["source"] = "Database"

        # ---------- MODEL PREDICTION ----------
        else:
            probs = model.predict_proba(features(mutation_input))[0]
            pred = probs.argmax()
            confidence = probs[pred]

            if confidence < 0.65:
                result = {
                    "gene": "GLA",
                    "protein_change": "Unknown",
                    "amenable": None,
                    "drug": "Migalastat",
                    "disease": "Fabry disease",
                    "note": "Prediction uncertain — mutation outside training distribution",
                    "confidence": f"{confidence*100:.1f}%",
                    "source": "Model (low confidence)"
                }
            else:
                result = {
                    "gene": "GLA",
                    "protein_change": "Unknown",
                    "amenable": bool(pred),
                    "drug": "Migalastat",
                    "disease": "Fabry disease",
                    "note": "Predicted by ML model",
                    "confidence": f"{confidence*100:.1f}%",
                    "source": "Model prediction"
                }

        subprocess.run([
            sys.executable,
            "quantum/simple_vqe.py",
            mutation_input
        ])

    return render_template(
        "index.html",
        result=result,
        mutation=mutation_input
    )

def features(mutation):
    if len(mutation) < 3:
        return [[0,0,0]]

    start = mutation[0]
    end = mutation[-1]

    pos = "".join([c for c in mutation if c.isdigit()])
    pos = int(pos) if pos else 0

    return [[pos, ord(start), ord(end)]]

if __name__ == "__main__":
    app.run(debug=True)