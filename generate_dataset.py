import json
import random
import string

# Load your existing dataset
with open("variants.json") as f:
    base = json.load(f)

dataset = dict(base)

# amino acids
aas = list("ACDEFGHIKLMNPQRSTVWY")

def random_mutation():
    start = random.choice(aas)
    end = random.choice(aas)
    pos = random.randint(20,450)
    return f"{start}{pos}{end}"

# template fields
def make_entry(mutation):
    return {
        "gene": "GLA",
        "protein_change": f"p.{mutation}",
        "mutation_type": "Missense",
        "amenable": random.choice([True, False]),
        "drug": "Migalastat",
        "disease": "Fabry disease",
        "severity": random.choice(["Mild","Moderate","Severe","Variable"]),
        "enzyme_activity_effect": random.choice([
            "Partial loss of function",
            "Reduced stability",
            "Minimal activity",
            "Structural disruption"
        ]),
        "mechanism": random.choice([
            "Potential folding instability",
            "Likely chaperone-responsive",
            "Active-site disturbance",
            "Unknown structural effect"
        ]),
        "evidence_level": "Synthetic training data",
        "clinical_notes": "Auto-generated prototype mutation"
    }

# generate until 200 mutations
while len(dataset) < 200:
    m = random_mutation()
    if m not in dataset:
        dataset[m] = make_entry(m)

# save new dataset
with open("variants_200.json","w") as f:
    json.dump(dataset,f,indent=2)

print("Generated dataset with",len(dataset),"mutations")