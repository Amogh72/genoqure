import sys
import json
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import TwoLocal
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorEstimator

# ----------------------------
# Silence deprecation warnings (important for clean JSON output)
# ----------------------------
warnings.filterwarnings("ignore")

# ----------------------------
# Get mutation input
# ----------------------------
mutation = sys.argv[1] if len(sys.argv) > 1 else "DEFAULT"

# ----------------------------
# Generate mutation-based scaling
# ----------------------------
pos = int("".join(c for c in mutation if c.isdigit()) or 0)
scale = pos / 100

# ----------------------------
# Hamiltonian (toy molecule)
# ----------------------------
H = SparsePauliOp.from_list([
    ("ZZ", -1.05 - scale),
    ("XX", 0.39 + scale/2),
    ("YY", 0.39 + scale/2),
    ("ZI", 0.4 - scale/3),
    ("IZ", -0.4 + scale/3)
])

# ----------------------------
# Ansatz circuit
# ----------------------------
ansatz = TwoLocal(
    num_qubits=2,
    rotation_blocks="ry",
    entanglement_blocks="cz",
    reps=2
)

# ----------------------------
# Optimizer
# ----------------------------
optimizer = COBYLA(maxiter=60)

energies = []

def callback(eval_count, params, value, metadata):
    energies.append(float(value))

# ----------------------------
# Estimator primitive
# ----------------------------
estimator = StatevectorEstimator()

# ----------------------------
# Run VQE
# ----------------------------
vqe = VQE(estimator, ansatz, optimizer=optimizer, callback=callback)
result = vqe.compute_minimum_eigenvalue(H)

final_energy = float(result.eigenvalue.real)
min_energy = float(min(energies)) if energies else final_energy
iterations = len(energies)
energy_variance = float(np.var(energies)) if energies else 0.0

# ----------------------------
# Plot convergence (safe absolute path)
# ----------------------------
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
save_path = os.path.join(base_dir, "static", "vqe_plot.png")

plt.figure()
plt.plot(energies)
plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.title("VQE Convergence")
plt.grid()
plt.savefig(save_path)
plt.close()

# ----------------------------
# OUTPUT JSON (ONLY OUTPUT)
# ----------------------------
output = {
    "final_energy": final_energy,
    "min_energy": min_energy,
    "iterations": iterations,
    "variance": energy_variance
}

print(json.dumps(output))