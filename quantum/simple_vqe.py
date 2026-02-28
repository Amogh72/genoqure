import sys
mutation = sys.argv[1] if len(sys.argv) > 1 else "DEFAULT"

import numpy as np
import matplotlib.pyplot as plt

from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import TwoLocal
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorEstimator

# ----------------------------
# Hamiltonian (toy molecule)
# ----------------------------
pos = int("".join(c for c in mutation if c.isdigit()) or 0)
scale = pos / 100

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
    energies.append(value)

# ----------------------------
# Estimator primitive (NEW API)
# ----------------------------
estimator = StatevectorEstimator()

# ----------------------------
# Run VQE
# ----------------------------
vqe = VQE(estimator, ansatz, optimizer=optimizer, callback=callback)

result = vqe.compute_minimum_eigenvalue(H)

print("\nEstimated ground state energy:", float(result.eigenvalue.real))

# ----------------------------
# Plot convergence
# ----------------------------
plt.figure()
plt.plot(energies)
plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.title("VQE Convergence")
plt.grid()

# SAVE for website
plt.savefig("static/vqe_plot.png")