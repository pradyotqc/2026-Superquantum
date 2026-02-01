import numpy as np
import scipy.linalg as la
import csv
from qiskit import QuantumCircuit, transpile
from qiskit.qasm2 import dump
from qiskit.quantum_info import Operator, random_unitary
from qiskit.circuit.library import DiagonalGate

# ======================================================
# Configuration & Metrics
# ======================================================
BASIS_GATES = ["h", "t", "tdg", "cx"]
theta = np.pi / 7
CSV_FILE = "results.csv"

# ======================================================
# Metrics
# ======================================================
def operator_norm_distance(U_exact, U_approx):
    """min_phi || U - exp(i phi) U_approx ||_op"""
    overlap = np.trace(U_exact.conj().T @ U_approx)
    phase = overlap / abs(overlap) if abs(overlap) > 1e-14 else 1.0
    diff = U_exact - phase * U_approx
    return la.svdvals(diff)[0]


def t_count(qc):
    ops = qc.count_ops()
    return ops.get("t", 0) + ops.get("tdg", 0)

# ======================================================
# CSV setup
# ======================================================
with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "circuit",
        "num_qubits",
        "t_count",
        "operator_norm_distance"
    ])

# ======================================================
# Compile → evaluate → save CSV
# ======================================================
def compile_and_evaluate(name, qc):
    qc_opt = transpile(
        qc,
        basis_gates=BASIS_GATES,
        # optimization_level=3,
        # approximation_degree=1e-12

    )

    dump(qc_opt, f"./output/{name}.qasm")

    U_exact = Operator(qc).data
    U_approx = Operator(qc_opt).data

    dist = operator_norm_distance(U_exact, U_approx)
    tcnt = t_count(qc_opt)

    # Console output
    print(f"{name:18s} | T-count = {tcnt:6d} | Dist = {dist:.3e}")

    # CSV output
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            name,
            qc.num_qubits,
            tcnt,
            dist
        ])

# ======================================================
# 1–11 Challenge Circuits
# ======================================================

# 1. Controlled-Y
qc1 = QuantumCircuit(2); qc1.cy(0, 1)
compile_and_evaluate("01_controlled_y", qc1)

# 2. Controlled-Ry(pi/7)
qc2 = QuantumCircuit(2); qc2.cry(theta, 0, 1)
compile_and_evaluate("02_cry_pi_7", qc2)

# 3. exp(i pi/7 Z⊗Z)
qc3 = QuantumCircuit(2); qc3.rzz(-2 * theta, 0, 1)
compile_and_evaluate("03_exp_zz", qc3)

# 4. exp(i pi/7 (XX + YY))
qc4 = QuantumCircuit(2)
qc4.h([0, 1]); qc4.rzz(-2 * theta, 0, 1); qc4.h([0, 1])
qc4.sdg([0, 1]); qc4.h([0, 1]); qc4.rzz(-2 * theta, 0, 1)
qc4.h([0, 1]); qc4.s([0, 1])
compile_and_evaluate("04_exp_h1", qc4)

# 5. exp(i pi/2 (XX + YY + ZZ)) = SWAP
qc5 = QuantumCircuit(2); qc5.swap(0, 1)
compile_and_evaluate("05_swap_exact", qc5)

# 6. Ising model
qc6 = QuantumCircuit(2)
qc6.rz(-2 * theta, 0); qc6.rz(-2 * theta, 1)
qc6.h([0, 1]); qc6.rzz(-2 * theta, 0, 1); qc6.h([0, 1])
compile_and_evaluate("06_ising", qc6)

# 8. Structured unitary 1
qc8 = QuantumCircuit(2)
qc8.h(0); qc8.cx(0, 1); qc8.rz(theta, 1); qc8.cx(0, 1); qc8.h(0)
compile_and_evaluate("08_structured_1", qc8)

# 9. 2-qubit QFT
qc9 = QuantumCircuit(2)
qc9.h(0); qc9.cp(np.pi/2, 1, 0); qc9.h(1); qc9.swap(0, 1)
compile_and_evaluate("09_qft", qc9)

# 10. Random unitary (seed=42)
u_rand = random_unitary(4, seed=42)
qc10 = QuantumCircuit(2); qc10.unitary(u_rand, [0, 1])
compile_and_evaluate("10_random", qc10)

# 11. 4-qubit diagonal unitary
phi = [
    0, np.pi, 1.25*np.pi, 1.75*np.pi,
    1.25*np.pi, 1.75*np.pi, 1.5*np.pi, 1.5*np.pi,
    1.25*np.pi, 1.75*np.pi, 1.5*np.pi, 1.5*np.pi,
    1.5*np.pi, 1.5*np.pi, 1.75*np.pi, 1.25*np.pi
]
qc11 = QuantumCircuit(4)
qc11.append(DiagonalGate(np.exp(1j * np.array(phi))), [0, 1, 2, 3])
compile_and_evaluate("11_diagonal", qc11)
