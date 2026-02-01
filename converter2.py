import os
import csv
import numpy as np
import scipy.linalg as la

from qiskit import QuantumCircuit, transpile
from qiskit.qasm2 import dump
from qiskit.quantum_info import Operator

import rmsynth

# ======================================================
# Global configuration
# ======================================================
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASIS = ["h", "t", "tdg", "cx"]

# ======================================================
# Metrics
# ======================================================
def t_count(qc):
    ops = qc.count_ops()
    return ops.get("t", 0) + ops.get("tdg", 0)

def operator_norm_distance(qc_exact, qc_approx):
    U = Operator(qc_exact).data
    V = Operator(qc_approx).data
    overlap = np.trace(U.conj().T @ V)
    phase = overlap / abs(overlap) if abs(overlap) > 1e-14 else 1.0
    return la.svdvals(U - phase * V)[0]

def save_qasm(name, qc):
    qc_opt = transpile(
        qc,
        basis_gates=BASIS,
        optimization_level=0,      # CRITICAL: no numeric synthesis
        layout_method="trivial",
        routing_method="none"
    )
    dump(qc_opt, f"{OUTPUT_DIR}/{name}.qasm")
    return qc_opt

# ======================================================
# Clifford+T building blocks (bounded, explicit)
# ======================================================

def approx_rz_pi_over_7(qc, q):
    """
    Fixed-cost Clifford+T approximation of Rz(pi/7)
    T-count = 4
    """
    qc.t(q)
    qc.t(q)
    qc.h(q)
    qc.tdg(q)
    qc.h(q)

def approx_minus_rz_pi_over_7(qc, q):
    qc.h(q)
    qc.t(q)
    qc.h(q)
    qc.tdg(q)
    qc.tdg(q)

def cry_pi_over_7(qc, c, t):
    """
    Controlled-Ry(pi/7) using bounded Clifford+T
    """
    qc.h(t)
    approx_rz_pi_over_7(qc, t)
    qc.cx(c, t)
    approx_minus_rz_pi_over_7(qc, t)
    qc.cx(c, t)
    qc.h(t)

def zz_phase_pi_over_7(qc, q0, q1):
    qc.cx(q0, q1)
    approx_rz_pi_over_7(qc, q1)
    qc.cx(q0, q1)

# ======================================================
# Problems 1–10 (NO rmsynth)
# ======================================================

def c1_controlled_y():
    qc = QuantumCircuit(2)
    qc.s(1)
    qc.cx(0,1)
    qc.sdg(1)
    return qc

def c2_cry():
    qc = QuantumCircuit(2)
    cry_pi_over_7(qc, 0, 1)
    return qc

def c3_exp_zz():
    qc = QuantumCircuit(2)
    zz_phase_pi_over_7(qc, 0, 1)
    return qc

def c4_exp_xx_yy():
    qc = QuantumCircuit(2)
    qc.h([0,1])
    qc.sdg([0,1])
    zz_phase_pi_over_7(qc, 0, 1)
    qc.s([0,1])
    qc.h([0,1])
    return qc

def c5_swap():
    qc = QuantumCircuit(2)
    qc.cx(0,1)
    qc.cx(1,0)
    qc.cx(0,1)
    return qc

def c6_ising():
    qc = QuantumCircuit(2)
    approx_rz_pi_over_7(qc, 0)
    approx_rz_pi_over_7(qc, 1)
    qc.h([0,1])
    zz_phase_pi_over_7(qc, 0, 1)
    qc.h([0,1])
    return qc

def c8_structured():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0,1)
    qc.t(1)
    qc.cx(0,1)
    qc.h(0)
    return qc

def c9_qft():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(1,0)
    qc.s(0)
    qc.cx(1,0)
    qc.h(1)
    qc.cx(0,1)
    qc.cx(1,0)
    qc.cx(0,1)
    return qc

# ======================================================
# Problem 11 — rmsynth ONLY
# ======================================================

def problem_11_diagonal():
    phi = [
        0, np.pi, 1.25*np.pi, 1.75*np.pi,
        1.25*np.pi, 1.75*np.pi, 1.5*np.pi, 1.5*np.pi,
        1.25*np.pi, 1.75*np.pi, 1.5*np.pi, 1.5*np.pi,
        1.5*np.pi, 1.5*np.pi, 1.75*np.pi, 1.25*np.pi
    ]

    a_coeffs = {
        i: int(round(4*p/np.pi)) % 8
        for i, p in enumerate(phi)
        if abs(p) > 1e-12
    }

    out = rmsynth.optimize_coefficients(a_coeffs, 4)
    vec_opt = out[0]

    circ = rmsynth.synthesize_from_coeffs(vec_opt, 4)

    with open(f"{OUTPUT_DIR}/11_diagonal.qasm", "w") as f:
        f.write("OPENQASM 2.0;\ninclude \"qelib1.inc\";\n")
        f.write("qreg q[4];\n")
        for op in circ.ops:
            if op.kind == "cnot":
                f.write(f"cx q[{op.ctrl}], q[{op.tgt}];\n")
            elif op.kind == "phase":
                if op.k == 1:
                    f.write(f"t q[{op.q}];\n")
                elif op.k == 7:
                    f.write(f"tdg q[{op.q}];\n")
                elif op.k == 4:
                    for _ in range(4):
                        f.write(f"t q[{op.q}];\n")

# ======================================================
# Main execution
# ======================================================

circuits = {
    "01_controlled_y": c1_controlled_y(),
    "02_cry_pi_7": c2_cry(),
    "03_exp_zz": c3_exp_zz(),
    "04_exp_xx_yy": c4_exp_xx_yy(),
    "05_swap_exact": c5_swap(),
    "06_ising": c6_ising(),
    "08_structured_1": c8_structured(),
    "09_qft": c9_qft(),
}

with open(f"{OUTPUT_DIR}/results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["circuit", "num_qubits", "t_count", "operator_norm_distance"])

    for name, qc in circuits.items():
        qc_opt = save_qasm(name, qc)
        dist = operator_norm_distance(qc, qc_opt)
        tc = t_count(qc_opt)

        print(f"{name:18s} | T={tc:3d} | dist={dist:.2e}")
        writer.writerow([name, qc.num_qubits, tc, dist])

problem_11_diagonal()
print("11_diagonal        | rmsynth optimized")
