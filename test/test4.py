import numpy as np
import os
import scipy.linalg as la
from typing import Dict, List, Tuple, Optional  # Fix for NameError 
import rmsynth
from qiskit import QuantumCircuit, transpile
from qiskit.qasm2 import dump
from qiskit.quantum_info import Operator, random_unitary
from qiskit.circuit.library import Diagonal, StatePreparation

# ======================================================
# RM Utilities (Internalized to fix AttributeError)
# ======================================================
def iter_nonzero_masks(n: int):
    for m in range(1, 1 << n):
        yield m

def mk_positions(n: int):
    order = list(iter_nonzero_masks(n))
    pos = {m: i for i, m in enumerate(order)}
    return order, pos

def coeffs_to_vec(a: Dict[int, int], n: int) -> List[int]:
    order, pos = mk_positions(n)
    vec = [0]*len(order)
    for y, val in a.items():
        if y in pos:
            vec[pos[y]] = val % 8
    return vec

# ======================================================
# Metrics & Submission Config [cite: 63, 76, 80]
# ======================================================
BASIS_GATES = ["h", "t", "tdg", "cx"]
theta = np.pi / 7

def save_and_optimize(qc, name, n_qubits):
    """
    Optimizes via rmsynth and saves for iQuHACK submission. [cite: 12, 100]
    """
    # Standard Qiskit transpilation as a baseline [cite: 67]
    qc_baseline = transpile(qc, basis_gates=BASIS_GATES, optimization_level=3)
    
    # Extract phase coefficients (Logic from Section 2) 
    # For complex unitaries, we pass them to the RM optimizer [cite: 101]
    try:
        # Note: In a full automated setup, you'd extract 'a_coeffs' from the matrix
        # For Problem 11, we pass the known phases [cite: 55, 61]
        if name == "11_diagonal":
            phi = [0, np.pi, 1.25*np.pi, 1.75*np.pi, 1.25*np.pi, 1.75*np.pi, 1.5*np.pi, 1.5*np.pi,
                   1.25*np.pi, 1.75*np.pi, 1.5*np.pi, 1.5*np.pi, 1.5*np.pi, 1.5*np.pi, 1.75*np.pi, 1.25*np.pi]
            a_coeffs = {i: int(round(4 * p / np.pi)) % 8 for i, p in enumerate(phi) if p != 0}
            vec_opt, _, _ = rmsynth.optimize_coefficients(a_coeffs, n_qubits)
            optimized_circ = rmsynth.synthesize_from_coeffs(vec_opt, n_qubits)
            print(f"✨ {name} Optimized via rmsynth | T-count: {optimized_circ.t_count()}")
            return None
        else:
            optimized_circ = qc_baseline
    except Exception as e:
        print(f"⚠️ rmsynth skip for {name}: {e}")
        optimized_circ = qc_baseline

    dump(optimized_circ, f"{name}.qasm")
    tcnt = optimized_circ.count_ops().get("t", 0) + optimized_circ.count_ops().get("tdg", 0)
    print(f"✅ Generated {name}.qasm | T-count: {tcnt}")

# ======================================================
# Challenges 1-11 [cite: 17, 18, 25, 28, 30, 31, 33, 34, 46, 55]
# ======================================================

# 1. Controlled-Y [cite: 18]
qc1 = QuantumCircuit(2); qc1.cy(0, 1); save_and_optimize(qc1, "01_controlled_y", 2)

# 5. exp(i pi/2 H2) = SWAP [cite: 31, 32]
qc5 = QuantumCircuit(2); qc5.swap(0, 1); save_and_optimize(qc5, "05_swap_exact", 2)

# 7. State Preparation (seed=42) [cite: 34, 39]
state = [0.1061479384-0.6796414671j, -0.3622775887-0.4536131361j, 
         0.2614190429+0.04453309691j, 0.3276449279-0.1101628411j]
qc7 = QuantumCircuit(2); qc7.append(StatePreparation(state), [0, 1]); save_and_optimize(qc7, "07_state_prep", 2)

# 9. Structured 2 (QFT) [cite: 42, 43]
qc9 = QuantumCircuit(2); qc9.h(0); qc9.cp(np.pi/2, 1, 0); qc9.h(1); qc9.swap(0, 1); save_and_optimize(qc9, "09_qft", 2)

# 10. Random Unitary (seed=42) [cite: 46, 54]
u_rand = random_unitary(4, seed=42); qc10 = QuantumCircuit(2); qc10.unitary(u_rand, [0, 1]); save_and_optimize(qc10, "10_random", 2)

# 11. 4-qubit Diagonal [cite: 55, 61]
# Handled via internal rmsynth logic in save_and_optimize
qc11 = QuantumCircuit(4); save_and_optimize(qc11, "11_diagonal", 4)