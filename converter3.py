import pennylane as qml
import numpy as np
import scipy.linalg as la

# ======================================================
# Global config
# ======================================================
theta = np.pi / 7
epsilon = 1e-12

# ======================================================
# Metrics
# ======================================================
def operator_norm_distance(U, V):
    """min_phi || U - e^{i phi} V ||_op"""
    overlap = np.trace(U.conj().T @ V)
    phase = overlap / abs(overlap) if abs(overlap) > 1e-14 else 1.0
    return la.svdvals(U - phase * V)[0]

def t_count_from_tape(tape):
    return sum(op.name == "T" for op in tape.operations)

# ======================================================
# Device
# ======================================================
dev2 = qml.device("default.qubit", wires=2)

# ======================================================
# Problem definitions (1–10)
# ======================================================
def problem_1():
    qml.CY(wires=[0, 1])

def problem_2():
    qml.CRY(theta, wires=[0, 1])

def problem_3():
    qml.IsingZZ(2 * theta, wires=[0, 1])

def problem_4():
    qml.IsingXX(theta, wires=[0, 1])
    qml.IsingYY(theta, wires=[0, 1])

def problem_5():
    qml.SWAP(wires=[0, 1])

def problem_6():
    qml.IsingZZ(theta, wires=[0, 1])
    qml.RZ(theta, wires=0)
    qml.RZ(theta, wires=1)

def problem_7():
    state = np.array([
        0.1061479384 - 0.6796414671j,
       -0.3622775887 - 0.4536131361j,
        0.2614190429 + 0.0445330969j,
        0.3276449279 - 0.1101628411j
    ])
    qml.StatePrep(state, wires=[0, 1])

def problem_8():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RZ(theta, wires=1)
    qml.CNOT(wires=[0, 1])
    qml.Hadamard(wires=0)

def problem_9():
    qml.Hadamard(wires=0)
    qml.ControlledPhaseShift(np.pi / 2, wires=[1, 0])
    qml.Hadamard(wires=1)
    qml.SWAP(wires=[0, 1])

def problem_10():
    rng = np.random.default_rng(42)
    U = qml.math.linalg.qr(
        rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    )[0]
    qml.QubitUnitary(U, wires=[0, 1])

problems = {
    "01_controlled_y": problem_1,
    "02_cry_pi_7": problem_2,
    "03_exp_zz": problem_3,
    "04_exp_xx_yy": problem_4,
    "05_swap": problem_5,
    "06_ising": problem_6,
    "07_state_prep": problem_7,
    "08_structured": problem_8,
    "09_qft": problem_9,
    "10_random": problem_10,
}

# ======================================================
# Loop over problems
# ======================================================
print(f"{'Circuit':18s} | {'T-count':7s} | {'Dist':>10s}")
print("-" * 45)

for name, builder in problems.items():

    @qml.qnode(dev2)
    def exact():
        builder()
        return qml.state()

    @qml.clifford_t_decomposition(method="gridsynth", epsilon=epsilon)
    @qml.qnode(dev2)
    def compiled():
        builder()
        return qml.state()

    U_exact = qml.matrix(exact)()
    U_compiled = qml.matrix(compiled)()

    tape = qml.tape.make_qscript(compiled)()

    tc = t_count_from_tape(tape)
    dist = operator_norm_distance(U_exact, U_compiled)

    print(f"{name:18s} | {tc:7d} | {dist:10.3e}")
