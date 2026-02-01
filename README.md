## Efficient Clifford+T Decomposition via Reed–Muller Synthesis (rmsynth)

For circuits dominated by diagonal or phase-based unitaries, we employ **Reed–Muller synthesis (rmsynth)** to achieve resource-efficient Clifford+T decompositions. This approach is particularly well-suited for fault-tolerant compilation, where the primary cost metric is the **T-count**.

### Phase Polynomial Representation

Any diagonal unitary whose phases are multiples of π/4 can be represented as a **phase polynomial** over ℤ₈. In this representation, the unitary is written as

U = diag(exp(i π/4 · f(x))),

where f(x) is a Boolean polynomial defined over computational basis states. Each monomial corresponds to a parity function over a subset of qubits.

Crucially, **odd coefficients in the phase polynomial correspond directly to non-Clifford resources**, namely T or T† gates.

### Reed–Muller Optimization

The core idea behind rmsynth is to **minimize the number of odd coefficients** in the phase polynomial. This is achieved by exploiting equivalence classes of Boolean functions under Reed–Muller codes, allowing algebraic transformations that preserve the implemented unitary while reducing the number of non-Clifford terms.

By performing this optimization prior to circuit synthesis, rmsynth directly minimizes the resulting T-count at the algebraic level, rather than relying on gate-level heuristics.

### Circuit Synthesis

Once the optimized phase polynomial is obtained, the circuit is synthesized using:
- CNOT networks to compute parity functions,
- single-qubit phase gates implementing π/4-multiples,
- and Clifford uncomputation to restore ancilla-free structure.

The resulting circuits use only the **Clifford+T gate set** {H, T, T†, CNOT} and are often provably near-optimal in T-count for diagonal unitaries.

### Advantages of rmsynthesis

- Avoids numerical approximation entirely  
- Produces exact unitary implementations  
- Directly targets the dominant fault-tolerant cost (T-count)  
- Scales efficiently for multi-qubit diagonal operators  
- Complements symbolic Clifford simplification for non-diagonal circuits  

### Summary

Reed–Muller synthesis provides a principled and algebraic approach to Clifford+T compilation for diagonal and phase-heavy unitaries. By optimizing phase polynomials before circuit construction, rmsynth achieves low T-count implementations that are well-suited for fault-tolerant quantum architectures and rigorous benchmarking.
