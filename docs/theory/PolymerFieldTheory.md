# Polymer Field Theory

> **⚠️ Warning:** This document was generated with assistance from a large language model (LLM). While it is based on the referenced literature and the codebase, it may contain errors, misinterpretations, or inaccuracies. Please verify the equations and descriptions against the original references before relying on this document for research or implementation.

This document describes the mathematical formulation of polymer field theory used in SCFT, L-FTS, and CL-FTS simulations. Three representations are covered: monomer potential fields, auxiliary fields (incompressible model), and auxiliary fields (compressible model).

## Table of Contents

1. [Monomer Potential Fields](#1-monomer-potential-fields)
2. [Incompressible Model](#2-incompressible-model)
3. [Compressible Model](#3-compressible-model)
4. [Implementation](#4-implementation)
5. [References](#5-references)

---

## 1. Monomer Potential Fields

The monomer potential field formulation expresses the Helmholtz free energy directly in terms of monomer potential fields (e.g., $w_A(\mathbf{r})$, $w_B(\mathbf{r})$). This formulation is implemented in `scft.py`.

### 1.1 Helmholtz Free Energy

**Per monomer unit:**

$$\beta F = -\sum_p n_p \log Q_p + \rho_0 \int d\mathbf{r} \left[ \frac{1}{2} \sum_{ij} \chi_{ij} \phi_i(\mathbf{r}) \phi_j(\mathbf{r}) - \sum_i w_i(\mathbf{r}) \phi_i(\mathbf{r}) \right]$$

**Per chain unit** (with rescaling $N w_i(\mathbf{r}) \rightarrow w_i(\mathbf{r})$):

$$\frac{\beta F}{n} = -\sum_p \frac{\bar{\phi}_p}{\alpha_p} \log Q_p + \frac{1}{V} \int d\mathbf{r} \left[ \frac{1}{2} \sum_{ij} \chi_{ij} N \phi_i(\mathbf{r}) \phi_j(\mathbf{r}) - \sum_i w_i(\mathbf{r}) \phi_i(\mathbf{r}) \right]$$

where:
- $n_p$: number of $p$-chains
- $Q_p$: single-chain partition function of $p$-chain
- $\alpha_p = N_p / N$: relative chain length
- $\bar{\phi}_p$: volume fraction of $p$-chains
- $\chi_{ij}$: Flory-Huggins interaction parameter between monomers $i$ and $j$

### 1.2 Self-Consistent Conditions

The self-consistent conditions for the potential fields are:

$$w_i(\mathbf{r}) = \sum_{j \neq i} \chi_{ij} N \phi_j(\mathbf{r}) + \xi(\mathbf{r})$$

where $\xi(\mathbf{r})$ is a Lagrange multiplier enforcing incompressibility.

### 1.3 Field Residuals (Incompressible)

The field residuals $\mathbf{R}_w$ are defined as:

$$\mathbf{R}_w = X \boldsymbol{\phi} - P \mathbf{w}$$

where:
- $X$ is the $\chi N$ matrix
- $\boldsymbol{\phi}$ and $\mathbf{w}$ are vectors of concentrations and fields
- $P$ is the projection matrix:

$$P = I - \frac{\mathbf{e} \mathbf{e}^T X^{-1}}{\mathbf{e}^T X^{-1} \mathbf{e}}$$

### 1.4 Self-Consistent Conditions (Compressible)

With finite compressibility $\zeta N$, the pressure field is known:

$$\xi(\mathbf{r}) = \zeta N \left[ \sum_j \phi_j(\mathbf{r}) - 1 \right]$$

$$w_i(\mathbf{r}) = \sum_{j \neq i} \chi_{ij} N \phi_j(\mathbf{r}) + \xi(\mathbf{r})$$

In SCFT iteration, the same projection residual $\mathbf{R}_w = X \boldsymbol{\phi} - P \mathbf{w}$ is used for the fluctuation part, while the field means are pinned to the saddle-point values:

$$\langle w_i \rangle = \sum_j X_{ij} \langle \phi_j \rangle + \xi, \qquad \xi = \zeta N \left( \left\langle \sum_j \phi_j \right\rangle - 1 \right)$$

See [SelfConsistentFieldTheory.md](SelfConsistentFieldTheory.md) for details on the compressible SCFT iteration.

### 1.5 Example: AB Diblock Copolymer

For AB-type systems:

$$\frac{\beta F}{n} = -\sum_p \frac{\bar{\phi}_p}{\alpha_p} \log Q_p + \frac{1}{V} \int d\mathbf{r} \left[ \chi_{AB} N \phi_A(\mathbf{r}) \phi_B(\mathbf{r}) - w_A(\mathbf{r}) \phi_A - w_B(\mathbf{r}) \phi_B \right]$$

Self-consistent conditions:

$$w_A(\mathbf{r}) = \chi_{AB} N \phi_B(\mathbf{r}) + \xi(\mathbf{r})$$

$$w_B(\mathbf{r}) = \chi_{AB} N \phi_A(\mathbf{r}) + \xi(\mathbf{r})$$

Field residuals:

$$\mathbf{R}_w = \begin{pmatrix} 0 & \chi N \\ \chi N & 0 \end{pmatrix} \begin{pmatrix} \phi_A(\mathbf{r}) \\ \phi_B(\mathbf{r}) \end{pmatrix} - \begin{pmatrix} 1/2 & -1/2 \\ -1/2 & 1/2 \end{pmatrix} \begin{pmatrix} w_A(\mathbf{r}) \\ w_B(\mathbf{r}) \end{pmatrix}$$

---

## 2. Incompressible Model

The incompressible model uses auxiliary fields to decouple the interaction terms. This formulation is implemented in `polymer_field_theory.py` and used by both `scft.py` and `lfts.py`.

### 2.1 Partition Function

The canonical partition function with incompressibility constraint:

$$\mathcal{Z} \propto \int \{\mathcal{D}\mathbf{r}_i\} \exp(-U_{id} - U_{int}) \, \delta\left[\sum_{i=1}^{M} \rho_i(\mathbf{r}) - \rho_0\right]$$

where:
- $U_{id}$: intramolecular potential energy
- $U_{int}$: intermolecular interaction energy

$$U_{int} = \frac{1}{2\rho_0} \int d\mathbf{r} \sum_{i,j} \chi_{ij} \rho_i(\mathbf{r}) \rho_j(\mathbf{r})$$

### 2.2 Field-Theoretic Representation

After Hubbard-Stratonovich transformation:

$$\mathcal{Z} \propto \int \{\mathcal{D}\Omega_i\} \exp(-\beta H[\{\Omega_i\}])$$

### 2.3 Effective Hamiltonian

**Per chain unit:**

$$\frac{\beta H}{C V / R_0^3} = -\sum_p \frac{\bar{\phi}_p}{\alpha_p} \log Q_p + \frac{1}{V} \int d\mathbf{r} \left[ \sum_{i=1}^{M-1} \frac{M \Omega_i^2}{2\lambda_i} + \sum_{i=1}^{M-1} \frac{\Omega_i S_i}{\lambda_i} - \Omega_M + \frac{1}{2M} \left( \mathbf{s}^T \mathbf{e} - \sum_{i=1}^{M-1} \frac{S_i^2}{\lambda_i} \right) \right]$$

where:
- $C = \rho_0 R_0^3 / N$: dimensionless chain number density
- $\bar{N} = C^2$: invariant polymerization index
- $O$: orthogonal matrix of $PXP$ with $OO^T = M$
- $\mathbf{s}^T = \mathbf{e}^T X / M$ and $\mathbf{S} = O^T \mathbf{s} / M$
- $\lambda_i$: eigenvalues of $PXP$

### 2.4 Field Transformations

The relations between auxiliary fields $\{\Omega_i\}$ and monomer potential fields $\{W_i\}$:

$$\begin{pmatrix} W_1(\mathbf{r}) \\ W_2(\mathbf{r}) \\ \vdots \\ W_M(\mathbf{r}) \end{pmatrix} = O \begin{pmatrix} \Omega_1(\mathbf{r}) \\ \Omega_2(\mathbf{r}) \\ \vdots \\ \Omega_M(\mathbf{r}) \end{pmatrix}$$

### 2.5 Example: AB Diblock (Incompressible)

For AB-type systems, the auxiliary fields are:
- $\Omega_-$: exchange field (composition fluctuations)
- $\Omega_+$: pressure-like field (incompressibility)

**Effective Hamiltonian:**

$$\frac{\beta H}{C V / R_0^3} = -\sum_p \frac{\bar{\phi}_p}{\alpha_p} \log Q_p + \frac{1}{V} \int d\mathbf{r} \left[ \frac{\Omega_-^2(\mathbf{r})}{\chi N} - \Omega_+(\mathbf{r}) + \frac{\chi N}{4} \right]$$

**Functional derivatives:**

$$\frac{\beta}{C/R_0^3} \frac{\delta H}{\delta \Omega_-(\mathbf{r})} = \Phi_-(\mathbf{r}) + \frac{2}{\chi N} \Omega_-(\mathbf{r})$$

$$\frac{\beta}{C/R_0^3} \frac{\delta H}{\delta \Omega_+(\mathbf{r})} = \Phi_+(\mathbf{r}) - 1$$

**Transformation matrix:**

$$O = \begin{pmatrix} 1 & 1 \\ -1 & 1 \end{pmatrix}$$

$$\begin{pmatrix} W_A(\mathbf{r}) \\ W_B(\mathbf{r}) \end{pmatrix} = O \begin{pmatrix} \Omega_-(\mathbf{r}) \\ \Omega_+(\mathbf{r}) \end{pmatrix}, \quad \begin{pmatrix} \phi_A(\mathbf{r}) \\ \phi_B(\mathbf{r}) \end{pmatrix} = O \begin{pmatrix} \Phi_-(\mathbf{r}) \\ \Phi_+(\mathbf{r}) \end{pmatrix}$$

---

## 3. Compressible Model

The compressible model includes a finite compressibility parameter $\zeta$. This formulation is implemented in `polymer_field_theory.py` and may be used for complex Langevin FTS.

### 3.1 Interaction Energy

$$U_{int} = \frac{1}{2\rho_0} \int d\mathbf{r} \left\lbrace \sum_{i,j} \chi_{ij} \rho_i(\mathbf{r}) \rho_j(\mathbf{r}) + \zeta \left[ \sum_{i=1}^{M} \rho_i(\mathbf{r}) - \rho_0 \right]^2 \right\rbrace$$

### 3.2 Effective Hamiltonian

**Per chain unit:**

$$\frac{\beta H}{C V / R_0^3} = -\sum_p \frac{\bar{\phi}_p}{\alpha_p} \log Q_p + \frac{1}{V} \int d\mathbf{r} \left[ \sum_{i=1}^{M} \frac{M \Omega_i^2}{2\lambda_i} + \sum_{i=1}^{M} \frac{\Omega_i S_i}{\lambda_i} + \frac{1}{2} \left( \zeta N - \sum_{i=1}^{M} \frac{S_i^2}{M\lambda_i} \right) \right]$$

where:
- $O$: orthogonal matrix of $U = X + \zeta \mathbf{e}\mathbf{e}^T$ with $OO^T = M$
- $\mathbf{S}^T = -\zeta \mathbf{e}^T O$

### 3.3 Example: AB Diblock (Compressible)

**Effective Hamiltonian:**

$$\frac{\beta H}{C V / R_0^3} = -\sum_p \frac{\bar{\phi}_p}{\alpha_p} \log Q_p + \frac{1}{V} \int d\mathbf{r} \left[ \frac{\Omega_-^2(\mathbf{r})}{\chi N} - \frac{\Omega_+^2(\mathbf{r})}{\chi N + 2\zeta N} - \frac{2\zeta N}{\chi N + 2\zeta N} \Omega_+(\mathbf{r}) + \frac{\chi N \cdot \zeta N}{2(\chi N + 2\zeta N)} \right]$$

**Functional derivatives:**

$$\frac{\beta}{C/R_0^3} \frac{\delta H}{\delta \Omega_-(\mathbf{r})} = \Phi_-(\mathbf{r}) + \frac{2}{\chi N} \Omega_-(\mathbf{r})$$

$$\frac{\beta}{C/R_0^3} \frac{\delta H}{\delta \Omega_+(\mathbf{r})} = \Phi_+(\mathbf{r}) - \frac{2}{\chi N + 2\zeta N} (\Omega_+(\mathbf{r}) + \zeta N)$$

---

## 4. Implementation

### 4.1 SymmetricPolymerTheory Class

The `polymerfts.SymmetricPolymerTheory` class computes the coefficients for the effective Hamiltonian:

$$\frac{\beta H}{C V / R_0^3} = -\sum_p \frac{\bar{\phi}_p}{\alpha_p} \log Q_p + \frac{1}{V} \int d\mathbf{r} \left[ \sum_{i=1}^{M} a_i \Omega_i^2(\mathbf{r}) + \sum_{i=1}^{M} b_i \Omega_i(\mathbf{r}) + U_{ref} \right]$$

**Functional derivative:**

$$\frac{\beta}{C/R_0^3} \frac{\delta H}{\delta \Omega_i(\mathbf{r})} = \Phi_i(\mathbf{r}) + 2 a_i \Omega_i(\mathbf{r}) + b_i$$

where $\Phi_i(\mathbf{r}) = \sum_j O_{ji} \phi_j(\mathbf{r})$.

### 4.2 Usage Example

```python
import polymerfts

# Define system
monomer_types = ["A", "B"]
chi_n = {"A,B": 20}

# Create theory object (incompressible)
mpt = polymerfts.SymmetricPolymerTheory(monomer_types, chi_n, zeta_n=None)

# Create theory object (compressible)
mpt_comp = polymerfts.SymmetricPolymerTheory(monomer_types, chi_n, zeta_n=100)

# Convert between field representations
omega = np.random.rand(2, n_grid)
w = mpt.to_monomer_fields(omega)      # Auxiliary → Monomer
omega = mpt.to_aux_fields(w)          # Monomer → Auxiliary

# Compute Hamiltonian
H = mpt.compute_hamiltonian(molecules, omega, total_partitions)

# Compute functional derivatives
h_deriv = mpt.compute_func_deriv(omega, phi, field_indices)
```

### 4.3 Sign Convention for Imaginary Fields

In the implementation, there is a negative sign in front of the functional derivative with respect to imaginary fields. This is because we find the minimum for real fields and maximum for imaginary fields:

```python
h_deriv = mpt.compute_func_deriv(omega, phi, [0, 1])
# h_deriv[0] = +δH/δΩ_- (real field, minimize)
# h_deriv[1] = -δH/δΩ_+ (imaginary field, maximize)
```

### 4.4 Key Attributes

| Attribute | Description |
|-----------|-------------|
| `matrix_o` | Orthogonal matrix $O$ (eigenvectors) |
| `matrix_a` | Mapping matrix $A = O$ |
| `matrix_a_inv` | Inverse mapping $A^{-1} = O^T / M$ |
| `eigenvalues` | Eigenvalues $\lambda_i$ of $PXP$ or $U$ |
| `aux_fields_real_idx` | Indices of real auxiliary fields |
| `aux_fields_imag_idx` | Indices of imaginary auxiliary fields |

---

## 5. References

1. Arora, A., Morse, D. C., Bates, F. S. & Dorfman, K. D. "Accelerating self-consistent field theory of block polymers in a variable unit cell." *J. Chem. Phys.* **146**, 244902 (2017).

2. Düchs, D., Delaney, K. T. & Fredrickson, G. H. "A multi-species exchange model for fully fluctuating polymer field theory simulations." *J. Chem. Phys.* **141**, 174103 (2014).

3. Morse, D. C., Yong, D. & Chen, K. "Polymer Field Theory for Multimonomer Incompressible Models: Symmetric Formulation and ABC Systems." *Macromolecules* **58**, 816 (2025).
