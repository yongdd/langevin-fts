# Polymer Field Theory

This document describes the mathematical formulation of polymer field theory used in SCFT, L-FTS, and CL-FTS simulations. Three representations are covered: monomer potential fields, auxiliary fields (incompressible model), and auxiliary fields (compressible model). Sections 4 and 5 add smeared interactions and the electrostatic interactions of charged monomers.

## Table of Contents

1. [Monomer Potential Fields](#1-monomer-potential-fields)
2. [Incompressible Model](#2-incompressible-model)
3. [Compressible Model](#3-compressible-model)
4. [Smeared Interactions](#4-smeared-interactions)
5. [Electrostatic Interactions](#5-electrostatic-interactions)
6. [Grand-Canonical Species](#6-grand-canonical-species)
7. [Implementation](#7-implementation)
8. [References](#8-references)

---

## 1. Monomer Potential Fields

The monomer potential field formulation expresses the Helmholtz free energy directly in terms of monomer potential fields (e.g., $`w_A(\mathbf{r})`$, $`w_B(\mathbf{r})`$). This formulation is implemented in `scft.py`.

### 1.1 Helmholtz Free Energy

The canonical Helmholtz free energy includes the translational entropy of each polymer species. Let $`n_{\mathrm{ref}}=\rho_0 V/N`$, $`\alpha_p=N_p/N`$ and $`c_p=\bar\phi_p/\alpha_p=n_p/n_{\mathrm{ref}}`$. The single-chain partition function $`Q_p`$ is normalized by volume, so a free chain in zero fields has $`Q_p=1`$.

With segment-unit potential fields, the total free energy is

```math
\beta F_{\mathrm{tot}} = \sum_p n_p\left[\ln\left(\frac{c_p}{Q_p}\right)-1\right]
+\rho_0\int d\mathbf r\left[\frac12\sum_{ij}\chi_{ij}\phi_i\phi_j-\sum_i w_i\phi_i\right].
```

After rescaling $`Nw_i\to w_i`$, the full dimensionless free energy stored as `SCFT.free_energy_absolute` is

```math
F\equiv\frac{\beta F_{\mathrm{tot}}}{n_{\mathrm{ref}}}
=\sum_p c_p\left[\ln\left(\frac{c_p}{Q_p}\right)-1\right]
+\left\langle\frac12\boldsymbol\phi^T X\boldsymbol\phi-\mathbf w\cdot\boldsymbol\phi\right\rangle,
\qquad X_{ij}=\chi_{ij}N.
```

For the compressible model, add $`\frac{\zeta N}{2}\langle(\sum_i\phi_i-1)^2\rangle`$. The convention at $`c_p=0`$ is $`c_p\ln c_p=0`$.

The homogeneous value stored as `free_energy_homogeneous` is

```math
F_{\mathrm{hom}}=I+\frac12\bar{\boldsymbol\phi}^T X\bar{\boldsymbol\phi},
\qquad I=\sum_p c_p(\ln c_p-1).
```

`SCFT.free_energy` and the iteration log report the relative value

```math
\Delta F=F-F_{\mathrm{hom}}.
```

It is zero for a disordered bulk state. The final summary prints both relative and full values, and `free_energy_absolute = free_energy + free_energy_homogeneous`. Comparisons across compositions and grand-canonical transforms use the full value. The normalization is per reference-chain volume; the full free energy per actual molecule is $`F/(\sum_p c_p)`$. At the saddle point, with the auxiliary fields at their saddle-point means, $`F-I`$ equals the field-theoretic Hamiltonian including its constant term, as recorded by L-FTS/CL-FTS. The ideal term is independent of the fields in a fixed-composition CE run.

### 1.2 Self-Consistent Conditions

The self-consistent conditions for the potential fields are:

```math
w_i(\mathbf{r}) = \sum_{j \neq i} \chi_{ij} N \phi_j(\mathbf{r}) + \xi(\mathbf{r})
```

where $`\xi(\mathbf{r})`$ is a Lagrange multiplier enforcing incompressibility.

### 1.3 SCFT Iteration (Incompressible)

The field residuals $`\mathbf{R}_w`$ are defined as:

```math
\mathbf{R}_w = P_\perp \left( X \boldsymbol{\phi} - \mathbf{w} \right) + \kappa \, \mathbf{e} \left( \sum_i \phi_i - 1 \right), \qquad P_\perp = I - \frac{\mathbf{e}\mathbf{e}^T}{M}
```

where:
- $`X`$ is the $`\chi N`$ matrix
- $`\boldsymbol{\phi}`$ and $`\mathbf{w}`$ are vectors of concentrations and fields
- $`\kappa = \max(\max_k \vert\lambda_k\vert, 1) / 2`$, with $`\lambda_k`$ the eigenvalues of $`P_\perp X P_\perp`$ (see [SelfConsistentFieldTheory.md](SelfConsistentFieldTheory.md#22-field-residuals))

In addition to the $`\mathbf{k} \neq 0`$ residual update, the $`\mathbf{k}=0`$ mode (field mean) must be normalized correctly. The pressure field $`W_+`$ is shift-invariant (gauge degree of freedom), but the exchange field $`W_-`$ has a **physical mean** determined by the saddle-point condition. Simply setting $`\langle w_i \rangle = 0`$ violates the $`\mathbf{k}=0`$ self-consistency condition when $`f \neq 0.5`$.

The saddle condition for the $`\mathbf{k}=0`$ mode is:

```math
P_\perp \langle \mathbf{w} \rangle = P_\perp \left( X \langle \boldsymbol{\phi} \rangle \right)
```

where $`\langle \cdot \rangle`$ denotes the spatial mean. Since $`\langle \phi_i \rangle = f_i`$ exactly by chain normalization, the right-hand side is a constant. At each iteration, the $`P_\perp`$-projected mean of $`\mathbf{w}`$ is shifted to the target:

```math
\mathbf{w} \leftarrow \mathbf{w} + P_\perp \left( X \langle \boldsymbol{\phi} \rangle \right) - P_\perp \langle \mathbf{w} \rangle
```

This pins $`\langle W_- \rangle`$ to the saddle value while leaving $`\langle W_+ \rangle`$ untouched (since $`P_\perp`$ annihilates the uniform $`[1, 1, \ldots]`$ direction). For AB diblocks, the saddle gives $`\langle W_- \rangle = (1 - 2f) \chi N / 2`$. The $`P_\perp`$ and $`X`$ matrices are $`M \times M`$ for $`M`$ monomer types, so this applies to arbitrary multi-monomer systems.

> **Implementation note:** `scft.py` shifts every incompressible monomer field to zero spatial mean during iteration. Since the polymer compositions are fixed, constant shifts of the fields cancel between $`\ln Q_p`$ and $`\langle\mathbf w\cdot\boldsymbol\phi\rangle`$ in the canonical free energy above. Both full and relative free energies are independent of these shifts. Chemical potentials require restoring the saddle-point field means; this is done analytically after the CE calculation, without recomputing propagators. See [Phase Equilibrium from Canonical SCFT](PhaseEquilibrium.md) for the pressure convention, reference-species conversion and coexistence conditions.

### 1.4 Self-Consistent Conditions (Compressible)

With finite compressibility $`\zeta N`$, the pressure field is known:

```math
\xi(\mathbf{r}) = \zeta N \left[ \sum_j \phi_j(\mathbf{r}) - 1 \right]
```

```math
w_i(\mathbf{r}) = \sum_{j \neq i} \chi_{ij} N \phi_j(\mathbf{r}) + \xi(\mathbf{r})
```

In SCFT iteration, the residual is the compressible one, $`\mathbf{R}_w = X \boldsymbol{\phi} + \zeta N (\sum_j \phi_j - 1)\,\mathbf{e} - \mathbf{w}`$, with its stiff part along $`\mathbf{e}`$ rescaled by $`\kappa / (\kappa + \zeta N)`$ ($`\kappa`$ of Section 1.3). The field means, which have no gauge freedom here, are always pinned to the saddle-point values:

```math
\langle w_i \rangle = \sum_j X_{ij} \langle \phi_j \rangle + \xi, \qquad \xi = \zeta N \left( \left\langle \sum_j \phi_j \right\rangle - 1 \right)
```

See [SelfConsistentFieldTheory.md](SelfConsistentFieldTheory.md) for details on the compressible SCFT iteration.

### 1.5 Example: AB Diblock Copolymer

For AB-type systems, the full CE free energy is:

```math
F=\frac{\beta F_{\mathrm{tot}}}{n_{\mathrm{ref}}}
=\sum_p \frac{\bar\phi_p}{\alpha_p}
\left[\ln\left(\frac{\bar\phi_p}{\alpha_p Q_p}\right)-1\right]
+\frac{1}{V}\int d\mathbf r\left[
\chi_{AB}N\phi_A(\mathbf r)\phi_B(\mathbf r)
-w_A(\mathbf r)\phi_A(\mathbf r)-w_B(\mathbf r)\phi_B(\mathbf r)\right].
```

For a homogeneous pure AB melt with $`\alpha=1`$ and A fraction f, this gives $`F=\chi_{AB}Nf(1-f)-1`$; the reported relative value `free_energy` is zero.

Self-consistent conditions:

```math
w_A(\mathbf{r}) = \chi_{AB} N \phi_B(\mathbf{r}) + \xi(\mathbf{r})
```

```math
w_B(\mathbf{r}) = \chi_{AB} N \phi_A(\mathbf{r}) + \xi(\mathbf{r})
```

Field residuals (Section 1.3 with $`\kappa = \chi N / 2`$ for $`\chi N \geq 1`$):

```math
\mathbf{R}_w = \begin{pmatrix} 1/2 & -1/2 \\ -1/2 & 1/2 \end{pmatrix} \left[ \begin{pmatrix} 0 & \chi N \\ \chi N & 0 \end{pmatrix} \begin{pmatrix} \phi_A(\mathbf{r}) \\ \phi_B(\mathbf{r}) \end{pmatrix} - \begin{pmatrix} w_A(\mathbf{r}) \\ w_B(\mathbf{r}) \end{pmatrix} \right] + \frac{\chi N}{2} \begin{pmatrix} 1 \\ 1 \end{pmatrix} \left( \phi_A(\mathbf{r}) + \phi_B(\mathbf{r}) - 1 \right)
```

---

## 2. Incompressible Model

The incompressible model uses auxiliary fields to decouple the interaction terms. This formulation is implemented in `polymer_field_theory.py` and used by both `scft.py` and `lfts.py`.

### 2.1 Partition Function

The canonical partition function with incompressibility constraint:

```math
\mathcal{Z} \propto \int \{\mathcal{D}\mathbf{r}_i\} \exp(-U_{id} - U_{int}) \, \delta\left[\sum_{i=1}^{M} \rho_i(\mathbf{r}) - \rho_0\right]
```

where:
- $`U_{id}`$: intramolecular potential energy
- $`U_{int}`$: intermolecular interaction energy

```math
U_{int} = \frac{1}{2\rho_0} \int d\mathbf{r} \sum_{i,j} \chi_{ij} \rho_i(\mathbf{r}) \rho_j(\mathbf{r})
```

### 2.2 Field-Theoretic Representation

After Hubbard-Stratonovich transformation:

```math
\mathcal{Z} \propto \int \{\mathcal{D}\Omega_i\} \exp(-\beta H[\{\Omega_i\}])
```

### 2.3 Effective Hamiltonian

**Per chain unit:**

```math
\frac{\beta H}{C V / R_0^3} = -\sum_p \frac{\bar{\phi}_p}{\alpha_p} \log Q_p + \frac{1}{V} \int d\mathbf{r} \left[ -\sum_{i=1}^{M-1} \frac{M \Omega_i^2}{2\lambda_i} + \sum_{i=1}^{M-1} \frac{\Omega_i S_i}{\lambda_i} - \Omega_M + \frac{1}{2M} \left( \mathbf{s}^T \mathbf{e} - \sum_{i=1}^{M-1} \frac{S_i^2}{\lambda_i} \right) \right]
```

where:
- $`C = \rho_0 R_0^3 / N`$: dimensionless chain number density
- $`\bar{N} = C^2`$: invariant polymerization index
- $`P_\perp = I - \mathbf{e}\mathbf{e}^T / M`$: orthogonal projector onto the composition modes (as in Section 1.3)
- $`O`$: orthogonal matrix of eigenvectors of $`P_\perp X P_\perp`$ with $`OO^T = M`$ (each column has length $`\sqrt{M}`$; the last column is $`\mathbf{e}`$, the pressure field $`\Omega_M`$)
- $`\mathbf{s}^T = \mathbf{e}^T X / M`$ and $`\mathbf{S} = O^T \mathbf{s}`$, with the same $`O`$ as in the field transformation below
- $`\lambda_i`$: eigenvalues of $`P_\perp X P_\perp`$

### 2.4 Field Transformations

The relations between auxiliary fields $`\{\Omega_i\}`$ and monomer potential fields $`\{W_i\}`$:

```math
\begin{pmatrix} W_1(\mathbf{r}) \\ W_2(\mathbf{r}) \\ \vdots \\ W_M(\mathbf{r}) \end{pmatrix} = O \begin{pmatrix} \Omega_1(\mathbf{r}) \\ \Omega_2(\mathbf{r}) \\ \vdots \\ \Omega_M(\mathbf{r}) \end{pmatrix}
```

### 2.5 Example: AB Diblock (Incompressible)

For AB-type systems, the auxiliary fields are:
- $`\Omega_-`$: exchange field (composition fluctuations)
- $`\Omega_+`$: pressure-like field (incompressibility)

**Effective Hamiltonian:**

```math
\frac{\beta H}{C V / R_0^3} = -\sum_p \frac{\bar{\phi}_p}{\alpha_p} \log Q_p + \frac{1}{V} \int d\mathbf{r} \left[ \frac{\Omega_-^2(\mathbf{r})}{\chi N} - \Omega_+(\mathbf{r}) + \frac{\chi N}{4} \right]
```

**Functional derivatives:**

```math
\frac{\beta}{C/R_0^3} \frac{\delta H}{\delta \Omega_-(\mathbf{r})} = \Phi_-(\mathbf{r}) + \frac{2}{\chi N} \Omega_-(\mathbf{r})
```

```math
\frac{\beta}{C/R_0^3} \frac{\delta H}{\delta \Omega_+(\mathbf{r})} = \Phi_+(\mathbf{r}) - 1
```

**Transformation matrix:**

```math
O = \begin{pmatrix} 1 & 1 \\ -1 & 1 \end{pmatrix}
```

```math
\begin{pmatrix} W_A(\mathbf{r}) \\ W_B(\mathbf{r}) \end{pmatrix} = O \begin{pmatrix} \Omega_-(\mathbf{r}) \\ \Omega_+(\mathbf{r}) \end{pmatrix}, \quad \begin{pmatrix} \Phi_-(\mathbf{r}) \\ \Phi_+(\mathbf{r}) \end{pmatrix} = O^T \begin{pmatrix} \phi_A(\mathbf{r}) \\ \phi_B(\mathbf{r}) \end{pmatrix}
```

so that $`\Phi_- = \phi_A - \phi_B`$ and $`\Phi_+ = \phi_A + \phi_B`$, consistent with $`\Phi_i = \sum_j O_{ji} \phi_j`$ in Section 7.1 and with the incompressibility condition $`\delta H / \delta \Omega_+ \propto \Phi_+ - 1`$.

---

## 3. Compressible Model

The compressible model includes a finite compressibility parameter $`\zeta`$. This formulation is implemented in `polymer_field_theory.py` and may be used for complex Langevin FTS.

### 3.1 Interaction Energy

```math
U_{int} = \frac{1}{2\rho_0} \int d\mathbf{r} \left\lbrace \sum_{i,j} \chi_{ij} \rho_i(\mathbf{r}) \rho_j(\mathbf{r}) + \zeta \left[ \sum_{i=1}^{M} \rho_i(\mathbf{r}) - \rho_0 \right]^2 \right\rbrace
```

### 3.2 Effective Hamiltonian

**Per chain unit:**

```math
\frac{\beta H}{C V / R_0^3} = -\sum_p \frac{\bar{\phi}_p}{\alpha_p} \log Q_p + \frac{1}{V} \int d\mathbf{r} \left[ -\sum_{i=1}^{M} \frac{M \Omega_i^2}{2\lambda_i} + \sum_{i=1}^{M} \frac{\Omega_i S_i}{\lambda_i} + \frac{1}{2} \left( \zeta N - \sum_{i=1}^{M} \frac{S_i^2}{M\lambda_i} \right) \right]
```

where:
- $`O`$: orthogonal matrix of $`U = X + \zeta \mathbf{e}\mathbf{e}^T`$ with $`OO^T = M`$
- $`\mathbf{S}^T = -\zeta \mathbf{e}^T O`$

### 3.3 Example: AB Diblock (Compressible)

**Effective Hamiltonian:**

```math
\frac{\beta H}{C V / R_0^3} = -\sum_p \frac{\bar{\phi}_p}{\alpha_p} \log Q_p + \frac{1}{V} \int d\mathbf{r} \left[ \frac{\Omega_-^2(\mathbf{r})}{\chi N} - \frac{\Omega_+^2(\mathbf{r})}{\chi N + 2\zeta N} - \frac{2\zeta N}{\chi N + 2\zeta N} \Omega_+(\mathbf{r}) + \frac{\chi N \cdot \zeta N}{2(\chi N + 2\zeta N)} \right]
```

**Functional derivatives:**

```math
\frac{\beta}{C/R_0^3} \frac{\delta H}{\delta \Omega_-(\mathbf{r})} = \Phi_-(\mathbf{r}) + \frac{2}{\chi N} \Omega_-(\mathbf{r})
```

```math
\frac{\beta}{C/R_0^3} \frac{\delta H}{\delta \Omega_+(\mathbf{r})} = \Phi_+(\mathbf{r}) - \frac{2}{\chi N + 2\zeta N} (\Omega_+(\mathbf{r}) + \zeta N)
```

---

## 4. Smeared Interactions

With contact interactions, the renormalization of $`\chi`$ by the fluctuations grows without bound as the grid is refined ([Langevin FTS, Section 5.1](LangevinFieldTheoreticSimulation.md#51-effective-interaction-and-chain-size)). Smearing gives the interactions a finite range (References 4 and 5); with a smearing function that decays at large $`k`$, this correction converges. `smearing.py` implements it for SCFT, L-FTS and CL-FTS; the [Parameter Reference](../references/Parameters.md#smearing) lists the inputs and restrictions.

### 4.1 Smeared Densities

The monomers interact through their smeared densities

```math
\rho_i^\Gamma(\mathbf{r}) = \int \Gamma_i(\mathbf{r} - \mathbf{r}') \, \rho_i(\mathbf{r}') \, d\mathbf{r}', \qquad \tilde\rho_i^\Gamma(\mathbf{k}) = \Gamma_i(\mathbf{k}) \, \tilde\rho_i(\mathbf{k})
```

which replace $`\rho_i`$ in the interaction energies of Sections 2.1 and 3.1 and in the incompressibility constraint of Section 2.1. $`\Gamma_i(\mathbf{r}) = \Gamma_i(-\mathbf{r})`$, so a field couples to a smeared density as the smeared field couples to the density:

```math
\int \Omega \, (\Gamma_i * \rho_i) \, d\mathbf{r} = \int (\Gamma_i * \Omega) \, \rho_i \, d\mathbf{r}
```

The Hubbard-Stratonovich transformation therefore leaves the quadratic terms of the effective Hamiltonians of Sections 2.3 and 3.2 unchanged. The chains feel the smeared fields, and the functional derivatives contain the smeared concentrations:

```math
Q_p = Q_p\big[\{\Gamma_i * W_i\}\big], \qquad \Phi_i(\mathbf{r}) \;\to\; \Phi_i^\Gamma(\mathbf{r}) = \sum_j O_{ji} \, (\Gamma_j * \phi_j)(\mathbf{r})
```

In the incompressible model, the pressure field therefore constrains the smeared total density, $`\sum_j \Gamma_j * \phi_j = 1`$. With a single $`\Gamma`$ that does not vanish, this is the same as $`\sum_j \phi_j = 1`$.

### 4.2 Smearing Functions

$`\Gamma(0) = 1`$, so smearing conserves the amount of each type.

| `type` | $`\Gamma(\mathbf{k})`$ | Parameters |
|--------|------------------------|------------|
| `gaussian` | $`\exp(-a^2 k^2 / 2)`$, the transform of $`(2\pi a^2)^{-3/2} \exp(-r^2 / 2a^2)`$ | range $`a`$ (`a_int`) in units of $`R_0`$ |
| `sigmoidal` | $`C_0 \left[ 1 - \tanh\left( (k - k_c)/\Delta k \right) \right]`$, $`C_0 = 1/\left[ 1 - \tanh(-k_c/\Delta k) \right]`$ | cutoff $`k_c`$ (`k_int`), width $`\Delta k`$ (`dk_int`) |

- **Wavenumber:** $`k`$ is the physical wavenumber of the cell, including the cell angles. $`\Gamma`$ follows the box, so the SCFT stress contains the derivative of $`\Gamma`$.
- **Walls:** along an axis with reflecting or absorbing walls, fields and concentrations are even across the wall, so they are smeared together with their mirror image, in the cosine (DCT-II) basis.
- **Per type:** CL-FTS accepts one Gaussian range $`a_i`$ for every type, which gives each type its own $`\Gamma_i`$. One $`\Gamma`$ applies to all types otherwise.

Smearing also enters the structure functions, whose recorded values are divided by $`\Gamma_i`$ ([Langevin FTS, Section 4.2](LangevinFieldTheoreticSimulation.md#42-estimator)), their RPA ([Section 4.4](LangevinFieldTheoreticSimulation.md#44-rpa-reference)) and the charge density of Section 5.

---

## 5. Electrostatic Interactions

Monomer type $`i`$ may carry the valence $`z_i`$ per segment (`electrostatics`). Small ions are molecules of one segment, which count in the total density and interact through $`\chi`$ like the other species, or point ions without volume and without $`\chi`$ (Section 5.5). The charges add a Coulomb interaction to the interactions of Sections 2 and 3. `electrostatics.py` implements it; CL-FTS samples the electrostatic potential ([CL-FTS, Section 6](ComplexLangevinSimulation.md#6-charged-systems)), and SCFT takes its saddle point ([SCFT, Section 2.5](SelfConsistentFieldTheory.md#25-charged-systems)).

### 5.1 Coulomb Interaction

The charges interact through the smeared charge density and the Coulomb kernel:

```math
c(\mathbf{r}) = \sum_i z_i \, (\Gamma_i * \phi_i)(\mathbf{r}), \qquad \frac{\beta U_{el}}{C V / R_0^3} = \frac{E}{2V} \int c \, (-\nabla^2)^{-1} c \, d\mathbf{r}
```

Here $`\Gamma_i`$ is the smearing function of type $`i`$ (Section 4); one Gaussian length per type plays the role of a Born radius. Point charges would have a self-energy that depends on the grid, so every charged type needs a positive smearing length.

### 5.2 Coupling Constant

In units of $`k_B T`$, charges interact through the kernel $`l_B / |\mathbf{r} - \mathbf{r}'|`$, where $`l_B = e^2 / (4\pi \varepsilon k_B T)`$ is the Bjerrum length. The operator $`(-\nabla^2)^{-1}`$ has the kernel $`1 / (4\pi |\mathbf{r} - \mathbf{r}'|)`$, so the charge number density $`\rho_0 c(\mathbf{r})`$ has the energy

```math
\beta U_{el} = \frac{l_B}{2} \iint \frac{\rho_0 c(\mathbf{r}) \, \rho_0 c(\mathbf{r}')}{|\mathbf{r} - \mathbf{r}'|} \, d\mathbf{r} \, d\mathbf{r}' = \frac{4\pi l_B \rho_0^2}{2} \int c \, (-\nabla^2)^{-1} c \, d\mathbf{r}
```

Measuring lengths in units of $`R_0`$ multiplies the volume element by $`R_0^3`$ and $`(-\nabla^2)^{-1}`$ by $`R_0^2`$. Dividing by the number of reference chains, $`C V / R_0^3 = \rho_0 V / N`$ with $`C = \rho_0 R_0^3 / N`$ (Section 2.3), gives the form of Section 5.1 with the coupling

```math
E = 4\pi l_B \rho_0 N R_0^2 = 4\pi (l_B/R_0) N^2 \bar{N}^{1/2}
```

The second form uses $`\rho_0 R_0^3 = N C = N \bar{N}^{1/2}`$, which follows from $`R_0 = b N^{1/2}`$ and $`\bar{N} = b^6 \rho_0^2 N`$. The factor $`N^2`$ comes from the valences per segment. A volume $`R_0^3`$ holds the charge $`N \bar{N}^{1/2} c`$, whose Coulomb energy $`l_B (N \bar{N}^{1/2} c)^2 / R_0`$, shared by the $`\bar{N}^{1/2}`$ chains in that volume, is of order $`E c^2`$ per chain.

`bjerrum_length` takes $`N = 1/\Delta s`$ for discrete chains. For continuous chains, $`\Delta s`$ is only a contour step: `n_segments` gives the physical $`N`$, or `coupling` gives $`E`$ directly.

### 5.3 Electrostatic Potential

A Hubbard-Stratonovich transformation decouples the interaction with an electrostatic potential $`\psi`$. Like the pressure field, $`\psi`$ is an imaginary-type field. Each type feels $`z_i \psi`$, so the propagator input becomes $`\Gamma_i * (w_i + z_i \psi)`$. The effective Hamiltonian of Sections 2.3 and 3.2 gains the term

```math
\frac{\beta H_{el}}{C V / R_0^3} = \frac{1}{2 E V} \int \psi \, \nabla^2 \psi \, d\mathbf{r}, \qquad \frac{\delta H}{\delta \psi} = \frac{\nabla^2 \psi}{E} + c
```

The saddle point is the Poisson equation $`-\nabla^2 \psi = E c`$. The uniform mode of $`\psi`$ is zero, because the system is electroneutral.

**Fixed charges.** A fixed charge density $`c_f(\mathbf{r})`$ (`fixed_charge`, charges per segment of $`\rho_0`$), for example of charged surfaces or particles, has no partition function through which it could couple to $`\psi`$; it adds $`\langle \psi \, c_f \rangle`$ to $`\beta H/(C V/R_0^3)`$, so that $`c`$ in $`\delta H/\delta \psi`$ includes $`c_f`$. Its mean counts in the electroneutrality.

**Closed boxes.** In a box with walls on every side the transforms are cosine series, which diagonalize $`\nabla^2`$ with no normal derivative at the walls: $`\psi`$ then describes a box with no field through its walls. This is the boundary to a medium of vanishing dielectric constant: behind each wall every charge has an image of the same sign, so the fluctuations of $`\psi`$ deplete ions next to the walls (Onsager–Samaras). In mean-field theory a laterally uniform neutral system has the same field as without the walls. The uniform mode is the $`m = 0`$ term of the series.

### 5.4 Screened Coulomb Interaction

An electrolyte that is not simulated explicitly (a background salt) screens the charges, and the kernel becomes the Yukawa interaction $`l_B e^{-\kappa r}/r`$ with the screening length $`1/\kappa`$ (`screening_length`, in units of $`R_0`$):

```math
\frac{\beta U_{el}}{C V / R_0^3} = \frac{E}{2V} \int c \, (\kappa^2 - \nabla^2)^{-1} c \, d\mathbf{r}, \qquad \frac{\beta H_{el}}{C V / R_0^3} = \frac{1}{2 E V} \int \psi \, (\nabla^2 - \kappa^2) \, \psi \, d\mathbf{r}
```

The saddle point is the screened Poisson equation $`(\kappa^2 - \nabla^2) \psi = E c`$. The kernel $`E/(k^2 + \kappa^2)`$ is finite at $`k = 0`$, so the uniform mode of $`\psi`$ fluctuates and a net charge of the explicit species is allowed: the background electrolyte neutralizes it. With $`\kappa = 0`$ this is Section 5.1.

In Debye-Hückel theory $`1/\kappa`$ is the Debye length of the background electrolyte alone, $`\kappa^2 = 4\pi l_B \sum_s z_s^2 c_s`$ over its ion species with number concentrations $`c_s`$ (in units of $`R_0`$: $`(\kappa R_0)^2 = 4\pi (l_B/R_0) \sum_s z_s^2 c_s R_0^3`$). It is not the screening length of the system: the explicit ions, molecules in `polymers`, screen in addition through the fluctuations of $`\psi`$, so the total screening length is shorter. The background enters only through its linear response, so the correlations among its ions, their volume and the nonlinear response of multivalent ions are left out; ions for which these matter belong in `polymers`.

### 5.5 Point Ions

Point ions (`ions`) have a charge but no volume: they take no part in $`\chi`$, in the incompressibility constraint or in the compressibility term. Ion species $`s`$ with valence $`z_s`$, Gaussian smearing function $`\Gamma_s`$ and concentration $`c_s`$ (ions per segment of $`\rho_0`$) feels the smeared potential $`z_s\,(\Gamma_s * \psi)/N`$: $`\psi`$ is a field per reference chain, so a segment feels $`z_i \psi / N`$, and an ion is charged like a segment. With

```math
Q_s = \frac{1}{V} \int e^{-z_s (\Gamma_s * \psi)(\mathbf{r})/N} \, d\mathbf{r}, \qquad c_s(\mathbf{r}) = c_s \, \frac{e^{-z_s (\Gamma_s * \psi)(\mathbf{r})/N}}{Q_s}
```

the $`N c_s C V/R_0^3`$ ions add to the effective Hamiltonian and to the charge density

```math
\frac{\beta H_{ion}}{C V / R_0^3} = -\sum_s N c_s \ln Q_s, \qquad c(\mathbf{r}) \;\to\; c(\mathbf{r}) + \sum_s z_s \, (\Gamma_s * c_s)(\mathbf{r})
```

so that $`\delta H/\delta \psi`$ of Section 5.3 keeps its form with the total charge density. The chemical potential of an ion, in the convention of the chains ($`\beta\mu_p = \ln(\bar\phi_p/\alpha_p) - \ln Q_p`$ with $`\bar\phi_s/\alpha_s = N c_s`$), is

```math
\beta \mu_s = \ln(N c_s) - \ln Q_s
```

In the linear response of a homogeneous state the ions screen with $`\sum_s z_s^2 c_s \Gamma_s(k)^2/N`$, which adds to $`(k^2 + \kappa^2)/E`$ (Debye-Hückel). The counterions that neutralize the polymers are added to one ion species (`counterion`). The concentrations of the given ions are those of the added salt.

---

## 6. Grand-Canonical Species

CL-FTS can fix the chemical potential of a species instead of its amount (`chemical_potential` of a polymer or of a point ion). For polymer $`p`$, the sum over the number of chains $`n_p`$ with the weight $`e^{\beta\mu_p n_p}`$ replaces the factor $`(V Q_p)^{n_p}/n_p!`$ of the canonical partition function by $`\exp(e^{\beta\mu_p} \, C V Q_p/R_0^3)`$, so that in the effective Hamiltonian

```math
-\frac{\bar\phi_p}{\alpha_p} \ln Q_p \;\to\; -e^{\beta\mu_p} Q_p, \qquad \frac{n_p}{C V/R_0^3} = e^{\beta\mu_p} Q_p, \qquad \phi_p(\mathbf{r}) = e^{\beta\mu_p} \int_0^{\alpha_p} q(\mathbf{r}, s) \, q^\dagger(\mathbf{r}, s) \, ds
```

per $`C V/R_0^3`$ reference chains. $`\beta\mu_p`$ is in the convention of the canonical operator, $`\beta\mu_p = \ln(\bar\phi_p/\alpha_p) - \ln Q_p`$, which leaves out the constant $`\ln(\rho_0/N)`$ common to all species. On the same fields the two ensembles give the same concentrations where $`e^{\beta\mu_p} = \bar\phi_p/(\alpha_p Q_p)`$. In a reservoir of the species alone without fields, $`Q_p = 1`$ and $`e^{\beta\mu_p}`$ is its number of chains per reference chain. The volume fractions of the canonical polymers sum to at most 1; the grand-canonical chains take the rest on average, through the pressure field.

A point ion (Section 5.5) with the chemical potential $`\beta\mu_s`$ has, in the same way,

```math
-N c_s \ln Q_s \;\to\; -e^{\beta\mu_s} Q_s, \qquad c_s(\mathbf{r}) = \frac{e^{\beta\mu_s}}{N} \, e^{-z_s (\Gamma_s * \psi)(\mathbf{r})/N}
```

with $`e^{\beta\mu_s} Q_s`$ ions per reference chain, and $`e^{\beta\mu_s}/N`$ is the concentration of a reservoir at $`\psi = 0`$.

**Electroneutrality.** In a closed system without screening the uniform mode $`\psi_0`$ of $`\psi`$ is zero, and the input must be neutral (Section 5.3). A charged grand-canonical species changes the charge of the system, so $`\psi_0`$ is integrated like the other modes. Its integral restricts the sum over the numbers of particles to neutral states. It has no Gaussian term, and its normalization does not depend on the volume. In the mean-field limit $`\psi_0`$ is the Donnan potential at which the average charge vanishes. A change of the chemical potentials of all species by their charge times $`\lambda`$ ($`z_s \lambda`$ for an ion, $`N \sum_i z_i \alpha_{p,i} \lambda`$ for a chain with the length $`\alpha_{p,i}`$ of type $`i`$) is absorbed by the shift $`\psi_0 \to \psi_0 + N\lambda`$, so only neutral combinations of chemical potentials are physical, as in a reservoir of neutral salt. With screening $`\psi_0`$ fluctuates in any case.

**Example.** A canonical polyelectrolyte with the charge $`z_A`$ per A segment and the overall A fraction $`\bar\phi_A`$ in a monovalent salt with $`\beta\mu_\pm = \ln(N c_0)`$ (a reservoir of concentration $`c_0`$) has, at the homogeneous saddle point, $`c_\pm = c_0 x^{\pm 1}`$ with $`x = e^{-\psi_0/N}`$ and

```math
z_A \bar\phi_A + c_0 \left(x - \frac{1}{x}\right) = 0, \qquad x = \frac{-z_A \bar\phi_A/c_0 + \sqrt{(z_A \bar\phi_A/c_0)^2 + 4}}{2}
```

the Donnan equilibrium; `TestClftsGrandCanonical` checks that CL-FTS reaches it in the mean-field limit.

---

## 7. Implementation

### 7.1 SymmetricPolymerTheory Class

The `polymerfts.SymmetricPolymerTheory` class computes the coefficients for the effective Hamiltonian:

```math
\frac{\beta H}{C V / R_0^3} = -\sum_p \frac{\bar{\phi}_p}{\alpha_p} \log Q_p + \frac{1}{V} \int d\mathbf{r} \left[ \sum_{i=1}^{M} a_i \Omega_i^2(\mathbf{r}) + \sum_{i=1}^{M} b_i \Omega_i(\mathbf{r}) + U_{ref} \right]
```

**Functional derivative:**

```math
\frac{\beta}{C/R_0^3} \frac{\delta H}{\delta \Omega_i(\mathbf{r})} = \Phi_i(\mathbf{r}) + 2 a_i \Omega_i(\mathbf{r}) + b_i
```

where $`\Phi_i(\mathbf{r}) = \sum_j O_{ji} \phi_j(\mathbf{r})`$.

### 7.2 Usage Example

```python
import numpy as np
import polymerfts

# Define system
monomer_types = ["A", "B"]
chi_n = {"A,B": 20}

# Create theory object (incompressible)
mpt = polymerfts.SymmetricPolymerTheory(monomer_types, chi_n, zeta_n=None)

# Create theory object (compressible)
mpt_comp = polymerfts.SymmetricPolymerTheory(monomer_types, chi_n, zeta_n=100)

# Convert between field representations
n_grid = 32**3                        # number of grid points
omega = np.random.rand(2, n_grid)
w = mpt.to_monomer_fields(omega)      # Auxiliary → Monomer
omega = mpt.to_aux_fields(w)          # Monomer → Auxiliary

# Compute Hamiltonian
# molecules: C++ Molecules object; total_partitions: list of Q_p;
# cb: ComputationBox object (required, used for volume averages)
# The constant U_ref is left out unless include_const_term=True
H = mpt.compute_hamiltonian(molecules, omega, total_partitions, cb, include_const_term=True)

# Compute functional derivatives
h_deriv = mpt.compute_func_deriv(omega, phi, field_indices)

# dH/dχN at fixed auxiliary fields (Section 7.5); phi: {monomer_type: concentration}
dH = mpt.compute_h_deriv_chin(w_aux=omega, phi=phi)
```

### 7.3 Sign Convention for Imaginary Fields

In the implementation, there is a negative sign in front of the functional derivative with respect to imaginary fields. This is because we find the minimum for real fields and maximum for imaginary fields:

```python
h_deriv = mpt.compute_func_deriv(omega, phi, [0, 1])
# h_deriv[0] = +δH/δΩ_- (real field, minimize)
# h_deriv[1] = -δH/δΩ_+ (imaginary field, maximize)
```

### 7.4 Key Attributes

| Attribute | Description |
|-----------|-------------|
| `matrix_chi` | Interaction matrix $`X`$ ($`\chi N`$, zero diagonal) in the order of `monomer_types` |
| `matrix_o` | Orthogonal matrix $`O`$ (eigenvectors) |
| `matrix_a` | Mapping matrix $`A = O`$ |
| `matrix_a_inv` | Inverse mapping $`A^{-1} = O^T / M`$ |
| `eigenvalues` | Eigenvalues $`\lambda_i`$ of $`P_\perp X P_\perp`$ or $`U`$ |
| `aux_fields_real_idx` | Indices of real auxiliary fields |
| `aux_fields_imag_idx` | Indices of imaginary auxiliary fields |

### 7.5 Derivative with Respect to χN

`compute_h_deriv_chin(w_aux=w_aux, phi=phi)` returns $`\partial (\beta H / (C V / R_0^3)) / \partial \chi_{ab} N`$ at fixed auxiliary fields $`\Omega`$, the integration variables. Averaged over the real fields with the imaginary fields at their saddle point (L-FTS), it is the derivative of the free energy of that approximation, because $`\partial H / \partial \Omega_I = 0`$ at the saddle point of the imaginary fields $`\Omega_I`$; with every field fluctuating (CL-FTS) it is the derivative of the full free energy; at a saddle point of all fields it equals $`\langle \phi_a \phi_b \rangle`$. The $`\chi`$-dependent normalization of the Hubbard-Stratonovich transformation is a constant independent of the fields and is not included. `phi` holds the concentrations of the fields (smeared, when smearing is used); it may be omitted for AB systems at $`\chi N \neq 0`$. The derivatives use the $`\chi N`$ the object was built with; the older call `compute_h_deriv_chin(chi_n, w_aux, phi)` still works and checks that `chi_n` equals it.

**Moving frame.** Let $`U = P_\perp X P_\perp`$ (incompressible) or $`U = X + \zeta \mathbf{e}\mathbf{e}^T`$ (compressible), $`\mathbf{o}_k`$ the $`k`$-th column of $`O`$ ($`U \mathbf{o}_k = \lambda_k \mathbf{o}_k`$, $`\vert\mathbf{o}_k\vert^2 = M`$), and $`dU = \partial U / \partial \chi_{ab} N`$, i.e. $`P_\perp E P_\perp`$ or $`E`$ with $`E_{ab} = E_{ba} = 1`$ and all other entries 0. When $`\chi`$ changes, the auxiliary fields follow a frame $`\tilde{O}(\chi)`$ with $`\tilde{O} = O`$ at the given $`\chi`$ and $`\partial \tilde{O} / \partial \chi_{ab} N = O K`$, where $`K`$ is antisymmetric:

```math
K_{jk} = \frac{\mathbf{o}_j^T \, dU \, \mathbf{o}_k}{M (\lambda_k - \lambda_j)}
```

if $`j`$ and $`k`$ have different roles (real fields $`\lambda < 0`$, imaginary fields $`\lambda > 0`$, dropped fields $`\lambda = 0`$), and $`K_{jk} = 0`$ otherwise (two fields of the same role, or the pressure field $`\Omega_M`$, for which $`P_\perp \mathbf{o}_M = 0`$). Between roles $`K`$ is the first-order motion of the eigenvectors: it moves the subspaces on which the real and the imaginary fields live, and its denominators never vanish. Within a role the frame is free: a rotation of the real fields among themselves (orthogonal, unit Jacobian) or of the imaginary fields among themselves (the saddle point does not depend on how it is parametrized) changes single values of the derivative but not their average, and $`K = 0`$ there needs no difference of nearly equal eigenvalues. With `wtmd=True` (used for the WTMD statistics) the leading real field (`aux_fields_real_idx[0]`) also follows its own eigenvector, the formula above applying to its pairs with the other real fields, so that an order parameter built from it stays fixed at fixed $`\Omega`$; single values of the derivative then grow as the inverse gap when its eigenvalue nears another real one (the average does not change), and $`K = 0`$ for a degenerate pair, whose eigenvectors are undefined.

**Coefficients.** In this frame $`\Lambda = \tilde{O}^T U \tilde{O} / M`$ stays block diagonal between the roles, $`\Lambda = \mathrm{diag}(\lambda)`$ at the given $`\chi`$, and with $`\mathbf{S} = \tilde{O}^T \mathbf{s}`$ ($`\mathbf{s} = -\zeta \mathbf{e}`$ for the compressible model)

```math
d\Lambda = \Lambda K - K \Lambda + \frac{1}{M} O^T dU \, O, \qquad d\mathbf{S} = -K \mathbf{S} + O^T d\mathbf{s},
```

with $`d\mathbf{s} = E \mathbf{e} / M`$ (incompressible) or 0 (compressible). Over the fields with $`\lambda \neq 0`$ (index set $`\kappa`$) the field part of Sections 2.3 and 3.2 is $`-\frac{M}{2} \Omega^T \Lambda^{-1} \Omega + (\Lambda^{-1} \mathbf{S}) \cdot \Omega - \frac{1}{2M} \mathbf{S}^T \Lambda^{-1} \mathbf{S}`$ plus a constant, so

```math
\frac{\partial}{\partial \chi_{ab} N} \frac{\beta H}{C V / R_0^3} = \frac{1}{V} \int d\mathbf{r} \left[ \frac{M}{2} \sum_{j,k \in \kappa} \frac{\Omega_j \, d\Lambda_{jk} \, \Omega_k}{\lambda_j \lambda_k} + \sum_{k \in \kappa} \frac{\Omega_k}{\lambda_k} \left( dS_k - \sum_{j \in \kappa} \frac{d\Lambda_{kj} S_j}{\lambda_j} \right) + \sum_{j,k} \phi_j (O K)_{jk} \Omega_k \right] - \sum_{k \in \kappa} \frac{dS_k S_k}{M \lambda_k} + \sum_{j,k \in \kappa} \frac{S_j \, d\Lambda_{jk} \, S_k}{2 M \lambda_j \lambda_k} + dc
```

where $`dc = \mathbf{e}^T E \mathbf{e} / (2M^2)`$ is the derivative of $`\mathbf{s}^T \mathbf{e} / (2M)`$ (incompressible; 0 for the compressible model). The term with $`O K`$ is the change of $`-\sum_p (\bar{\phi}_p / \alpha_p) \log Q_p`$ through $`W = \tilde{O} \Omega`$.

**Dropped fields.** A field with $`\lambda = 0`$ other than the pressure field (index set $`\delta`$) has no field and no term in $`H`$ (its row of `w_aux` is not used). Its mean-field part $`\frac{1}{2M} \mathbf{y}^T \Lambda_{\delta\delta} \mathbf{y} + \frac{1}{M} \mathbf{S}_\delta \cdot \mathbf{y}`$, with $`y_k = \mathbf{o}_k \cdot \boldsymbol{\phi}`$, vanishes at the given $`\chi`$ when $`\mathbf{S}_\delta = 0`$, but its derivative does not:

```math
\frac{1}{V} \int d\mathbf{r} \left[ \frac{1}{2M} \sum_{j,k \in \delta} y_j \, d\Lambda_{jk} \, y_k + \frac{1}{M} \sum_{k \in \delta} \left( dS_k \, y_k + S_k \, dy_k \right) \right], \qquad dy_k = \sum_j K_{jk} \, y_j
```

is added, and the columns of $`O K`$ of the dropped fields are left out of the rotation term (for AB at $`\chi N = 0`$ it is $`-\langle (\phi_A - \phi_B)^2 \rangle / 4`$). In a fluctuating simulation the exact term holds the correlation of the microscopic densities in place of $`y_j y_k`$, which also has an intrachain part of order $`1/C`$ that no field represents; that part is missing.

**AB systems.** $`K = 0`$ and $`d\Lambda = d\lambda`$; for the incompressible model

```math
\frac{\partial}{\partial \chi N} \frac{\beta H}{C V / R_0^3} = \frac{1}{4} - \frac{1}{V} \int d\mathbf{r} \frac{\Omega_-^2(\mathbf{r})}{(\chi N)^2}.
```

---

## 8. References

1. Arora, A., Morse, D. C., Bates, F. S. & Dorfman, K. D. "Accelerating self-consistent field theory of block polymers in a variable unit cell." *J. Chem. Phys.* **146**, 244902 (2017).

2. Düchs, D., Delaney, K. T. & Fredrickson, G. H. "A multi-species exchange model for fully fluctuating polymer field theory simulations." *J. Chem. Phys.* **141**, 174103 (2014).

3. Morse, D. C., Yong, D. & Chen, K. "Polymer Field Theory for Multimonomer Incompressible Models: Symmetric Formulation and ABC Systems." *Macromolecules* **58**, 816 (2025).

4. Delaney, K. T. & Fredrickson, G. H. "Recent Developments in Fully Fluctuating Field-Theoretic Simulations of Polymer Melts and Solutions." *J. Phys. Chem. B* **120**, 7615 (2016).

5. Matsen, M. W., Ramírez, J., Willis, J. D. & Pina, P. D. "Accessing the universal phase behavior of block copolymer melts with complex-Langevin field-theoretic simulations." *J. Chem. Phys.* **164**, 014905 (2026).
