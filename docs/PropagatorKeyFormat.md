# Propagator Code and Key Format Specification

> **⚠️ Warning:** This document was generated with assistance from a large language model (LLM). While it is based on the referenced literature and the codebase, it may contain errors, misinterpretations, or inaccuracies. Please verify the equations and descriptions against the original references before relying on this document for research or implementation.

This document describes the string representations used to identify and manage propagator computations in the polymer field theory simulation library.

## Overview

Propagators are identified using two related string formats:
- **Code format (DKN)**: Used internally during code generation
- **Key format (DK+M)**: Used in computation maps and scheduling

Both formats encode the same information but serve different purposes in the computation pipeline.

## Chain Model Differences

The key format differs between continuous and discrete chain models:

| Aspect | Continuous Chain | Discrete Chain |
|--------|------------------|----------------|
| Block length constraint | Arbitrary (non-integer multiples of ds allowed) | Must be integer multiple of ds |
| Numbers in aggregated keys | `length_index` (0 for sliced propagators) | `n_segment` directly |
| Minimum sliced segments | 0 | 1 |

## ContourLengthMapping

The `ContourLengthMapping` class manages the relationship between contour lengths and their indices. This mapping is primarily used for **continuous chains**.

| Property | Description |
|----------|-------------|
| `length_index` | Index for unique contour lengths (1-based; 0 reserved for sliced propagators) |
| `ds_index` | Index for unique ds values (0-based) |
| `n_segment` | Number of segments = round(contour_length / ds) |

### Special Case: length_index = 0

For continuous chain aggregation, `length_index=0` is reserved for sliced propagators with `n_segment=0` (representing `contour_length=0`). This allows consistent use of `length_index` throughout the key format.

### Lookup Functions

```cpp
// Get length_index from contour_length (returns 0 for contour_length=0)
int length_index = mapping.get_length_index(contour_length);

// Get contour_length from length_index (returns 0.0 for index=0)
double contour_length = mapping.get_length_from_index(length_index);

// Get ds_index from contour_length
int ds_index = mapping.get_ds_index(contour_length);

// Get n_segment from contour_length (returns 0 for contour_length=0)
int n_segment = mapping.get_n_segment(contour_length);

// Convert (n_segment, ds_index) to length_index (for continuous chain aggregation)
int length_index = mapping.get_length_index_from_n_segment(n_segment, ds_index);
```

### Example Mapping

For a system with blocks of contour lengths 0.3, 0.35, and 0.5 with ds=0.1:

| contour_length | length_index | n_segment | local_ds | ds_index |
|----------------|--------------|-----------|----------|----------|
| 0.0 (sliced) | 0 | 0 | - | - |
| 0.3 | 1 | 3 | 0.1 | 1 |
| 0.35 | 2 | 4 | 0.0875 | 0 |
| 0.5 | 3 | 5 | 0.1 | 1 |

Note: `n_segment = round(contour_length / ds)` and `local_ds = contour_length / n_segment`.
The block with contour_length=0.35 has a different local_ds, resulting in a different ds_index.
A similar approach for local_ds is used in PSCF (https://github.com/dmorse/pscfpp).

## Components

| Symbol | Name | Description |
|--------|------|-------------|
| **D** | Dependency | Encodes the upstream propagator dependencies (nested structure) |
| **K** | Monomer type | The monomer type of the current block (e.g., "A", "B", "C") |
| **N** | Length index or n_segment | Integer suffix (interpretation depends on context and chain model) |
| **M** | ds_index | Integer index representing the contour step size (ds) |

## Code Format: DKN

The code format is used during propagator code generation and encodes the polymer topology.

### Structure
```
[Dependencies]MonomerType + N
```

Where N is:
- **Continuous chains**: `length_index`
- **Discrete chains**: `n_segment`

### Examples

| Code | Description |
|------|-------------|
| `A3` | A-type propagator with N=3 (free end) |
| `B2` | B-type propagator with N=2 (free end) |
| `(A3)B2` | B-type propagator depending on A3 |
| `(A3B2)C4` | C-type propagator at junction of A3 and B2 |
| `(A3B2:2)C4` | C-type propagator with two identical B2 branches |

### Dependency Notation

- **Parentheses `()`**: Junction point merging multiple branches
- **Colon `:`**: Repetition count (e.g., `B2:3` = three identical B2 branches)
- **Curly braces `{}`**: Custom initial condition (e.g., `{wall}A3`)

## Key Format: DK+M

The key format is derived from the code format and includes the ds_index for looking up computation parameters.

### Structure
```
[Dependencies]MonomerType+ds_index
```

### Conversion from Code to Key

The `get_key_from_code(code, mapping)` function converts DKN to DK+M:
1. Extract the DK part (everything except trailing digits)
2. Extract the N from trailing digits
3. Look up ds_index M from the ContourLengthMapping
4. Return DK+M format

### Examples

| Code (DKN) | Key (DK+M) | Notes |
|------------|------------|-------|
| `A3` | `A+0` | ds_index=0 for N=3 |
| `B2` | `B+0` | ds_index=0 for N=2 |
| `(A3)B2` | `(A3)B+0` | Only outer level has +M |
| `(A3B2)C4` | `(A3B2)C+0` | Inner deps keep N format |

**Important**: Only the outermost level has the `+M` suffix. Inner dependencies retain the DKN format.

## Aggregated Key Format

When identical propagator branches are detected, they are merged into aggregated computations to reduce redundancy.

### Structure
```
[dep_code1,dep_code2,...]MonomerType+ds_index
```

### Components

- **Square brackets `[]`**: Indicate an aggregated computation
- **dep_code**: Dependencies with trailing N (e.g., `(A)B0`)
- **Comma `,`**: Separates multiple dependencies
- **Colon `:`**: Repetition count after dep_code (e.g., `(A)B0:2`)

### Interpretation of N Inside Brackets

The trailing number N inside brackets differs by chain model:

| Chain Model | N represents | Example | Meaning |
|-------------|--------------|---------|---------|
| Continuous | `length_index` | `[(D)B0,(E)B0]B+0` | `length_index=0` (sliced to n_segment=0) |
| Discrete | `n_segment` | `[(D)B1,(E)B1]B+0` | `n_segment=1` (sliced to minimum) |

### Examples

**Continuous chain aggregation:**
```
Input propagators:
  (D)B+0: n_segment_right=4
  (E)B+0: n_segment_right=4

After slicing (n_segment=0, length_index=0):
  (D)B+0: n_segment_right=0
  (E)B+0: n_segment_right=0

Aggregated key: [(D)B0,(E)B0]B+0
                    ^     ^
                length_index=0 (for continuous chains)
```

**Discrete chain aggregation:**
```
Input propagators:
  (D)B+0: n_segment_right=4
  (E)B+0: n_segment_right=4

After slicing (minimum_n_segment=1):
  (D)B+0: n_segment_right=1
  (E)B+0: n_segment_right=1

Aggregated key: [(D)B1,(E)B1]B+0
                    ^     ^
                n_segment=1 (for discrete chains)
```

### Why Different Formats?

**Continuous chains** use `length_index`:
- Sliced propagators have `n_segment=0`, which corresponds to `length_index=0`
- This provides a consistent mapping through `ContourLengthMapping`

**Discrete chains** use `n_segment` directly:
- Block lengths must be integer multiples of ds, so `local_ds = global_ds` always
- No need for `ContourLengthMapping` during aggregation
- `n_segment` values are used directly without conversion

## Discrete Chain Validation

For discrete chains, all block contour lengths must be integer multiples of the global ds:

```
contour_length = n_segment × ds  (where n_segment is a positive integer)
```

If a non-integer multiple is provided, an error is thrown with the closest valid value:

```
For discrete chain model, block contour_length (0.375000) must be an integer
multiple of ds (0.083333). Closest valid value: 0.416667
```

## Initial Condition from Dependency (Continuous Chains)

For continuous Gaussian chains, the propagator $q(\mathbf{r}, s)$ satisfies the modified diffusion equation:

$$\frac{\partial q}{\partial s} = \frac{b^2}{6} \nabla^2 q - w(\mathbf{r}) q$$

where $b$ is the statistical segment length and $w(\mathbf{r})$ is the potential field.

The **dependency part D** of the propagator code determines the initial condition $q(\mathbf{r}, 0)$:

### Free End (Empty D)

For a propagator starting from a free chain end (e.g., `A+0`):

$$q(\mathbf{r}, 0) = 1$$

### Junction Point (D with Parentheses)

For a propagator at a junction point where multiple branches meet (e.g., `(A3B2)C+0`):

$$q(\mathbf{r}, 0) = \prod_{i} \left[ q_i(\mathbf{r}, N_i) \right]^{n_i}$$

where:
- $q_i(\mathbf{r}, N_i)$ is the propagator of dependency $i$ evaluated at its final segment $N_i$
- $n_i$ is the repetition count (from `:n` notation, default 1)

### Aggregated Computation (D with Square Brackets)

For aggregated propagators that combine multiple independent branches (e.g., `[(A)B0,(C)D0]E+0`):

$$q(\mathbf{r}, 0) = \sum_{i} n_i \cdot q_i(\mathbf{r}, N_i)$$

where the sum replaces the product to enable efficient parallel computation of similar branches.

### Custom Initial Condition (D with Curly Braces)

For propagators with user-specified initial conditions (e.g., `{wall}A+0`):

$$q(\mathbf{r}, 0) = q_{\text{init}}[\text{name}](\mathbf{r})$$

where `name` is the identifier inside the curly braces (e.g., "wall").

## Usage in Computation

### PropagatorComputationOptimizer

The optimizer uses keys (DK+M format) to:
1. Identify unique propagator computations
2. Track dependencies between propagators

### Scheduler

The scheduler uses keys to schedule parallel computations.
It uses the height (nesting depth) of keys to determine computation order:
- Height 0: Free-end propagators (no dependencies)
- Height 1+: Junction propagators (depend on lower-height propagators)

### Key Comparison

Keys are compared using `ComparePropagatorKey`:
1. First by height (lower height first)
2. Then lexicographically (for deterministic ordering)

## Summary Table

| Format | Example | Where Used |
|--------|---------|------------|
| Code (DKN) | `(A3B2)C4` | Code generation, inside non-aggregated deps |
| Key (DK+M) | `(A3B2)C+0` | Computation maps, scheduling |
| Aggregated (Continuous) | `[(A)B0,(C)D0]E+0` | Merged branches (length_index inside) |
| Aggregated (Discrete) | `[(A)B1,(C)D1]E+0` | Merged branches (n_segment inside) |

## References

- `PropagatorCode.h/cpp`: Code generation and parsing
- `PropagatorComputationOptimizer.h/cpp`: Optimization using keys
- `PropagatorAggregator.h/cpp`: Aggregated key generation
- `Scheduler.h/cpp`: Parallel computation scheduling
- `ContourLengthMapping.h/cpp`: Index mapping utilities
