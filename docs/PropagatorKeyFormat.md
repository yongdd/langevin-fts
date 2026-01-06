# Propagator Code and Key Format Specification

This document describes the string representations used to identify and manage propagator computations in the polymer field theory simulation library.

## Overview

Propagators are identified using two related string formats:
- **Code format (DKN)**: Used internally during code generation
- **Key format (DK+M)**: Used in computation maps and scheduling

Both formats encode the same information but serve different purposes in the computation pipeline.

## ContourLengthMapping

The `ContourLengthMapping` class manages the relationship between contour lengths and their indices. This mapping is fundamental to understanding the code and key formats.

| Property | Description |
|----------|-------------|
| `length_index` | Index for unique contour lengths (starting from 1) |
| `ds_index` | Index for unique ds values (starting from 1) |
| `n_segment` | Number of segments = round(contour_length / ds) |

### Lookup Functions

```cpp
// Get length_index from contour_length
int length_index = mapping.get_length_index(contour_length);

// Get contour_length from length_index
double contour_length = mapping.get_length_from_index(length_index);

// Get ds_index from contour_length
int ds_index = mapping.get_ds_index(contour_length);

// Get n_segment from contour_length
int n_segment = mapping.get_n_segment(contour_length);
```

### Example Mapping

For a system with blocks of contour lengths 0.4, 0.9, and 1.2 with ds=0.1:

| contour_length | length_index | n_segment | ds_index |
|----------------|--------------|-----------|----------|
| 0.4 | 1 | 4 | 1 |
| 0.9 | 2 | 9 | 1 |
| 1.2 | 3 | 12 | 1 |

## Components

| Symbol | Name | Description |
|--------|------|-------------|
| **D** | Dependency | Encodes the upstream propagator dependencies (nested structure) |
| **K** | Monomer type | The monomer type of the current block (e.g., "A", "B", "C") |
| **N** | Length index | Integer index representing the block's contour length |
| **M** | ds_index | Integer index representing the contour step size (ds) |

## Code Format: DKN

The code format is used during propagator code generation and encodes the polymer topology.

### Structure
```
[Dependencies]MonomerType + LengthIndex
```

### Examples

| Code | Description |
|------|-------------|
| `A3` | A-type propagator with length_index=3 (free end) |
| `B2` | B-type propagator with length_index=2 (free end) |
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
[Dependencies]MonomerType + ds_index
```

### Conversion from Code to Key

The `get_key_from_code(code, mapping)` function converts DKN to DK+M:
1. Extract the DK part (everything except trailing digits)
2. Extract the length_index N from trailing digits
3. Look up ds_index M from the ContourLengthMapping
4. Return DK+M format

### Examples

| Code (DKN) | Key (DK+M) | Notes |
|------------|------------|-------|
| `A3` | `A+1` | ds_index=1 for length_index=3 |
| `B2` | `B+1` | ds_index=1 for length_index=2 |
| `(A3)B2` | `(A3)B+1` | Only outer level has +M |
| `(A3B2)C4` | `(A3B2)C+1` | Inner deps keep N format |

**Important**: Only the outermost level has the `+M` suffix. Inner dependencies retain the DKN format.

## Aggregated Key Format

When identical propagator branches are detected, they are merged into aggregated computations to reduce redundancy.

### Structure
```
[dep_code1,dep_code2,...]MonomerType+ds_index
```

### Components

- **Square brackets `[]`**: Indicate an aggregated computation
- **dep_code**: Dependencies in DKN format (e.g., `(A)B3`)
- **Comma `,`**: Separates multiple dependencies
- **Colon `:`**: Repetition count after dep_code (e.g., `(A)B3:2`)

### Examples

| Aggregated Key | Description |
|----------------|-------------|
| `[(A)B3,(C)D2]E+1` | E-type propagator aggregating (A)B with 3 segments and (C)D with 2 segments |
| `[(A)B3:2,(C)D2]E+1` | Same, but (A)B appears twice |
| `[A0,B0]C+1` | C-type aggregating two free-end propagators |

### Parsing Aggregated Keys

When parsing dependencies from an aggregated key:
1. Content inside `[...]` contains dep_codes in DKN format
2. Each dep_code's trailing digits represent n_segment (not length_index)
3. All deps share the same ds_index as the outer key

For example, parsing `[(A)B3,(C)D2]E+1`:
- Dependency 1: key=`(A)B+1`, n_segment=3
- Dependency 2: key=`(C)D+1`, n_segment=2
- The ds_index=1 is inherited from the outer `+1`

## Usage in Computation

### PropagatorComputationOptimizer

The optimizer uses keys (DK+M format) to:
1. Identify unique propagator computations
2. Track dependencies between propagators
3. Schedule parallel computations

### Scheduler

The scheduler uses the height (nesting depth) of keys to determine computation order:
- Height 0: Free-end propagators (no dependencies)
- Height 1+: Junction propagators (depend on lower-height propagators)

### Key Comparison

Keys are compared using `ComparePropagatorKey`:
1. First by height (lower height first)
2. Then lexicographically (for deterministic ordering)

## Summary Table

| Format | Example | Where Used |
|--------|---------|------------|
| Code (DKN) | `(A3B2)C4` | Code generation, inside aggregated keys |
| Key (DK+M) | `(A3B2)C+1` | Computation maps, scheduling |
| Aggregated | `[(A)B3,(C)D2]E+1` | Merged branch computations |

## References

- `PropagatorCode.h/cpp`: Code generation and parsing
- `PropagatorComputationOptimizer.h/cpp`: Optimization using keys
- `PropagatorAggregator.h/cpp`: Aggregated key generation
- `ContourLengthMapping.h/cpp`: Index mapping utilities
