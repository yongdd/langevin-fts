# Crystal Systems for Box Optimization

This directory contains examples demonstrating SCFT simulations with various crystal systems for stress-driven box optimization. The crystal system determines which lattice parameters (lengths and angles) are optimized and what symmetry constraints are enforced.

## Overview

When `box_is_altering: True` is set, SCFT uses stress tensor components to optimize the simulation box. The `crystal_system` parameter controls:
1. Which box dimensions can vary independently
2. Which angles can vary
3. What symmetry constraints are enforced (e.g., a = b for hexagonal)

## Available Crystal Systems

### 2D Crystal Systems

| System | Lengths | Angles | Constraints |
|--------|---------|--------|-------------|
| **Hexagonal2D** | a = b | γ = 120° (fixed) | Single lattice parameter |
| **Oblique2D** | a, b independent | γ optimized | Most general 2D system |

### 3D Crystal Systems

| System | Lengths | Angles | Constraints |
|--------|---------|--------|-------------|
| **Cubic** | a = b = c | α = β = γ = 90° | Single lattice parameter |
| **Tetragonal** | a = b ≠ c | α = β = γ = 90° | Two lattice parameters |
| **Orthorhombic** | a ≠ b ≠ c | α = β = γ = 90° | Three independent lengths |
| **Hexagonal** | a = b ≠ c | α = β = 90°, γ = 120° | Two lattice parameters |
| **Monoclinic** | a, b, c independent | α = γ = 90°, β optimized | One angle varies |
| **Triclinic** | a, b, c independent | α, β, γ all optimized | Most general 3D system |

## Parameter Reference

### Lattice Angles Convention

The three angles are defined as:
- **α (alpha)**: Angle between b and c axes
- **β (beta)**: Angle between a and c axes
- **γ (gamma)**: Angle between a and b axes

For 2D systems, only γ is relevant.

### Usage

```python
# 3D example with full angles
params_3d = {
    "nx": [32, 32, 32],
    "lx": [2.0, 2.0, 2.0],
    "angles": [90.0, 90.0, 120.0],  # [α, β, γ] in degrees

    "crystal_system": "Hexagonal",  # Choose crystal system
    "box_is_altering": True,        # Enable box optimization
    "scale_stress": 0.3,            # Stress scaling factor
    # ... other parameters
}

# 2D example with single gamma
params_2d = {
    "nx": [32, 32],
    "lx": [2.0, 2.0],
    "angles": [120.0],  # Single gamma for 2D (α=β=90° implicit)

    "crystal_system": "Hexagonal2D",
    "box_is_altering": True,
    # ... other parameters
}
```

## Examples

### HexagonalCylinder2D.py
2D hexagonal cylinder phase with fixed γ = 120° and a = b constraint.
- Best for: Well-ordered hexagonal cylinder phases
- Angle: Fixed at 120°
- Lengths: Single parameter (a = b)

### ObliqueCylinder2D.py
2D oblique system with angle optimization from 115° → 120°.
- Best for: Finding optimal angle for 2D phases
- Angle: γ is optimized (starts at 115°, converges to 120°)
- Lengths: a and b vary independently

### TriclinicCylinder3D.py
3D triclinic system with full angle optimization.
- Best for: Finding optimal angles in 3D without constraints
- Angles: All three (α, β, γ) can be optimized
- Lengths: All three (a, b, c) vary independently
- Result: Finds hexagonal packing with γ → 120°, α = β = 90°

### MonoclinicLamella3D.py
3D monoclinic system with β angle optimization.
- Best for: Tilted lamellar phases
- Angles: Only β varies, α = γ = 90° fixed
- Lengths: All three vary independently

### AngleOptimizationDemo.py
Demonstrates angle optimization behavior with various starting conditions.

### ConstrainedAngleDemo.py
Shows how different crystal systems constrain the optimization.

## Tips for Angle Optimization

1. **Starting Angle**: For hexagonal packing (γ = 120°), start from γ > 105° to avoid local minima near rectangular packing (γ = 90°). Starting from 115° works well.

2. **Perturbation**: Add small random perturbation to initial fields to break symmetry and enable stress-driven optimization:
   ```python
   np.random.seed(42)
   perturbation = np.random.randn(*w_A.shape) * 0.5
   perturbation = gaussian_filter(perturbation, sigma=2.0, mode='wrap')
   w_A = w_A + perturbation
   ```

3. **Scale Factor**: The `scale_stress` parameter controls how aggressively the box is updated. Typical values: 0.2-0.5.

4. **Convergence**: Angle optimization may require more iterations than fixed-angle simulations.

## Crystal System Selection Guide

| Morphology | Recommended System |
|------------|-------------------|
| 2D Hex cylinders (known) | Hexagonal2D |
| 2D Hex cylinders (finding) | Oblique2D |
| 3D Lamellae | Orthorhombic or Tetragonal |
| 3D Hex cylinders | Hexagonal |
| 3D Gyroid/BCC | Cubic |
| Tilted structures | Monoclinic or Triclinic |
| Unknown structure | Triclinic (most general) |

## Stress Array Components

The stress is stored as a 6-component array (Voigt notation):

| Index | Component | Drives Optimization of |
|-------|-----------|------------------------|
| 0 | $\sigma_a$ | $L_a$ |
| 1 | $\sigma_b$ | $L_b$ |
| 2 | $\sigma_c$ | $L_c$ |
| 3 | $\sigma_{ab}$ | $\gamma$ (angle between **a** and **b**) |
| 4 | $\sigma_{ac}$ | $\beta$ (angle between **a** and **c**) |
| 5 | $\sigma_{bc}$ | $\alpha$ (angle between **b** and **c**) |

For 2D systems, only indices 0, 1, and 2 (for $\sigma_{12}$) are used.

At equilibrium, all stress components vanish. The box optimization uses gradient descent with the `scale_stress` parameter controlling the step size.
