Features
========

This repository contains a library for polymer field theory simulations and its applications, such as SCFT and L-FTS.
The most time-consuming and common tasks in polymer field theory simulations are the computation of chain propagators, stresses, partition functions, and polymer concentrations in external fields. 
These routines are implemented in C++/CUDA and provided as Python classes, enabling you to write programs using Python with numerous useful libraries. 
This library automatically avoids redundant computations in the chain propagator calculations for branched polymers.
It supports the following features:

* Any number of monomer types
* Arbitrary acyclic branched polymers
* Arbitrary mixtures of block copolymers and homopolymers
* Arbitrary initial conditions of propagators at chain ends
* Access to chain propagators
* Conformational asymmetry
* Simulation box dimension: 3D, 2D and 1D
* Automatic optimization of chain propagator computations
* Chain models: continuous, discrete
* Pseudo-spectral method

   * 4th-order Richardson extrapolation method for continuous chain
   * Support continuous and discrete chains
   * Periodic boundaries only

* Real-space method (**beta**)

   * 2th-order Crank-Nicolson method
   * Support only continuous chain
   * Support periodic, reflecting, absorbing boundaries

* Can set impenetrable region using a mask (**beta**)
* Anderson mixing
* Platforms: MKL (CPU) and CUDA (GPU)
* Parallel computations of propagators with multi-core CPUs (up to 8), or multi CUDA streams (up to 4) to maximize GPU usage
* Memory saving option
* Common interfaces regardless of chain model, simulation box dimension, and platform

Dependencies
============

Installation
============

Usages
======

Contribution
============

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide:

.. toctree::
   :maxdepth: 2
   :caption: Polymer Field Theory:
