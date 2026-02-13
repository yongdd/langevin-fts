1. Parallel Tempering
6. Packaging with Conda, including scft.py and lfts.py
8. GPU bandwidth test
9. UnitTest: Anderson Mixing
10. Validation Check for Pseudo Parameters
16. Process Bar
20. Graphic User Interface
27. Syntax check of Keys in Molecules 
31. Throws Exception in openMP block
35. Write UnitTest: TestComparePropagatorKey
37. Reduce the number of split in scheduler
40. Figure out why two GPUs performance keeps changing in A100
42. Check if two GPUs can be utilized for CudaAndersonMixing
43. Operator Overloading for class Array
44. Check if CUDA cub works in CUDA 12.0 and higher
46. Replace "new" keyword to a safe memory allocation
47. Add tests for example files
48. Check if CUDA can be installed using Conda
50. Check if cuFFT callback function is applicable
51. Figure out why 'std::map<std::string, std::map<int, double *>>' causes errors in 'omp parallel for'
52. Figure out why 'propagator_half_steps[key].find(0) == propagator_half_steps[key].end()' fixes above problem
53. And a option to print a data file in a human-readable text.
54. Extend scft.py to cover singular chi matrix
55. Add tests for stress computation with complex type
56. Check get_negative_frequency_mapping() and compute_single_segment_stress()
57. Add tests for Anderson mixing with complex type
58. Figure out why 'os.environ' does not work under Python and openmp installation with conda-forge
59. Parameter sweep / continuation (use previous solution as initial guess for phase diagram construction)
60. Grand canonical ensemble (GCE) - core functions implemented (get_stress_gce, get_total_concentration_gce), needs SCFT/LFTS high-level API
61. Documentation: Python API reference (Sphinx + autodoc for PropagatorSolver, SCFT, LFTS classes)
62. Documentation: File formats (fields_*.mat, structure_function_*.mat, YAML/JSON config)
63. Documentation: Examples guide (detailed README for examples/scft/, examples/lfts/)
64. SCFT preconditioner for Anderson Mixing (left-preconditioning failed; try right-preconditioning or modified AM inner product)