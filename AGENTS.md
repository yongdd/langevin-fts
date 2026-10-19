# AGENTS.md

**All project guidance for coding agents lives in [CLAUDE.md](CLAUDE.md). Read it in
full before doing anything in this repository**: build commands, testing rules, code
architecture, numerical methods, and the mandatory workflow rules (never commit without
permission, never modify test parameters, use SLURM for long jobs).

This file only holds the notes that are specific to Codex.

## Codex-specific notes

### Sandbox / Escalation (Important!)

In this environment, GPU/NVML access from Codex often fails inside the sandbox
(e.g., `nvidia-smi` -> NVML Unknown Error, CUDA tests -> `cudaErrorOperatingSystem`).
When running any GPU-related command or CUDA test from Codex, request escalated
permissions and set the GPU inline:

```bash
# Example: run CUDA tests on an idle GPU with escalation
CUDA_VISIBLE_DEVICES=1 ctest -L basic -R TestCudaFFT --output-on-failure
```

Assume GPU commands require `require_escalated` unless proven otherwise.

### Additional workflow rules

- **Default to extra-high reasoning**: State assumptions explicitly, check edge cases and
  plausible failure modes, and validate conclusions against repository constraints before
  giving the final answer.
- **로그인 노드에서 빌드/설치/테스트 허용**: 로그인 노드에서 `make`, `make install`, `ctest`
  실행을 허용한다. 단 `nice -n 10 make -j4`까지만(로그인 노드는 공용, 8코어 cgroup 제한).
- **Review before committing**: run the tests that cover the change and have the diff
  reviewed (a second agent or a careful second pass) before asking for permission to
  commit. Commit only after the user says "commit"; never stage `devel/`.

## Notes for any agent on this cluster

- **`devel/` is the maintainer's private folder**: do not audit, clean up, reformat or
  commit it, and never link to it from docs. Its files often show up as modified in
  `git status`; leave them alone.
- **Build environment**: `conda activate polymerfts` plus the CUDA toolkit on PATH
  (`/opt/ohpc/pub/cuda/11.8.0/bin`, or `module load cuda/11.8`). The CMake build links
  libraries by absolute path and passes the CUDA include/lib directories explicitly, so
  it does not depend on `CPLUS_INCLUDE_PATH`/`LIBRARY_PATH` of the shell. Configure with
  `-DCMAKE_PREFIX_PATH=$CONDA_PREFIX` so MKL/FFTW/spglib/LAPACKE come from that
  environment and not from another conda env that happens to be on the path.
- **Warnings**: CI builds with `-DPOLYMERFTS_WERROR=ON` (GCC 13). Check new host C++
  with `g++ -O3 -DNDEBUG -Wall -Werror -c` (a `-fsyntax-only` pass misses
  `-Wmaybe-uninitialized`). nvcc-compiled `.cu` files are not covered by the option.
- **GPU quota**: the QOS allows 8 GPUs per user and the maintainer's own training jobs
  often hold them all, so a `--gres=gpu:1` test job can pend for hours
  (`QOSMaxGRESPerUser`). Fallback accepted by the maintainer: CPU-only SLURM jobs
  (no `--gres`). On a CPU node every test that loops over `avail_platforms()` still sees
  `cuda` and ends with `cudaErrorNoDevice`; those failures (about 42 of 71 in the C++
  suite) are expected there, not regressions.
- **Python tests run from the build tree**: `make` stages the package in
  `build/python/polymerfts`; `ctest` sets `PYTHONPATH` to it, so `make install` is not
  needed for testing. For a quick check after editing `src/python/*.py`, run `make`
  (seconds) rather than reinstalling.
- **Constructing `SCFT`/`LFTS` writes `polymer_<id>.png`** into the current directory
  (chain-architecture drawings). Run throwaway checks from a scratch directory, not from
  the repository root.
- **Several versions side by side**: `scripts/install_versioned.sh` builds a wheel into
  `~/wheels` and installs it into `~/venvs/polymerfts-<version>` (see
  docs/getting-started/Installation.md).
