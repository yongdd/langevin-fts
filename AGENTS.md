# AGENTS.md

**All project guidance for coding agents lives in [CLAUDE.md](CLAUDE.md). Read it in
full before doing anything in this repository**: build commands, testing rules, code
architecture, numerical methods, and the mandatory workflow rules (never commit without
permission, never modify test parameters, leave `devel/` alone, use SLURM for long jobs).

This file only holds what CLAUDE.md does not: Codex-specific notes and notes about this
cluster.

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
- **Review before asking to commit**: run the tests that cover the change and have the
  diff reviewed (a second agent or a careful second pass) first.

## Notes for any agent on this cluster

- **CUDA toolkit**: `conda activate polymerfts` does not provide `nvcc`; put the toolkit
  on PATH (`/opt/ohpc/pub/cuda/11.8.0/bin`, or `module load cuda/11.8`) before configuring.
- **Warning check for new host C++** (CI builds with `POLYMERFTS_WERROR`, see CLAUDE.md):
  `g++ -O3 -DNDEBUG -Wall -Werror -c file.cpp` (a `-fsyntax-only` pass misses
  `-Wmaybe-uninitialized`). nvcc-compiled `.cu` files are not covered by the option.
- **GPU quota**: the QOS allows 8 GPUs per user and the maintainer's own training jobs
  often hold them all, so a `--gres=gpu:1` test job can pend for hours
  (`QOSMaxGRESPerUser`). Fallback accepted by the maintainer: CPU-only SLURM jobs
  (no `--gres`). Without a visible GPU, `avail_platforms()` omits `cuda`, so the Python
  tests pass there; the C++ tests that construct CUDA objects unconditionally
  (TestPseudo*, TestCrankNicolson*, TestCrysFFT*, about 22 of 74) fail with
  `cudaErrorNoDevice`, which is expected on such a node, not a regression.
- **Constructing `SCFT`/`LFTS`/`CLFTS` writes `polymer_<id>.png`** into the current
  directory (chain-architecture drawings). Run throwaway checks from a scratch directory,
  not from the repository root.
- **Several versions side by side**: `scripts/install_versioned.sh` builds a wheel into
  `~/wheels` and installs it into `~/venvs/polymerfts-<version>` (see
  docs/getting-started/Installation.md).
