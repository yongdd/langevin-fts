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
  실행을 허용한다.
