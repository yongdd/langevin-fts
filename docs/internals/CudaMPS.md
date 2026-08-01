# CUDA Multi-Process Service (MPS) Guide

> **⚠️ Warning:** This document was generated with assistance from a large language model (LLM). While it is based on the referenced literature and the codebase, it may contain errors, misinterpretations, or inaccuracies. Please verify the equations and descriptions against the original references before relying on this document for research or implementation.

When running multiple simulations on the same GPU, NVIDIA's default time-slicing between CUDA contexts introduces significant overhead. CUDA MPS (Multi-Process Service) enables true concurrent kernel execution from different processes, eliminating context-switch overhead.

## When to Use MPS

| Scenario | Without MPS | With MPS |
|----------|-------------|----------|
| Small grid (16^3), 2 processes | 2x slower per process | ~5% slower per process |
| Medium grid (40^3), 2 processes | 2x slower per process | ~1.6x slower per process |
| Separate GPUs | No benefit | No benefit (not needed) |

**Use MPS when:**
- Running multiple simulations on the **same GPU**
- The grid is small enough that a single simulation doesn't fully saturate the GPU

**Not needed when:**
- Each simulation runs on a separate GPU
- Only one simulation runs at a time

## Usage from Command Line

### Start MPS

```bash
# Set the target GPU and pipe directory
export CUDA_VISIBLE_DEVICES=2              # GPU to share
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$USER
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$USER
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY

# Start the MPS daemon
nvidia-cuda-mps-control -d
```

### Run Programs

```bash
# Client processes must NOT set CUDA_VISIBLE_DEVICES
# Only set the pipe directory so they connect to the MPS daemon
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$USER
unset CUDA_VISIBLE_DEVICES

python simulation1.py &
python simulation2.py &
wait
```

### Stop MPS

```bash
echo quit | CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$USER nvidia-cuda-mps-control
```

## Usage from Python

MPS can be managed entirely from within a Python script using `subprocess` and `multiprocessing`:

```python
import os
import subprocess
import time
import multiprocessing as mp


GPU_ID = "2"
MPS_PIPE = f"/tmp/nvidia-mps-{os.environ['USER']}"
MPS_LOG = f"/tmp/nvidia-mps-log-{os.environ['USER']}"


def start_mps(gpu_id):
    """Start MPS daemon on the specified GPU."""
    os.makedirs(MPS_PIPE, exist_ok=True)
    os.makedirs(MPS_LOG, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    env["CUDA_MPS_PIPE_DIRECTORY"] = MPS_PIPE
    env["CUDA_MPS_LOG_DIRECTORY"] = MPS_LOG
    subprocess.run(["nvidia-cuda-mps-control", "-d"], env=env, check=True)
    time.sleep(0.5)


def stop_mps():
    """Stop the MPS daemon."""
    env = os.environ.copy()
    env["CUDA_MPS_PIPE_DIRECTORY"] = MPS_PIPE
    subprocess.run(["nvidia-cuda-mps-control"], input=b"quit", env=env)
    time.sleep(0.5)


def worker(seed, q_out):
    """Worker function running in a child process."""
    # MPS client setup: remove CUDA_VISIBLE_DEVICES, set pipe directory
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ["CUDA_MPS_PIPE_DIRECTORY"] = MPS_PIPE

    import numpy as np
    from polymerfts.propagator_solver import PropagatorSolver

    np.random.seed(seed)
    solver = PropagatorSolver(
        nx=[32]*3, lx=[4.0]*3, ds=1/40,
        bond_lengths={"A": 1.0, "B": 1.0},
        bc=["periodic"]*6,
        chain_model="continuous", numerical_method="rqm4", platform="cuda",
    )
    solver.add_polymer(1.0, [["A", 0.5, 0, 1], ["B", 0.5, 1, 2]])
    w = {"A": np.random.normal(0, 5.0, 32**3),
         "B": np.random.normal(0, 5.0, 32**3)}

    solver.compute_propagators(w)
    Q = solver.get_partition_function(0)
    q_out.put((seed, Q))


if __name__ == "__main__":
    # Must use "spawn" - CUDA does not support fork
    ctx = mp.get_context("spawn")

    start_mps(GPU_ID)
    try:
        q = ctx.Queue()
        p0 = ctx.Process(target=worker, args=(0, q))
        p1 = ctx.Process(target=worker, args=(1, q))
        p0.start(); p1.start()
        p0.join(); p1.join()

        for _ in range(2):
            seed, Q = q.get()
            print(f"Solver {seed}: Q = {Q:.10e}")
    finally:
        stop_mps()
```

## Important Notes

1. **Client must not set `CUDA_VISIBLE_DEVICES`**: The MPS daemon already owns the GPU. If a client process sets `CUDA_VISIBLE_DEVICES`, it will get a "no CUDA-capable device" error.

2. **Use `spawn`, not `fork`**: CUDA contexts are not fork-safe. Always use `mp.get_context("spawn")` when creating child processes.

3. **One daemon per GPU**: Each MPS daemon manages one GPU. To share multiple GPUs, start separate daemons with different pipe directories.

4. **Same-user only**: MPS shares the GPU among processes of the same user. Different users need separate MPS daemons.

5. **Scaling depends on GPU occupancy**: Small grids leave spare GPU capacity, allowing near-perfect scaling. Large grids that already saturate the GPU will see diminishing returns from MPS.

## Benchmark Results (NVIDIA A10, ds=1/40, continuous chain, RQM4)

### Same GPU, 2 processes

| Grid | Without MPS | With MPS |
|------|-------------|----------|
| 16^3 | 0.90x (slower than single) | **1.81x** speedup |
| 40^3 | 0.79x (slower than single) | **1.21x** speedup |

### Separate GPUs (MPS not needed)

| Grid | Speedup |
|------|---------|
| 40^3 | **1.86x** |
