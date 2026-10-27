"""The CUDA part of a Python test in a CUDA build without a visible CUDA device.

PlatformSelector.avail_platforms() leaves out "cuda" when no device is visible, so a
test that runs every available platform would drop its CUDA part and pass, which
reads as a validated CUDA path. As the C++ tests do, such a test runs its CPU parts
and then fails. A build without the CUDA backend has no CUDA part to run.
"""

from polymerfts import _core


def cuda_part_missing():
    """True, after printing why, if this build has the CUDA backend but no CUDA device is visible."""
    if _core.build_info["cuda"] == "TRUE" and "cuda" not in _core.PlatformSelector.avail_platforms():
        print("FAILED: this build has the CUDA backend, but no CUDA device is visible: "
              "the CUDA part of this test did not run.")
        return True
    return False
