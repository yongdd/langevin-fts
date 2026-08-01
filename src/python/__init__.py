# Package version. The pip/conda metadata is authoritative when the package
# was installed through pip (pyproject.toml); the fallback constant covers
# the plain 'make install' path, which ships no dist metadata. Keep the
# fallback in sync with CMakeLists.txt / pyproject.toml.
try:
    from importlib.metadata import version as _dist_version
    __version__ = _dist_version("polymerfts")
except Exception:
    __version__ = "1.0.0"

# Core C++ bindings (re-export all for user convenience)
from ._core import *

# High-level Python classes
from .polymer_field_theory import SymmetricPolymerTheory
from .propagator_solver import PropagatorSolver
from .smearing import Smearing
from .scft import SCFT
from .lfts import LFTS
from .clfts import CLFTS
from .wtmd import WTMD
from .validation import ValidationError
from .result import SCFTResult, LFTSResult, IterationInfo
from .config import load_config, save_config, create_template_config, ConfigError

# Submodules
from . import compressor
from . import utils
from . import io

# Utility exports
from .utils import configure_logging, deprecated, warn_deprecated_param