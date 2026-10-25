# Package version. The pip/conda metadata is authoritative when the package
# was installed through pip (pyproject.toml); the fallback constant covers
# the plain 'make install' path, which ships no dist metadata. Keep the
# fallback in sync with CMakeLists.txt / pyproject.toml.
try:
    from importlib.metadata import version as _dist_version
    __version__ = _dist_version("polymerfts")
except Exception:
    __version__ = "0.10.0"

# Windows does not search PATH or an RPATH for the libraries of an extension
# module: the spglib library that _core links lives in the spglib package.
import sys as _sys
if _sys.platform == "win32":
    import os as _os
    import importlib.util as _importlib_util
    _spglib_spec = _importlib_util.find_spec("spglib")
    if _spglib_spec is not None and _spglib_spec.origin:
        _os.add_dll_directory(_os.path.dirname(_spglib_spec.origin))

# The console output names quantities with Greek letters and subscripts (χN, κ,
# βF/(ρ₀V/N)). A stream redirected to a file or pipe uses the locale encoding,
# e.g. cp949 or cp1252 on Windows, which cannot encode all of them, and print()
# raised UnicodeEncodeError. Such a stream writes the characters it cannot
# encode as backslash escapes; everything it can encode is written unchanged.
for _stream in (_sys.stdout, _sys.stderr):
    try:
        "βκμξρχΔΨ₀".encode(_stream.encoding, _stream.errors)
    except UnicodeEncodeError:
        try:
            _stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass   # not a reconfigurable text stream
    except (AttributeError, LookupError, TypeError):
        pass       # no stream, or one without a text encoding

# Core C++ bindings (re-export all for user convenience)
from ._core import *

# High-level Python classes
from .polymer_field_theory import SymmetricPolymerTheory
from .propagator_solver import PropagatorSolver
from .smearing import Smearing
from .integrators import LeimkuhlerMatthews, EulerMaruyama
from .scft import SCFT
from .lfts import LFTS
from .clfts import CLFTS
from .wtmd import WTMD
from .validation import ValidationError, SaddlePointError
from .scft import SCFTResult, IterationInfo
from .phase_equilibrium import coexistence_quantities, homogeneous_thermodynamics, find_binary_coexistence
from .lfts import LR, LRAM
from .utils import load_config, ConfigError, save_vtk, draw_polymer_architectures

# Submodules
from . import utils
