# This file was automatically generated by SWIG (http://www.swig.org).
# Version 4.0.1
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
if _swig_python_version_info < (2, 7, 0):
    raise RuntimeError("Python 2.7 or later required")

# Import the low-level C/C++ module
if __package__ or "." in __name__:
    from . import _langevin_fts
else:
    import _langevin_fts

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)


def _swig_setattr_nondynamic_instance_variable(set):
    def set_instance_attr(self, name, value):
        if name == "thisown":
            self.this.own(value)
        elif name == "this":
            set(self, name, value)
        elif hasattr(self, name) and isinstance(getattr(type(self), name), property):
            set(self, name, value)
        else:
            raise AttributeError("You cannot add instance attributes to %s" % self)
    return set_instance_attr


def _swig_setattr_nondynamic_class_variable(set):
    def set_class_attr(cls, name, value):
        if hasattr(cls, name) and not isinstance(getattr(cls, name), property):
            set(cls, name, value)
        else:
            raise AttributeError("You cannot add class attributes to %s" % cls)
    return set_class_attr


def _swig_add_metaclass(metaclass):
    """Class decorator for adding a metaclass to a SWIG wrapped class - a slimmed down version of six.add_metaclass"""
    def wrapper(cls):
        return metaclass(cls.__name__, cls.__bases__, cls.__dict__.copy())
    return wrapper


class _SwigNonDynamicMeta(type):
    """Meta class to enforce nondynamic attributes (no new attributes) for a class"""
    __setattr__ = _swig_setattr_nondynamic_class_variable(type.__setattr__)


class SwigPyIterator(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    __swig_destroy__ = _langevin_fts.delete_SwigPyIterator

    def value(self):
        return _langevin_fts.SwigPyIterator_value(self)

    def incr(self, n=1):
        return _langevin_fts.SwigPyIterator_incr(self, n)

    def decr(self, n=1):
        return _langevin_fts.SwigPyIterator_decr(self, n)

    def distance(self, x):
        return _langevin_fts.SwigPyIterator_distance(self, x)

    def equal(self, x):
        return _langevin_fts.SwigPyIterator_equal(self, x)

    def copy(self):
        return _langevin_fts.SwigPyIterator_copy(self)

    def next(self):
        return _langevin_fts.SwigPyIterator_next(self)

    def __next__(self):
        return _langevin_fts.SwigPyIterator___next__(self)

    def previous(self):
        return _langevin_fts.SwigPyIterator_previous(self)

    def advance(self, n):
        return _langevin_fts.SwigPyIterator_advance(self, n)

    def __eq__(self, x):
        return _langevin_fts.SwigPyIterator___eq__(self, x)

    def __ne__(self, x):
        return _langevin_fts.SwigPyIterator___ne__(self, x)

    def __iadd__(self, n):
        return _langevin_fts.SwigPyIterator___iadd__(self, n)

    def __isub__(self, n):
        return _langevin_fts.SwigPyIterator___isub__(self, n)

    def __add__(self, n):
        return _langevin_fts.SwigPyIterator___add__(self, n)

    def __sub__(self, *args):
        return _langevin_fts.SwigPyIterator___sub__(self, *args)
    def __iter__(self):
        return self

# Register SwigPyIterator in _langevin_fts:
_langevin_fts.SwigPyIterator_swigregister(SwigPyIterator)

class IntArray3(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _langevin_fts.IntArray3_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _langevin_fts.IntArray3___nonzero__(self)

    def __bool__(self):
        return _langevin_fts.IntArray3___bool__(self)

    def __len__(self):
        return _langevin_fts.IntArray3___len__(self)

    def __getslice__(self, i, j):
        return _langevin_fts.IntArray3___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _langevin_fts.IntArray3___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _langevin_fts.IntArray3___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _langevin_fts.IntArray3___delitem__(self, *args)

    def __getitem__(self, *args):
        return _langevin_fts.IntArray3___getitem__(self, *args)

    def __setitem__(self, *args):
        return _langevin_fts.IntArray3___setitem__(self, *args)

    def __init__(self, *args):
        _langevin_fts.IntArray3_swiginit(self, _langevin_fts.new_IntArray3(*args))

    def empty(self):
        return _langevin_fts.IntArray3_empty(self)

    def size(self):
        return _langevin_fts.IntArray3_size(self)

    def swap(self, v):
        return _langevin_fts.IntArray3_swap(self, v)

    def begin(self):
        return _langevin_fts.IntArray3_begin(self)

    def end(self):
        return _langevin_fts.IntArray3_end(self)

    def rbegin(self):
        return _langevin_fts.IntArray3_rbegin(self)

    def rend(self):
        return _langevin_fts.IntArray3_rend(self)

    def front(self):
        return _langevin_fts.IntArray3_front(self)

    def back(self):
        return _langevin_fts.IntArray3_back(self)

    def fill(self, u):
        return _langevin_fts.IntArray3_fill(self, u)
    __swig_destroy__ = _langevin_fts.delete_IntArray3

# Register IntArray3 in _langevin_fts:
_langevin_fts.IntArray3_swigregister(IntArray3)

class DoubleArray3(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _langevin_fts.DoubleArray3_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _langevin_fts.DoubleArray3___nonzero__(self)

    def __bool__(self):
        return _langevin_fts.DoubleArray3___bool__(self)

    def __len__(self):
        return _langevin_fts.DoubleArray3___len__(self)

    def __getslice__(self, i, j):
        return _langevin_fts.DoubleArray3___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _langevin_fts.DoubleArray3___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _langevin_fts.DoubleArray3___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _langevin_fts.DoubleArray3___delitem__(self, *args)

    def __getitem__(self, *args):
        return _langevin_fts.DoubleArray3___getitem__(self, *args)

    def __setitem__(self, *args):
        return _langevin_fts.DoubleArray3___setitem__(self, *args)

    def __init__(self, *args):
        _langevin_fts.DoubleArray3_swiginit(self, _langevin_fts.new_DoubleArray3(*args))

    def empty(self):
        return _langevin_fts.DoubleArray3_empty(self)

    def size(self):
        return _langevin_fts.DoubleArray3_size(self)

    def swap(self, v):
        return _langevin_fts.DoubleArray3_swap(self, v)

    def begin(self):
        return _langevin_fts.DoubleArray3_begin(self)

    def end(self):
        return _langevin_fts.DoubleArray3_end(self)

    def rbegin(self):
        return _langevin_fts.DoubleArray3_rbegin(self)

    def rend(self):
        return _langevin_fts.DoubleArray3_rend(self)

    def front(self):
        return _langevin_fts.DoubleArray3_front(self)

    def back(self):
        return _langevin_fts.DoubleArray3_back(self)

    def fill(self, u):
        return _langevin_fts.DoubleArray3_fill(self, u)
    __swig_destroy__ = _langevin_fts.delete_DoubleArray3

# Register DoubleArray3 in _langevin_fts:
_langevin_fts.DoubleArray3_swigregister(DoubleArray3)

class StringVector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _langevin_fts.StringVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _langevin_fts.StringVector___nonzero__(self)

    def __bool__(self):
        return _langevin_fts.StringVector___bool__(self)

    def __len__(self):
        return _langevin_fts.StringVector___len__(self)

    def __getslice__(self, i, j):
        return _langevin_fts.StringVector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _langevin_fts.StringVector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _langevin_fts.StringVector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _langevin_fts.StringVector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _langevin_fts.StringVector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _langevin_fts.StringVector___setitem__(self, *args)

    def pop(self):
        return _langevin_fts.StringVector_pop(self)

    def append(self, x):
        return _langevin_fts.StringVector_append(self, x)

    def empty(self):
        return _langevin_fts.StringVector_empty(self)

    def size(self):
        return _langevin_fts.StringVector_size(self)

    def swap(self, v):
        return _langevin_fts.StringVector_swap(self, v)

    def begin(self):
        return _langevin_fts.StringVector_begin(self)

    def end(self):
        return _langevin_fts.StringVector_end(self)

    def rbegin(self):
        return _langevin_fts.StringVector_rbegin(self)

    def rend(self):
        return _langevin_fts.StringVector_rend(self)

    def clear(self):
        return _langevin_fts.StringVector_clear(self)

    def get_allocator(self):
        return _langevin_fts.StringVector_get_allocator(self)

    def pop_back(self):
        return _langevin_fts.StringVector_pop_back(self)

    def erase(self, *args):
        return _langevin_fts.StringVector_erase(self, *args)

    def __init__(self, *args):
        _langevin_fts.StringVector_swiginit(self, _langevin_fts.new_StringVector(*args))

    def push_back(self, x):
        return _langevin_fts.StringVector_push_back(self, x)

    def front(self):
        return _langevin_fts.StringVector_front(self)

    def back(self):
        return _langevin_fts.StringVector_back(self)

    def assign(self, n, x):
        return _langevin_fts.StringVector_assign(self, n, x)

    def resize(self, *args):
        return _langevin_fts.StringVector_resize(self, *args)

    def insert(self, *args):
        return _langevin_fts.StringVector_insert(self, *args)

    def reserve(self, n):
        return _langevin_fts.StringVector_reserve(self, n)

    def capacity(self):
        return _langevin_fts.StringVector_capacity(self)
    __swig_destroy__ = _langevin_fts.delete_StringVector

# Register StringVector in _langevin_fts:
_langevin_fts.StringVector_swigregister(StringVector)

class ParamParser(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined")
    __repr__ = _swig_repr

    @staticmethod
    def get_instance():
        return _langevin_fts.ParamParser_get_instance()

    def read_param_file(self, param_file_name, verbose=True):
        return _langevin_fts.ParamParser_read_param_file(self, param_file_name, verbose)

    def display_usage_info(self):
        return _langevin_fts.ParamParser_display_usage_info(self)

    def get(self, *args):
        return _langevin_fts.ParamParser_get(self, *args)

# Register ParamParser in _langevin_fts:
_langevin_fts.ParamParser_swigregister(ParamParser)

def ParamParser_get_instance():
    return _langevin_fts.ParamParser_get_instance()

class PolymerChain(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    NN = property(_langevin_fts.PolymerChain_NN_get, _langevin_fts.PolymerChain_NN_set)
    NNf = property(_langevin_fts.PolymerChain_NNf_get, _langevin_fts.PolymerChain_NNf_set)
    f = property(_langevin_fts.PolymerChain_f_get, _langevin_fts.PolymerChain_f_set)
    ds = property(_langevin_fts.PolymerChain_ds_get, _langevin_fts.PolymerChain_ds_set)
    chi_n = property(_langevin_fts.PolymerChain_chi_n_get, _langevin_fts.PolymerChain_chi_n_set)
    type = property(_langevin_fts.PolymerChain_type_get)

    def __init__(self, f, NN, chi_n):
        _langevin_fts.PolymerChain_swiginit(self, _langevin_fts.new_PolymerChain(f, NN, chi_n))
    __swig_destroy__ = _langevin_fts.delete_PolymerChain

    def set_chin(self, chi_n):
        return _langevin_fts.PolymerChain_set_chin(self, chi_n)

# Register PolymerChain in _langevin_fts:
_langevin_fts.PolymerChain_swigregister(PolymerChain)

class SimulationBox(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    nx = property(_langevin_fts.SimulationBox_nx_get, _langevin_fts.SimulationBox_nx_set)
    lx = property(_langevin_fts.SimulationBox_lx_get, _langevin_fts.SimulationBox_lx_set)
    dx = property(_langevin_fts.SimulationBox_dx_get, _langevin_fts.SimulationBox_dx_set)
    MM = property(_langevin_fts.SimulationBox_MM_get, _langevin_fts.SimulationBox_MM_set)
    dv = property(_langevin_fts.SimulationBox_dv_get, _langevin_fts.SimulationBox_dv_set)
    volume = property(_langevin_fts.SimulationBox_volume_get, _langevin_fts.SimulationBox_volume_set)

    def __init__(self, *args):
        _langevin_fts.SimulationBox_swiginit(self, _langevin_fts.new_SimulationBox(*args))
    __swig_destroy__ = _langevin_fts.delete_SimulationBox

    def dv_at(self, i):
        return _langevin_fts.SimulationBox_dv_at(self, i)

    def multi_inner_product(self, n_comp, g, h):
        return _langevin_fts.SimulationBox_multi_inner_product(self, n_comp, g, h)

    def integral(self, *args):
        return _langevin_fts.SimulationBox_integral(self, *args)

    def inner_product(self, *args):
        return _langevin_fts.SimulationBox_inner_product(self, *args)

    def zero_mean(self, *args):
        return _langevin_fts.SimulationBox_zero_mean(self, *args)

# Register SimulationBox in _langevin_fts:
_langevin_fts.SimulationBox_swigregister(SimulationBox)

class Pseudo(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, sb, pc):
        _langevin_fts.Pseudo_swiginit(self, _langevin_fts.new_Pseudo(sb, pc))
    __swig_destroy__ = _langevin_fts.delete_Pseudo

    def get_partition(self, q1_out, q2_out, n):
        return _langevin_fts.Pseudo_get_partition(self, q1_out, q2_out, n)

    def find_phi(self, *args):
        return _langevin_fts.Pseudo_find_phi(self, *args)

# Register Pseudo in _langevin_fts:
_langevin_fts.Pseudo_swigregister(Pseudo)

class AndersonMixing(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def reset_count(self):
        return _langevin_fts.AndersonMixing_reset_count(self)

    def caculate_new_fields(self, *args):
        return _langevin_fts.AndersonMixing_caculate_new_fields(self, *args)

    def __init__(self):
        _langevin_fts.AndersonMixing_swiginit(self, _langevin_fts.new_AndersonMixing())
    __swig_destroy__ = _langevin_fts.delete_AndersonMixing

# Register AndersonMixing in _langevin_fts:
_langevin_fts.AndersonMixing_swigregister(AndersonMixing)

class KernelFactory(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _langevin_fts.KernelFactory_swiginit(self, _langevin_fts.new_KernelFactory(*args))

    def create_polymer_chain(self, f, NN, chi_n):
        return _langevin_fts.KernelFactory_create_polymer_chain(self, f, NN, chi_n)

    def create_simulation_box(self, *args):
        return _langevin_fts.KernelFactory_create_simulation_box(self, *args)

    def create_pseudo(self, sb, pc):
        return _langevin_fts.KernelFactory_create_pseudo(self, sb, pc)

    def create_anderson_mixing(self, sb, n_comp, max_anderson, start_anderson_error, mix_min, mix_init):
        return _langevin_fts.KernelFactory_create_anderson_mixing(self, sb, n_comp, max_anderson, start_anderson_error, mix_min, mix_init)
    __swig_destroy__ = _langevin_fts.delete_KernelFactory

# Register KernelFactory in _langevin_fts:
_langevin_fts.KernelFactory_swigregister(KernelFactory)


