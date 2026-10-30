"""
PUD Operation Library
=====================

Each operation is a generator function that produces a concrete
``UProgram`` (list of ``Primitive`` objects) for a given bit-precision.

Usage::

    from pud_simulator.operations import get_operation, list_operations
    uprogram = get_operation("addition")(n=32)
"""

from typing import Callable, Dict
from ..primitives import UProgram

# Registry: short_name -> generator function
_REGISTRY: Dict[str, Callable[[int], UProgram]] = {}


def register(short_name: str):
    """Decorator to register an operation generator."""
    def decorator(fn):
        _REGISTRY[short_name] = fn
        return fn
    return decorator


def get_operation(name: str) -> Callable[[int], UProgram]:
    """Look up a registered operation by short name."""
    key = name.lower().replace("-", "_")
    if key not in _REGISTRY:
        raise KeyError(
            f"Unknown operation '{name}'. "
            f"Available: {', '.join(sorted(_REGISTRY.keys()))}"
        )
    return _REGISTRY[key]


def list_operations():
    """Return sorted list of (short_name, docstring) pairs."""
    return [(k, _REGISTRY[k].__doc__ or "") for k in sorted(_REGISTRY)]


# Import submodules to trigger registration
from . import arithmetic   # noqa: F401, E402
from . import comparison   # noqa: F401, E402
from . import reduction    # noqa: F401, E402
from . import other        # noqa: F401, E402
