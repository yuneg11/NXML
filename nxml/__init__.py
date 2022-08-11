from typing import TYPE_CHECKING

from . import core
from .core import LazyModule

if TYPE_CHECKING:
    from . import general
    from . import numpy
    from . import torch
    from . import jax
    from . import tensorflow
    from . import dev
else:
    general = LazyModule(".general", "general", globals(), __package__)
    numpy = LazyModule(".numpy", "numpy", globals(), __package__)
    torch = LazyModule(".torch", "torch", globals(), __package__)
    jax = LazyModule(".jax", "jax", globals(), __package__)
    tensorflow = LazyModule(".tensorflow", "tensorflow", globals(), __package__)
    dev = LazyModule(".dev", "dev", globals(), __package__)


__all__ = [
    "core",
    "general",
    "numpy",
    "torch",
    "jax",
    "tensorflow",
    "dev",
]

__version__ = "0.0.5"
