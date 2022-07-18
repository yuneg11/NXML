from typing import TYPE_CHECKING

from . import core
from .core import LazyModule

if TYPE_CHECKING:
    from . import numpy
    from . import jax
    from . import tensorflow
    from . import torch
    from . import general
else:
    numpy      = LazyModule("nxml.numpy")
    jax        = LazyModule("nxml.jax")
    torch      = LazyModule("nxml.torch")
    tensorflow = LazyModule("nxml.tensorflow")
    general    = LazyModule("nxml.general")


__all__ = [
    "core",
    "numpy",
    "jax",
    "torch",
    "tensorflow",
    "general",
]

__version__ = "0.0.4"
