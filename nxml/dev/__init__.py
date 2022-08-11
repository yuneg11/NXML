from typing import TYPE_CHECKING

from ..core import LazyModule

if TYPE_CHECKING:
    from . import jax
    from . import torch
    from . import general
else:
    jax = LazyModule(".jax", "jax", globals(), __package__)
    torch = LazyModule(".torch", "torch", globals(), __package__)
    general = LazyModule(".general", "general", globals(), __package__)


__all__ = [
    "jax",
    "torch",
    "general",
]
