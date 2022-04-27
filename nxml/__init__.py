from typing import TYPE_CHECKING

from nxml import core
from nxml.core import LazyModule

if TYPE_CHECKING:
    from nxml import jax
    from nxml import torch
    from nxml import univ
else:
    jax   = LazyModule("nxml.jax")
    torch = LazyModule("nxml.torch")
    univ  = LazyModule("nxml.univ")


__version__ = "0.0.1.dev0"
