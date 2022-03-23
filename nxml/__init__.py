from typing import TYPE_CHECKING

from nxml import core
from nxml.core import LazyModule

if TYPE_CHECKING:
    from nxml import functional
    from nxml import jax
    from nxml import torch
else:
    functional = LazyModule("nxml.functional")
    jax        = LazyModule("nxml.jax")
    torch      = LazyModule("nxml.torch")


__version__ = "0.0.1"
