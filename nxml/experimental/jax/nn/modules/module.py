from __future__ import annotations

from typing import Optional, Union, Dict
from flax.linen import Module as FlaxModule
from nxcl.config import ConfigDict


__all__ = [
    "Module",
]


ConfigType = Union[ConfigDict, Dict]


class Module(FlaxModule):

    def get_config(self) -> ConfigType:
        raise NotImplementedError()

    @classmethod
    def from_config(cls, config: ConfigType, prefix: Optional[str] = None) -> Module:
        if prefix is not None:
            config = config.get(prefix)

        raise NotImplementedError()  # implement auto __init__ call
