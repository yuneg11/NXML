from __future__ import annotations

from typing import Optional, Union, Dict
from torch.nn import Module as TorchModule
from nxcl.config import ConfigDict


__all__ = [
    "Module",
]


ConfigType = Union[ConfigDict, Dict]


class Module(TorchModule):

    def get_config(self) -> ConfigType:
        raise NotImplementedError()

    @classmethod
    def from_config(cls, config: ConfigType, prefix: Optional[str] = None) -> Module:
        if prefix is not None:
            config = config.get(prefix)

        raise NotImplementedError()  # implement auto __init__ call
