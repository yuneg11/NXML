from typing import Any, Optional
from functools import wraps

from torch import Tensor
from torch.nn.functional import (
    nll_loss as _nll_loss,
)

from nxml.univ.nn.functional.metric import (
    accuracy as _accuracy,
)


__all__ = [
    "nll_loss",
    "accuracy",
]


# wrappers

@wraps(_nll_loss)
def nll_loss(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    ignore_index: int = -100,
    reduction: str = "mean",
):
    return _nll_loss(input, target, weight, ignore_index=ignore_index, reduction=reduction)


@wraps(_accuracy)
def accuracy(
    input: Tensor,
    target: Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
):
    return _accuracy(input, target, ignore_index=ignore_index, reduction=reduction)


# TODO: Change wraps to real "docstrings" to support IDE hints.
