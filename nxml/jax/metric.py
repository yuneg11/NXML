from typing import Any, Optional
from functools import partial, wraps

from jax import jit

from ..univ.metric import (
    nll_loss as _nll_loss,
    accuracy as _accuracy,
)

Array = Any


__all__ = [
    "nll_loss",
    "accuracy",
]


@wraps(_nll_loss)
@partial(jit, static_argnames=("ignore_index", "reduction"))
def nll_loss(
    input: Array,
    target: Array,
    weight: Optional[Array] = None,
    ignore_index: int = -100,
    reduction: str = "mean",
):
    return _nll_loss(input, target, weight, ignore_index=ignore_index, reduction=reduction)


# TODO: Change defaults when ignore_index implemented
@wraps(_accuracy)
@partial(jit, static_argnames=("ignore_index", "reduction"))
def accuracy(
    input: Array,
    target: Array,
    ignore_index: int = -100,
    reduction: str = "mean",
):
    return _accuracy(input, target, ignore_index=ignore_index, reduction=reduction)
