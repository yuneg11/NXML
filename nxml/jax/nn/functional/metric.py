from typing import Any, Optional
from functools import wraps

from jax import jit

from nxml.univ.nn.functional.metric import (
    nll_loss as _nll_loss,
    accuracy as _accuracy,
)

Array = Any


__all__ = [
    "nll_loss",
    "accuracy",
]


# jitted versions functions

jit_nll_loss = jit(_nll_loss, static_argnames=("ignore_index", "reduction"))
jit_accuracy = jit(_accuracy, static_argnames=("ignore_index", "reduction"))


# wrappers for the jitted versions

@wraps(_nll_loss)
def nll_loss(
    input: Array,
    target: Array,
    weight: Optional[Array] = None,
    ignore_index: int = -100,
    reduction: str = "mean",
):
    return jit_nll_loss(input, target, weight, ignore_index=ignore_index, reduction=reduction)


@wraps(_accuracy)
def accuracy(
    input: Array,
    target: Array,
    ignore_index: int = -100,
    reduction: str = "mean",
):
    return jit_accuracy(input, target, ignore_index=ignore_index, reduction=reduction)


# TODO: Change wraps to real "docstrings" to support IDE hints.
