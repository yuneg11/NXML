from typing import Any, Optional
from functools import wraps

from tensorflow import function

from nxml.univ.nn.functional import (
    nll_loss as _nll_loss,
    cross_entropy as _cross_entropy,
    accuracy as _accuracy,
)

Tensor = Any


__all__ = [
    "nll_loss",
    "cross_entropy",
    "accuracy",
]


fn_nll_loss = function(
    _nll_loss,
)


@wraps(_nll_loss)
def nll_loss(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    ignore_index: int = -100,
    reduction: str = "mean",
):
    return fn_nll_loss(input, target, weight, ignore_index=ignore_index, reduction=reduction)


fn_cross_entropy = function(
    _cross_entropy,
)


@wraps(_cross_entropy)
def cross_entropy(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
):
    return fn_cross_entropy(
        input, target, weight, ignore_index=ignore_index, reduction=reduction,
        label_smoothing=label_smoothing,
    )


fn_accuracy = function(
    _accuracy,
)


@wraps(_accuracy)
def accuracy(
    input: Tensor,
    target: Tensor,
    topk: int = 1,
    ignore_index: int = -100,
    reduction: str = "mean",
):
    return fn_accuracy(input, target, topk=topk, ignore_index=ignore_index, reduction=reduction)


# TODO: Change wraps to real "docstrings" to support IDE hints.
