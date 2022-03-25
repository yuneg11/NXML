# Docstrings are copied from the original PyTorch implementation.
# See: https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py

"""
Implementation of framework-agnostic metrics.
"""

from typing import Optional

import eagerpy as ep
from eagerpy import Tensor


__all__ = [
    "nll_loss",
    "accuracy",
]


def _reduction(
    input: Tensor,
    reduction: str,
) -> Tensor:
    """
    Reduce the input tensor.

    Args:
        input (Tensor): _description_
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``

    Returns:
        reduced_input (Tensor): _description_
    """

    if reduction == "mean":
        return ep.mean(input)
    elif reduction == "sum":
        return ep.sum(input)
    elif reduction == "none":
        return input
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def nll_loss(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> Tensor:
    r"""
    Negative log likelihood

    Args:
        input (Tensor): :math:`(N, C)` where `C = number of classes` or :math:`(N, C, H, W)`
            in case of 2D Loss, or :math:`(N, C, d_1, d_2, ..., d_K)` where :math:`K \geq 1`
            in the case of K-dimensional loss. `input` is expected to be log-probabilities.
        target (Tensor): :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`,
            or :math:`(N, d_1, d_2, ..., d_K)` where :math:`K \geq 1` for K-dimensional loss.
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, has to be a Tensor of size `C`. Default: None
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When :attr:`size_average` is
            ``True``, the loss is averaged over non-ignored targets. Default: -100
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``

    Returns:
        nll (Tensor)
    """

    input, target = ep.astensors(input, target)
    if weight is not None:
        weight = ep.astensor(weight)

    # TODO: Implement ignore_index
    if ignore_index != -100:
        raise NotImplementedError("ignore_index is not implemented")


    return _reduction(input, reduction)


def accuracy(
    input: Tensor,
    target: Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> Tensor:
    """
    Classification accuracy

    Args:
        input (Tensor): :math:`(N, C)` where `C = number of classes` or :math:`(N, C, H, W)`
            in case of 2D Loss, or :math:`(N, C, d_1, d_2, ..., d_K)` where :math:`K \geq 1`
            in the case of K-dimensional loss. `input` is expected to be logits, log-probabilities
            or probabilities.
        target (Tensor): :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`,
            or :math:`(N, d_1, d_2, ..., d_K)` where :math:`K \geq 1` for K-dimensional loss.
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, has to be a Tensor of size `C`. Default: None
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When :attr:`size_average` is
            ``True``, the loss is averaged over non-ignored targets. Default: -100
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``

    Returns:
        accuracy (Tensor)
    """

    input, target = ep.astensors(input, target)
    if weight is not None:
        weight = ep.astensor(weight)

    # TODO: Implement ignore_index
    if ignore_index != -100:
        raise NotImplementedError("ignore_index is not implemented")

    return _reduction(input, reduction)
