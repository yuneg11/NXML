# Source code modified from https://github.com/facebookresearch/detectron2
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from functools import lru_cache

import torch
from torch import distributed as dist
from torch.distributed import Backend


__all__ = [
    "get_world_size",
    "get_local_world_size",
    "get_rank",
    "get_local_rank",
    "is_master_process",
    "synchronize",
    "all_gather",
    "gather",
    "share_random_seed",
    "reduce_dict",
]


# A torch process group which only includes processes that on the same machine as the current process.
# This variable is set when processes are spawned by `launch()` in "engine/launch.py".

_LOCAL_PROCESS_GROUP = None


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_local_world_size() -> int:
    """
    Returns:
        The size of the per-machine process group,
        i.e. the number of processes per machine.
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size(group=_LOCAL_PROCESS_GROUP)


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    """
    Returns:
        The rank of the current process within the local (per-machine) process group.
    """

    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    assert (
        _LOCAL_PROCESS_GROUP is not None
    ), "Local process group is not created! Please use launch() to spawn processes!"
    return dist.get_rank(group=_LOCAL_PROCESS_GROUP)


def is_master_process() -> bool:
    return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """

    if not dist.is_available():
        return
    if not dist.is_initialized():
        return

    world_size = dist.get_world_size()
    if world_size == 1:
        return

    if dist.get_backend() == Backend.NCCL:
        # This argument is needed to avoid warnings.
        # It's valid only for NCCL backend.
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


@lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """

    if dist.get_backend() == Backend.NCCL:
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD


def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: list of data gathered from each rank
    """

    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()  # use CPU group by default, to reduce GPU RAM usage.

    world_size = dist.get_world_size(group)
    if world_size == 1:
        return [data]

    output = [None for _ in range(world_size)]
    dist.all_gather_object(output, data, group=group)
    return output


def gather(data, dst=0, group=None):
    """
    Run gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        dst (int): destination rank
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: on dst, a list of data gathered from each rank. Otherwise,
            an empty list.
    """

    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()

    world_size = dist.get_world_size(group=group)
    if world_size == 1:
        return [data]
    rank = dist.get_rank(group=group)

    if rank == dst:
        output = [None for _ in range(world_size)]
        dist.gather_object(data, output, dst=dst, group=group)
        return output
    else:
        dist.gather_object(data, None, dst=dst, group=group)
        return []


def share_random_seed(seed: int):
    """
    Returns:
        int: a random number that is the same across all workers.
        If workers need a shared RNG, they can use this shared seed to create one.

    All workers must call this function, otherwise it will deadlock.
    """

    all_ints = all_gather(seed)
    return all_ints[0]


def reduce_dict(input_dict, average=True):
    """
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the reduced results.

    Args:
        input_dict (dict): inputs to be reduced. All the values must be scalar CUDA Tensor.
        average (bool): whether to do average or sum

    Returns:
        a dict with the same keys as input_dict, after reduction.
    """

    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.reduce(values, dst=0)
        if dist.get_rank() == 0 and average:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict
