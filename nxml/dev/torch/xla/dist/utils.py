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


from torch.nn import Module


__all__ = [
    "create_ddp_model",
]


def create_ddp_model(
    model: Module,
    *,
    device_ids = None,
    # fp16_compression=False,
    **kwargs,
) -> Module:
    """
    Dummy function to match the interface with GPU version.
    """

    return model
