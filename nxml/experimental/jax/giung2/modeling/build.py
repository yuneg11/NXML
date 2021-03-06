import jax.numpy as jnp
import flax.linen as nn

from nxcl.config import ConfigDict as CfgNode
from .architecture import *
from .backbone import *
from .classifier import *


def build_backbone(cfg: CfgNode) -> nn.Module:
    """
    Args:
        cfg: CfgNode instance that contains blueprint of a backbone.

    Returns:
        A backbone network.
    """
    name = cfg.MODEL.BACKBONE.NAME

    if name == 'ResNet':
        backbone = build_resnet_backbone(cfg)
    elif name == 'PreResNet':
        backbone = build_preresnet_backbone(cfg)
    elif name == 'ResNeXt':
        backbone = build_resnext_backbone(cfg)
    elif name == 'VGGNet':
        backbone = build_vggnet_backbone(cfg)
    elif name == 'LeNet':
        backbone = build_lenet_backbone(cfg)
    elif name == 'ViT':
        backbone = build_vit_backbone(cfg)
    else:
        raise NotImplementedError(
            f'Unknown cfg.MODEL.BACKBONE.NAME: {name}'
        )

    return backbone


def build_classifier(cfg: CfgNode) -> nn.Module:
    """
    Args:
        cfg: CfgNode instance that contains blueprint of a classifier.

    Returns:
        A classifier.
    """
    name = cfg.MODEL.CLASSIFIER.NAME

    if name == 'SoftmaxClassifier':
        classifier = build_softmax_classifier(cfg)
    else:
        raise NotImplementedError(
            f'Unknown cfg.MODEL.CLASSIFIER.NAME: {name}'
        )

    return classifier


def build_model(cfg: CfgNode) -> nn.Module:
    """
    Args:
        cfg: CfgNode instance that contains blueprint of a model.

    Returns:
        A model consisting of backbone and classifier.
    """
    name = cfg.MODEL.META_ARCHITECTURE.NAME

    if name == 'ImageClassificationModelBase':
        model = ImageClassificationModelBase(
            backbone   = build_backbone(cfg),
            classifier = build_classifier(cfg),
            pixel_mean = jnp.asarray(cfg.MODEL.PIXEL_MEAN),
            pixel_std  = jnp.asarray(cfg.MODEL.PIXEL_STD),
        )
    else:
        raise NotImplementedError(
            f'Unknown cfg.MODEL.META_ARCHITECTURE.NAME: {name}'
        )

    return model
