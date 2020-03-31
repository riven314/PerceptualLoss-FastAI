import os
from collections import namedtuple

import torch

from src.model.vgg import VGG16, VGG16_woHook
from src.model.hook import HookVGGFeatures, HOOK_VGG_IDXS
from unit_test.test_common import set_seed

set_seed(100)


def test_vgg_hook():
    t = torch.randn(1, 3, 64, 64)
    vgg16 = VGG16()

    # setup vgg hook
    vgg16_tup = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
    out_whook = vgg16_tup(
        *[HookVGGFeatures(vgg16.vgg_subnet[i]) for i in HOOK_VGG_IDXS]
        )
    _ = vgg16(t)

    # vgg without hook
    vgg16_wohook = VGG16_woHook()
    out_wohook = vgg16_wohook(t)

    # sanity check
    for hook, feat in zip(out_whook, out_wohook):
        assert (hook.features == feat).all().item(), f'feature not match'
    return out_whook, out_wohook


if __name__ == '__main__':
    out_whook, out_wohook = test_vgg_hook()