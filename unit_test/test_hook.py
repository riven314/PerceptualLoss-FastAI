import os
from collections import namedtuple

import torch

from src.model.vgg import VGG16, VGG16_woHook
from src.model.hook import VGGHooks
from unit_test.test_common import set_seed

import logging
logging.basicConfig(level = logging.INFO, handlers = [logging.StreamHandler()],
                    format = "%(asctime)s — %(name)s — %(levelname)s — %(message)s")


HOOK_VGG_IDXS = [3, 8, 15, 22] # layer indexes of VGG net to be hooked
set_seed(100)


def test_vgg_hook():
    t = torch.randn(1, 3, 64, 64)
    vgg16 = VGG16()

    # setup vgg hook
    ms = [vgg16.subnet[i] for i in HOOK_VGG_IDXS]
    hooks = VGGHooks(ms)
    _ = vgg16(t)
    # store hook output as namedtuple
    hook_stored = hooks.stored
    out_whook = hook_stored

    # vgg without hook
    vgg16_wohook = VGG16_woHook()
    out_wohook = vgg16_wohook(t)

    # sanity check
    names = [i for i in dir(hook_stored) if 'relu' in i]
    is_pass = False
    for name, a, b in zip(names, out_whook, out_wohook):
        assert (a == b).all().item(), f'feature not match'
        logging.info(f'[{name}] {True}')
        is_pass = True
    assert is_pass, 'is_pass must be True'

    return out_whook, out_wohook


if __name__ == '__main__':
    out_whook, out_wohook = test_vgg_hook()