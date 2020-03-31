import os

HOOK_VGG_IDXS = [3, 8, 15, 22] # layer indexes of VGG net to be hooked


class HookVGGFeatures:
    def __init__(self, m):
        self.handle = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, m, inp, outp):
        self.features = outp
    def remove(self):
        self.handle.remove()