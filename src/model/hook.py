"""
reference:
- most code extracted from fastai source code: https://github.com/fastai/fastai/blob/master/fastai/callbacks/hooks.py#L34
"""
import os
from collections import namedtuple


class Hook():
    def __init__(self, m, hook_func, detach = True):
        """
        
        :param:
            m : nn.Module, module to be hooked
            hook_func : function, must have the following signature:
                        hook_func(m: nn.Module, inp: Tensors, outp: Tensors)
            detach : bool, whether to deactivate grad mode for input, output
        """
        self.hook_func = hook_func
        self.detach = detach
        self.stored = None
        self.hook = m.register_forward_hook(self.hook_fn)
        self.removed = False
    
    def hook_fn(self, m, inp, outp):
        """
        :param:
            inp : tuple of Tensors, (BN, C, H, W)
            outp : tuple of Tensors, (BN, C, H, W)
        """
        if self.detach:
            inp = (o.detach() for o in inp) if isinstance(inp, tuple) else inp.detach()
            outp = (o.detach() for o in outp) if isinstance(outp, tuple) else outp.detach()
        self.stored = self.hook_func(m, inp, outp)
    
    def remove(self):
        if not self.removed:
            self.hook.remove()
            self.removed = True
    
    def __enter__(self, *args):
        """ 
        magic __enter__, __exit__ define behavior on with statement 
        """
        return self

    def __exit__(self, *args):
        self.remove()


class VGGHooks():
    def __init__(self, ms, detach = False):
        """
        assume ms is in the right order: vgg[3], vgg[8], vgg[15], vgg[22]
        """
        self.hooks = [Hook(m, self.hook_feature, detach = detach) for m in ms]

    def hook_feature(self, m, inp, outp):
        return outp

    def __getitem__(self,i):
        return self.hooks[i]

    def __len__(self): 
        return len(self.hooks)

    def __iter__(self): 
        return iter(self.hooks)

    @property
    def stored(self):
        vgg_tup = namedtuple(
            'VggOutputs', ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
            )
        return vgg_tup(*[o.stored for o in self])

    def remove(self):
        for h in self.hooks: h.remove()

    def __enter__(self, *args): 
        return self

    def __exit__ (self, *args): 
        self.remove()
