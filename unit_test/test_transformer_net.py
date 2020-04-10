import os

import torch

from src.model.transformer_net import TransformerNet, TransformerNetDownsample


t = torch.randn(8, 3, 128, 128)
transformer = TransformerNet()
transformer_down = TransformerNetDownsample()

out1 = transformer(t)
out2 = transformer_down(t)
assert out1.shape == out2.shape, 'out1 must have same shape as out2'