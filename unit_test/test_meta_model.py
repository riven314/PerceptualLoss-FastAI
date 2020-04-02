import os

import torch

from src.model.meta_model import MetaModel
from unit_test.test_common import set_seed

set_seed(100)

meta_model = MetaModel()
t1 = torch.randn(1, 3, 64, 64).cuda()
t2 = torch.randn(1, 3, 64, 64).cuda()

out1 = meta_model(t1)
out2 = meta_model(t2)