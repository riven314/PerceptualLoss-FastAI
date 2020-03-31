import os
import random

import numpy as np
import torch

import logging
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()],
                    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s")


def set_seed(seed):
    """
    after set_seed(),
    sequence of torch.randn(10, 10) are always the same

    e.g.
    set_seed(100)
    t1, t2 = torch.randn(10, 10), torch.randn(10, 10)
    set_seed(100)
    t3, t4 = torch.randn(10, 10), torch.randn(10, 10)
    
    ** t1 = t3, t2 = t4
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logging.info(f'manual seed is set')
    return None
    