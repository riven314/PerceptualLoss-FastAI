import os
import time
from functools import partial

from fastai import *
from fastai.vision import *

from src.model.meta_model import MetaModel
from src.data.dataset import NeuralDataset
from src.data.tfms import get_transforms
from src.train.loss import PerceptualLoss
from src.train.callback import EssentialCallback, SaveCallback

import logging
logging.basicConfig(level = logging.INFO, handlers = [logging.StreamHandler()],
                    format = "%(asctime)s — %(name)s — %(levelname)s — %(message)s")


img_style_path = None
data_dir = None
img_size = 128
bs = 8
content_weight = 1e5
style_weight = 1e10
is_one_cycle = True
n_epochs = 2
lr = 1e-3

ds = NeuralDataset(data_dir, transform = get_transforms)
data = DataBunch(ds)

model = MetaModel(vgg_grad = False)

essential_cb = partial(
    EssentialCallback, 
    meta_model = model, 
    img_style_path = img_style_path, 
    img_size = img_size, bs = bs
    )
save_cb = partial(
    SaveCallback, meta_model = model, chkpt_epoch = 1
    )

perceptual_loss = partial(
    PerceptualLoss, model = model, 
    content_weight = content_weight, 
    style_weight = style_weight
    )


learn = Learner(data, model, 
                loss_func = perceptual_loss(), 
                metrics = [metrics.loss],
                callback_fns = [essential_cb, save_cb])

start = time.time()
if is_one_cycle:
    learn.fit_one_cycle(n_epochs, lr)
    suffix = 'wcycle'
else:
    learn.fit(n_epochs, lr)
    suffix = 'wocycle'
learn.save(f'coco_{suffix}_e{epoch:03}fit')
end = time.time()
print(f'training complete: {(end - start) / 60} mins')