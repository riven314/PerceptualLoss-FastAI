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


data_dir = '/userhome/34/h3509807/train2014'
style_img_path = 'style/cuson_arts.jpg'
chkpt_model_dir = 'chkpts'
img_size = 128
bs = 16
content_weight = 1e5
style_weight = 1e10
is_one_cycle = True
n_epochs = 2
lr = 1e-3

tfms = get_transforms(img_size)
ds = NeuralDataset(data_dir, transform = tfms)
train_dl = DataLoader(
	        ds, batch_size = bs, shuffle = True, num_workers = 0
	        )
val_dl = DataLoader(
            ds, batch_size = 1, shuffle = False, num_workers = 0
            )
data = DataBunch(train_dl, val_dl)

model = MetaModel(vgg_grad = False)

essential_cb = partial(
    EssentialCallback, 
    meta_model = model, 
    style_img_path = style_img_path, 
    img_size = img_size, bs = bs
    )
save_cb = partial(
    SaveCallback, meta_model = model, 
    chkpt_epoch = 1, chkpt_model_dir = os.getcwd()
    )

perceptual_loss = partial(
    PerceptualLoss, model = model, 
    content_weight = content_weight, 
    style_weight = style_weight
    )


learn = Learner(data, model, # only optimize on transformer net 
                loss_func = perceptual_loss(), 
                opt_func = partial(optim.Adam, betas = (0.5, 0.99)) if is_one_cycle else optim.Adam,
                callback_fns = [essential_cb, save_cb],
                layer_groups = model.transformer)

start = time.time()
if is_one_cycle:
    learn.fit_one_cycle(n_epochs, lr)
    suffix = 'wcycle'
else:
    learn.fit(n_epochs, lr)
    suffix = 'wocycle'
end = time.time()
print(f'training complete: {(end - start) / 60} mins')
