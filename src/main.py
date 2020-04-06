import os
import time
import argparse
from functools import partial

from fastai import *
from fastai.vision import *
from torch.utils.data import DataLoader

from src.model.meta_model import MetaModel
from src.data.dataset import NeuralDataset
from src.data.tfms import get_transforms
from src.train.loss import PerceptualLoss
from src.train.callback import EssentialCallback, SaveCallback

import logging
logging.basicConfig(level = logging.INFO, handlers = [logging.StreamHandler()],
                    format = "%(asctime)s — %(name)s — %(levelname)s — %(message)s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fire a training for fast neural style transfer')
    parser.add_argument('--data_dir', required = True, type = str)
    parser.add_argument('--style_img_path', required = True, type = str)
    parser.add_argument('--chkpt_model_dir', required = True, type = str)

    # optional training setup
    parser.add_argument('--img_size', default = 128, type = int)
    parser.add_argument('--bs', default = 16, type = int)
    parser.add_argument('--content_weight', default = 1e5, type = float)
    parser.add_argument('--style_weight', default = 1e10, type = float)
    parser.add_argument('--n_epochs', default = 2, type = int)
    parser.add_argument('--lr', default = 1e-3, type = float)

    # optional functionality from fastai
    parser.add_argument('--is_one_cycle', action = 'store_true')
    args = parser.parse_args()

    # setup data
    tfms = get_transforms(args.img_size)
    ds = NeuralDataset(args.data_dir, transform = tfms)
    train_dl = DataLoader(ds, batch_size = args.bs, shuffle = True, num_workers = 0)
    # val_dl serves as dummy
    val_dl = DataLoader(ds, batch_size = 1, shuffle = False, num_workers = 0)
    data = DataBunch(train_dl, val_dl)

    # setup model and callbacks
    model = MetaModel(vgg_grad = False)
    essential_cb = partial(
        EssentialCallback, 
        style_img_path = args.style_img_path
        )
    save_cb = partial(
        SaveCallback, 
        chkpt_epoch = 1, 
        chkpt_model_dir = args.chkpt_model_dir
        )

    # setup loss
    perceptual_loss = partial(
        PerceptualLoss, model = model, 
        content_weight = args.content_weight, 
        style_weight = args.style_weight
        )

    # setup learner and start training
    learn = Learner(
        data, model, # only optimize on transformer net 
        loss_func = perceptual_loss(), 
        opt_func = partial(optim.Adam, betas = (0.5, 0.99)) if not args.is_one_cycle else optim.Adam,
        callback_fns = [essential_cb, save_cb],
        layer_groups = model.transformer
        )

    start = time.time()
    if args.is_one_cycle:
        learn.fit_one_cycle(args.n_epochs, args.lr)
    else:
        learn.fit(args.n_epochs, args.lr)
    end = time.time()
    print(f'training complete: {(end - start) / 60} mins')
