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
from src.train.callback import EssentialCallback, SaveCallback, TensorboardCallback

import logging
logging.basicConfig(level = logging.INFO, handlers = [logging.StreamHandler()],
                    format = "%(asctime)s — %(name)s — %(levelname)s — %(message)s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fire a training for fast neural style transfer')
    parser.add_argument('--data_dir', required = True, type = str, help = 'path to the data directory')
    parser.add_argument('--style_img_path', required = True, type = str, help = 'path to the style image')
    parser.add_argument('--content_img_path', required = True, type = str, help = 'path to the content image to be plotted in gram matrix and stylisation evolution')
    parser.add_argument('--chkpt_model_dir', required = True, type = str, help = 'dir for saving model, tensorboard and plots')

    # optional training setup
    parser.add_argument('--img_size', default = 128, type = int, help = 'resize for training')
    parser.add_argument('--bs', default = 16, type = int, help = 'batch size')
    parser.add_argument('--content_weight', default = 1e5, type = float, help = 'weight for content weight')
    parser.add_argument('--style_weight', default = 1e10, type = float, help = 'weight for style loss, need to be high enough as style loss has small scale')
    parser.add_argument('--n_epochs', default = 2, type = int, 'no. of epochs')
    parser.add_argument('--lr', default = 5e-3, type = float, help = 'learning rate') # found by learner.lr_find
    parser.add_argument('--update_iter', default = 200, type = int, help = 'iterations to checkpoint test image and gram matrix plotting')
    parser.add_argument('--is_downsample', action = 'store_true', help = 'whether to use downsample transformer for speeding up both training and inference')

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
    data.val_dl = None

    # setup model and callbacks
    if args.is_downsample:
        model = MetaModel(is_downsample = True, vgg_grad = False)
    else:
        model = MetaModel(is_downsample = False, vgg_grad = False)

    essential_cb = partial(
        EssentialCallback, 
        chkpt_dir = args.chkpt_model_dir,
        content_path = args.content_img_path,
        style_path = args.style_img_path,
        plot_iter = args.update_iter
        )
    save_cb = partial(
        SaveCallback, 
        chkpt_epoch = 1, 
        chkpt_model_dir = args.chkpt_model_dir
        )
    tb_cb = partial(
        TensorboardCallback,
        log_dir = args.chkpt_model_dir,
        update_iter = args.update_iter
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
        callback_fns = [essential_cb, save_cb, tb_cb],
        layer_groups = model.transformer
        )

    start = time.time()
    if args.is_one_cycle:
        learn.fit_one_cycle(args.n_epochs, args.lr)
    else:
        learn.fit(args.n_epochs, args.lr)
    end = time.time()
    print(f'training complete: {(end - start) / 60} mins')
