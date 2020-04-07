import os
from collections import namedtuple

import numpy as np
from PIL import Image

import torch
from tensorboardX import SummaryWriter
from fastai.vision import *
from fastai.basic_train import LearnerCallback

from src.common.lin_utils import gram_matrix
from src.common.os_utils import load_img, process_img
from src.common.vis_utils import plot_all_gram_matrices, plot_image_pair
from src.data.tfms import get_style_transforms, get_test_transforms
from src.model.hook import VGGHooks

import logging
logging.basicConfig(level = logging.INFO, handlers = [logging.StreamHandler()],
                    format = "%(asctime)s — %(name)s — %(levelname)s — %(message)s")


class EssentialCallback(LearnerCallback):
    """
    apply hooks on VGG features
    apply tensorboard to keep track of loss and other evolution
    """
    _order = 1
    hook_vgg_idxs = [3, 8, 15, 22]    

    def __init__(self, learn, chkpt_dir, content_path, style_path, plot_iter):
        """ pre-compute gram matrix for the style image """
        super().__init__(learn)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.named_tup = namedtuple('StyleGramMatrices', ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        
        self.init_hooks()
        self.precompute_style_gms(style_path)
        self.content_img = load_img(content_path)
        
        # init dir for plotting
        self.plot_iter = plot_iter
        self.chkpt_dir = chkpt_dir
        self.gm_dir = os.path.join(chkpt_dir, 'gram_matrix')
        self.test_dir = os.path.join(chkpt_dir, 'test')
        os.makedirs(self.gm_dir, exist_ok = True)
        os.makedirs(self.test_dir, exist_ok = True)
        
    def init_hooks(self):
        ms = [self.learn.model.vgg.subnet[idx] for idx in self.hook_vgg_idxs]
        # hooks are used in backprog
        self._hooks = VGGHooks(ms, detach = False)
        logging.info('hooks are initialized')

    @property
    def hooks(self):
        return self._hooks.stored

    def precompute_style_gms(self, style_path):
        # infer batch size from train dataloader
        bs = self.learn.data.train_dl.batch_size
        # infer image size from train dataloader sample
        x_sample, _ = next(iter(self.learn.data.train_dl))
        img_size = x_sample.shape[2]

        # load in and transform style image
        style_img = load_img(style_path)
        logging.info(f'read style img: {style_path}')
        style_t = get_style_transforms(img_size)(style_img)
        style_t = style_t.repeat(bs, 1, 1, 1).to(self.device)
        with torch.no_grad():
            _ = self.learn.model(style_t, vgg_only = True)

        # store gram matrix as namedtuple
        self.style_gms = self.named_tup(*[gram_matrix(t) for t in self.hooks])
        logging.info(f'gram matrices for style image is precomputed')

    def on_train_begin(self, **kwargs):
        return {'skip_validate': True}

    def on_batch_begin(self, last_input, **kwargs):
        """ update state_dict['last_target'] """
        with torch.no_grad():
            _ = self.learn.model(last_input, vgg_only = True)
        target_dict = {'content_target': self.hooks, 'style_target': self.style_gms}
        return {'last_target':  target_dict}

    def on_loss_begin(self, last_input, **kwargs):
        """ update state_dict['last_output'], list of feature maps """
        return {'last_output': self.hooks}
    
    def on_batch_end(self, iteration, **kwargs):
        """
        plot and save evolution of gram matrix + stylised image 
        """
        if iteration % self.plot_iter == 0:
            content_t = get_test_transforms()(self.content_img)
            content_t = content_t.unsqueeze(0).to(self.device)
            
            # plot gram matrices
            with torch.no_grad():
                _ = self.learn.model(content_t, vgg_only = False)
            content_gms = self.named_tup(*[gram_matrix(t) for t in self.hooks])
            plot_all_gram_matrices(content_gms, self.style_gms, iteration, self.gm_dir)
            
            # plot stylised content image
            with torch.no_grad():
                stylised = self.learn.model.transformer(content_t)
            plot_image_pair(stylised[0], content_t[0], iteration, self.test_dir)
        return None
        

class SaveCallback(LearnerCallback):
    _order = 2
    
    def __init__(self, learn, chkpt_epoch, chkpt_model_dir):
        super().__init__(learn)
        self.chkpt_epoch = chkpt_epoch
        self.chkpt_model_dir = chkpt_model_dir
        os.makedirs(chkpt_model_dir, exist_ok = True)
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

    def save_transformer(self, chkpt_model_path, is_train = True):
        self.learn.model.transformer.eval().cpu()
        torch.save(self.learn.model.transformer.state_dict(), chkpt_model_path)
        if is_train:
            self.learn.model.transformer.to(self.device).train()
        
    def on_epoch_end(self, epoch, **kwargs):
        if epoch % self.chkpt_epoch == 0:            
            chkpt_model_fname = f'chkpt_epoch_{epoch:03}.pth'
            chkpt_model_path = os.path.join(self.chkpt_model_dir, chkpt_model_fname)
            self.save_transformer(chkpt_model_path, is_train = True)
            logging.info(f'[epoch: {epoch}] model saved: {chkpt_model_path}')
        return None

    def on_train_end(self, epoch, **kwargs):
        chkpt_model_fname = f'final_epoch_{epoch:03}.pth'
        chkpt_model_path = os.path.join(self.chkpt_model_dir, chkpt_model_fname)
        self.save_transformer(chkpt_model_path, is_train = False)
        logging.info(f'[train complete] model saved: {chkpt_model_path}')

        
class TensorboardCallback(LearnerCallback):
    _order = 3
    
    def __init__(self, learn, log_dir, update_iter):
        super().__init__(learn=learn)
        self.tbwriter = SummaryWriter(log_dir)
        self.update_iter = update_iter
    
    def on_batch_end(self, iteration, **kwargs):
        if iteration % self.update_iter == 0:
            content_loss = self.learn.loss_func.content_loss
            style_loss = self.learn.loss_func.style_loss
            total_loss = content_loss + style_loss
            
            tag = '/metrics/loss'
            loss_dict = {'content_loss': content_loss, 
                         'style_loss': style_loss, 
                         'total_loss': total_loss}
            self.tbwriter.add_scalars(tag, loss_dict, iteration)
        return None