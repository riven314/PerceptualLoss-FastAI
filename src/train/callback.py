import os
from collections import namedtuple

import torch
from fastai.basic_train import LearnerCallback

from src.common.lin_utils import gram_matrix
from src.common.os_utils import load_img
from src.data.tfms import get_style_transforms
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

    def __init__(self, learn, style_img_path):
        """ pre-compute gram matrix for the style image """
        super().__init__(learn)
        self.init_hooks()
        self.precompute_style_gms(style_img_path)
        
    def init_hooks(self):
        ms = [self.learn.model.vgg.subnet[idx] for idx in self.hook_vgg_idxs]
        # hooks are used in backprog
        self._hooks = VGGHooks(ms, detach = False)
        logging.info('hooks are initialized')

    @property
    def hooks(self):
        return self._hooks.stored

    def precompute_style_gms(self, style_img_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # infer batch size from train dataloader
        bs = self.learn.data.train_dl.batch_size
        # infer image size from train dataloader sample
        x_sample, _ = next(iter(self.learn.data.train_dl))
        img_size = x_sample.shape[2]

        # load in and transform style image
        style_img = load_img(style_img_path)
        logging.info(f'read style img: {style_img_path}')
        style_t = get_style_transforms(img_size)(style_img)
        style_t = style_t.repeat(bs, 1, 1, 1).to(device)
        with torch.no_grad():
            _ = self.learn.model(style_t, vgg_only = True)

        # store gram matrix as namedtuple
        gms_tup = namedtuple('StyleGramMatrices', 
                             ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        self.style_gms = gms_tup(*[gram_matrix(t) for t in self.hooks])
        logging.info(f'gram matrices for style image is precomputed')

    def on_train_begin(self, **kwargs):
        return {'skip_validate': True}

    def on_batch_begin(self, last_input, **kwargs):
        """ update state_dict['last_target'] """
        with torch.no_grad():
            _ = self.learn.model(last_input, vgg_only = True)
        target_dict = {
            'content_target': self.hooks,
            'style_target': self.style_gms
            }
        return {'last_target':  target_dict}

    def on_loss_begin(self, last_input, **kwargs):
        """ update state_dict['last_output'] """
        return {'last_output': self.hooks}
        

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
