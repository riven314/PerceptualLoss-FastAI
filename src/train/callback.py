import os
from collections import namedtuple

import torch
from fastai.basic_train import LearnerCallback

from src.common.lin_utils import gram_matrix
from src.common.os_utils import load_img
from src.data.tfms import get_style_transforms

import logging
logging.basicConfig(level = logging.INFO, handlers = [logging.StreamHandler()],
                    format = "%(asctime)s — %(name)s — %(levelname)s — %(message)s")


class EssentialCallback(LearnerCallback):
    """
    workflow that feed right inputs to the loss function
    compute the feature-wise gram matrix of a style image only once
    """
    _order = 1

    def __init__(self, learn, meta_model, style_img_path, img_size, bs):
        """ pre-compute gram matrix for the style image """
        super().__init__(learn)
        # pretransform style image
        self.precompute_style_gms(
            meta_model, style_img_path, img_size, bs
            )

    def precompute_style_gms(self, meta_model, style_img_path, img_size, bs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        style_img = load_img(style_img_path)
        logging.info(f'read style img: {style_img_path}')
        style_t = get_style_transforms(img_size)(style_img)
        style_t = style_t.repeat(bs, 1, 1, 1).to(device)
        style_batch = meta_model(style_t, vgg_only = True)
        # store gram matrix as namedtuple
        gms_tup = namedtuple('StyleGramMatrices', 
                             ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        self.style_gms = gms_tup(*[gram_matrix(t) for t in style_batch])
        logging.info(f'gram matrix for style image is precomputed')

    def on_batch_begin(self, last_target, **kwargs):
        """
        overwrite last_target into a dict before feeding into loss function
        """
        last_target['style_target'] = self.style_gms
        return {'last_target': last_target} 


class SaveCallback(LearnerCallback):
    _order = 2
    
    def __init__(self, learn, meta_model, chkpt_epoch, chkpt_model_dir):
        super().__init__(learn)
        self.chkpt_epoch = chkpt_epoch
        self.meta_model = meta_model
        self.chkpt_model_dir = chkpt_model_dir
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
            
    def on_epoch_end(self, epoch, **kwargs):
        if epoch % self.chkpt_epoch == 0:            
            chkpt_model_fname = f'chkpt_epoch_{epoch:03}.pth'
            self.meta_model.transformer.eval().cpu()
            chkpt_model_path = os.path.join(self.chkpt_model_dir, chkpt_model_fname)
            torch.save(
                self.meta_model.transformer.state_dict(), chkpt_model_path
                )
            self.meta_model.transformer.to(self.device).train()
            logging.info(f'[epoch: {epoch}] model saved: {chkpt_model_path}')
        return None

    def on_train_end(self, epoch, **kwargs):
        self.meta_model.transformer.eval().cpu()
        chkpt_model_fname = f'final_epoch_{epoch:03}.pth'
        chkpt_model_path = os.path.join(self.chkpt_model_dir, chkpt_model_fname)
        torch.save(
            self.meta_model.transformer.state_dict(), chkpt_model_path
            )
        logging.info(f'[train complete] model saved: {chkpt_model_path}')
