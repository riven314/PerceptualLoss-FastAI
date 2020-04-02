import os

import torch

from fastai.callback import Callback

import logging
logging.basicConfig(level = logging.INFO, handlers = [logging.StreamHandler()],
                    format = "%(asctime)s — %(name)s — %(levelname)s — %(message)s")


class SaveCallback(Callback):
    _order = 1
    
    def __init__(self, learn, meta_model, chkpt_epoch, chkpt_model_dir):
        super().__init__(learn)
        self.chkpt_epoch = chkpt_epoch
        self.meta_model = meta_model
        self.chkpt_model_dir = self.chkpt_model_dir
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
        chkpt_model_fname = 'final_epoch_{epcoh:03}.pth'
        chkpt_model_path = os.path.join(self.chkpt_model_dir, chkpt_model_fname)
        torch.save(
            self.meta_model.transformer.state_dict(), chkpt_model_path
            )
        logging.info(f'[train complete] model saved: {chkpt_model_path}')