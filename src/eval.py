import os
import re

import torch
from torchvision import transforms

from src.common.os_utils import load_img, save_img
from src.data.tfms import get_test_transforms
from src.model.meta_model import MetaModel

import logging
logging.basicConfig(level = logging.INFO, handlers = [logging.StreamHandler()],
                    format = "%(asctime)s — %(name)s — %(levelname)s — %(message)s")


def process_img(img_path, tfms):
    """
    load in image and transform it into Tensors, ready for model input
    """
    assert os.path.isfile(img_path), f'img_path not exist: {img_path}'
    content_img = load_img(img_path)    
    content_img = tfms(content_img)
    # convert to (B, C, H, W)
    content_img = content_img.unsqueeze(0)
    return content_img


def load_model(model_path):
    """ loaded in trained transformer, saved model already in eval mode """
    os.path.isfile(model_path), f'model_path not exist: {model_path}'
    meta_model = MetaModel()
    state_dict = torch.load(model_path)
    # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
    for k in list(state_dict.keys()):
        if re.search(r'in\d+\.running_(mean|var)$', k):
            del state_dict[k]
    meta_model.transformer.load_state_dict(state_dict)
    logging.info('transformer loaded: {model_path}')
    return meta_model.transformer


def eval_test_dir(test_dir, model_path):
    model_dir, _ = os.path.split(model_path)
    write_dir = os.path.join(model_dir, 'test')
    os.makedirs(write_dir, exist_ok = True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transformer = load_model(model_path).to(device)
    tfms = get_test_transforms()
    fnames = os.listdir(test_dir)
    
    logging.info(f'test images to be processed: {len(fnames)}')
    for fname in fnames:
        img_path = os.path.join(test_dir, fname)
        img = process_img(img_path, tfms).to(device)
        with torch.no_grad():
            out = transformer(img)
        w_path = os.path.join(write_dir, f'style_{fname}')
        save_img(w_path, out[0])
        logging.info(f'stylised image written: {w_path}')
    return None
        

if __name__ == '__main__':
    model_path = 'chkpts/first_blood/final_epoch_002.pth'
    test_dir = 'test'
    eval_test_dir(test_dir, model_path)


    
