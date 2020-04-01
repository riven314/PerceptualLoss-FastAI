import os

import torch
import numpy as np
from PIL import Image

import logging
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()],
                    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s")


def load_img(img_path, resize = None):
    """
    both read fname image and resize it

    :param:
        img_path : str, path to the image file
        resize : tuple, (h, w) target resize
    """
    assert os.path.isfile(img_path), f'img file not exist: {img_path}'
    img = Image.open(img_path)
    if resize is not None:
        h, w = resize
        img = img.resize((w, h), Image.ANTIALIAS)
    return img


def save_img(w_path, data):
    """
    :param:
        w_path : str, path to write the image
        data : Tensors, (C, H, W)
    """
    img = data.clone().clamp(0, 255).cpu().numpy()
    img = img.transpose(1, 2, 0).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(w_path)
    logging.info(f'image written: {w_path}')
    return None
