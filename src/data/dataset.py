import os
import mimetypes

import torch
from torch.utils.data import Dataset

from src.common.os_utils import load_img

import logging
logging.basicConfig(level = logging.INFO, handlers = [logging.StreamHandler()],
                    format = "%(asctime)s — %(name)s — %(levelname)s — %(message)s")


class NeuralDataset(Dataset):
    EXTENSIONS = set(k for k,v in mimetypes.types_map.items() if v.startswith('image/'))

    def __init__(self, data_dir, **kwargs):
        super(NeuralDataset, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.fnames = [f for f in os.scandir(data_dir) if f.split('.')[-1].lower() in self.EXTENSIONS]
        logging.info(f'NeuralDataset built -- ({len(self)})')

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = load_img(os.path.join(self.img_dir, fname))
        return img