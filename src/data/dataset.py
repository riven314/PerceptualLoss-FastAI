import os
import mimetypes

import torch
from torch.utils.data import Dataset

from src.common.os_utils import load_img

import logging
logging.basicConfig(level = logging.INFO, handlers = [logging.StreamHandler()],
                    format = "%(asctime)s — %(name)s — %(levelname)s — %(message)s")


class NeuralDataset(Dataset):

    def __init__(self, data_dir, transform):
        self.data_dir = data_dir
        self.fnames = [f.name for f in os.scandir(data_dir)]
        self.transform = transform
        logging.info(f'NeuralDataset built -- ({len(self)})')

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = load_img(os.path.join(self.data_dir, fname))
        img = self.transform(img)
        return img, {'content_target': img.detach()}
