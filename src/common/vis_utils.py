import os

import numpy as np
import matplotlib.pyplot as plt


def tensor2np(t_ls):
    """
    convert list of tensor to list of numpy
    """
    outs = []
    for t in t_ls:
        tmp = t.detach().cpu().numpy()
        outs.append(tmp)
    return outs


def plot_gram_matrix_pairs(gm1, gm2, fname = None, title1 = 'Content Image Gram Matrix', title2 = 'Style Image Gram Matrix'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize= (15, 8))
    ax1.imshow(gm1[0], vmin = 0, vmax = 0.08)
    ax1.set_title(title1)
    ax2.imshow(gm2[0], vmin = 0, vmax = 0.08)
    ax2.set_title(title2)
    plt.tight_layout()
    if fname is not None:
        fig.savefig(fname)
        print(f'figure saved: {fname}')
    return None