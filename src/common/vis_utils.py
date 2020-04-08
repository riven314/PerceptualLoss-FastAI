import os

import numpy as np
import matplotlib.pyplot as plt



def plot_all_gram_matrices(content_gms, style_gms, iteration, write_dir):
    """
    gms is namedtuple, key = Tensors (B, C_i, C_i)
    """
    for layer_name, content_gm in content_gms._asdict().items():
        style_gm = getattr(style_gms, layer_name)
        content_gm, style_gm = tensor2np(content_gm), tensor2np(style_gm)
        content_title = f'[{layer_name}] Content Image Gram Matrix'
        style_title = f'[{layer_name}] Style Image Gram Matrix'
        
        fname = f'{layer_name}_iter{iteration:07}.png'
        w_path = os.path.join(write_dir, fname)
        plot_gram_matrix_pair(content_gm, style_gm, 
                              content_title, style_title, 
                              fname = w_path)
    plt.close('all')
    return None


def plot_gram_matrix_pair(content_gm, style_gm, title1, title2, fname = None):
    """
    both gm1, gm2 are np.array (1, C_i, C_i)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize= (15, 8))
    ax1.imshow(content_gm[0], vmin = 0, vmax = 0.08)
    ax1.set_title(title1)
    ax2.imshow(style_gm[0], vmin = 0, vmax = 0.08)
    ax2.set_title(title2)
    plt.tight_layout()
    if fname is not None:
        fig.savefig(fname)
        print(f'figure saved: {fname}')
    return None


def plot_image_pair(stylised_t, orig_t, iteration, write_dir):
    """
    both stylised_t, orig_t are Tensors (C, H, W)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize= (15, 8))
    stylised_img = tensor2uint8(stylised_t)
    orig_img = tensor2uint8(orig_t)
    ax1.imshow(orig_img)
    ax2.imshow(stylised_img)
    ax1.axis('off')
    ax2.axis('off')
    plt.tight_layout()
    
    fname = f'stylised_iter{iteration:07}.png'
    w_path = os.path.join(write_dir, fname)
    fig.savefig(w_path)
    print(f'stylised image written: {w_path}')
    plt.close('all')
    return None

        
def tensor2np(t):
    return t.detach().cpu().numpy()
        

def tensor2uint8(t):
    np_arr = t.detach().clamp(0, 255).cpu().numpy() 
    return np_arr.transpose(1, 2, 0).astype(np.uint8)