"""
helper function for linear algebra computation e.g. gram matrix compute
"""
import os

import torch


def gram_matrix(y):
    """
    calculate gram matrix of a layer of feature maps 
    it measures unnormalize covariance of feature maps across channels

    :param:
        y : features map Tensors of shape (B, C, H, W), supposed pull from hook
    """
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram