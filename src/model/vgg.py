from collections import namedtuple

import torch
from torchvision import models


class VGG16(torch.nn.Module):
    def __init__(self, requires_grad = False):
        super(VGG16, self).__init__()
        vgg = models.vgg16(pretrained = True).features
        self.subnet = torch.nn.Sequential()
        for i in range(23):
            self.subnet.add_module(str(i), vgg[i])
    
    def forward(self, x):
        return self.subnet(x)


class VGG16_woHook(torch.nn.Module):
    """
    it's not used for training. it's only used for unit testing

    from: https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/vgg.py
    """
    def __init__(self, requires_grad=False):
        super(VGG16_woHook, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out