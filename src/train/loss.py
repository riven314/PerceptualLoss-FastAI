import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralStyleLoss(nn.Module):
    # view is used to expand dimension
    IMAGENET_MEAN = torch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    IMAGENET_STD = torch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

    def __init__(self, model, content_weight, style_weight):
        super().__init__()
        self.vgg = model.vgg
        self.mse = nn.MSELoss()

    def normalize_batch(self, batch):
        batch = batch.div_(255.)
        return (batch - self.IMAGENET_MEAN) / self.IMAGENET_STD

    def gram_matrix(self, x):
        """
        calculate gram matrix of a layer of feature maps 
        it measures unnormalize covariance of feature maps across channels

        :param:
            x : features map Tensors of shape (B, C, H, W), supposed pull from hook
        """
        (b, c, h, w) = x.size()
        features = x.view(b, c, w * h) 
        features_t = features.transpose(1, 2)
        return features.bmm(features_t) / (c * h * w)

    def forward(self, output, target):
        
        return self.id_loss+self.gen_loss+self.cyc_loss