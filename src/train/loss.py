import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.lin_utils import gram_matrix


class PerceptualLoss(nn.Module):
    def __init__(self, model, content_weight, style_weight):
        super().__init__()
        self.model = model
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.mse = nn.MSELoss()

    def forward(self, output, target):
        feat_x, gram_style = target['content_target'], target['style_target']
        content_loss = self.content_weight * self.mse(feat_x.relu2_2, output.relu2_2)

        style_loss = 0.
        for ft_y, gm_s in zip(output, gram_style):
            # gm_y / gm_s: (BS, C_i, C_i)
            gm_y = gram_matrix(ft_y)
            # batch_n for handle uneven last batch
            batch_n = gm_y.shape[0]
            style_loss += self.mse(gm_y, gm_s[:batch_n])
        style_loss *= self.style_weight
        
        self.content_loss = content_loss.detach().item()
        self.style_loss = style_loss.detach().item()
        return content_loss + style_loss
