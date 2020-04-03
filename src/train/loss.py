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
        """
        output = vgg(normalize(transformer(x)))
        target = x + style z

        :param:
            output: namedtuple, feature maps output from MetaModel -- {
                        relu1_2, relu2_2, relu3_3, relu4_3
                    }
                    ** each: (BS, C_i, W, H)

            target: dict, original input of MetaModel {
                    'style_target': nametuple, gram matrix for style image -- {
                            relu1_2, relu2_2, relu3_3, rel4_3
                        }
                    ** each: (BS, C_i, C_i)
                    'content_target': original model input
                    }
        
        """
        x, gram_style = target['content_target'], target['style_target']
        feat_x = self.model(x, vgg_only = True)

        content_loss = self.mse(feat_x.relu2_2, output.relu2_2)
        content_loss *= self.content_weight

        style_loss = 0.
        for ft_y, gm_s in zip(output, gram_style):
            # gm_y / gm_s: (BS, C_i, C_i)
            gm_y = gram_matrix(ft_y)
            style_loss += self.mse(gm_y, gm_s)
        style_loss *= self.style_weight
        return content_loss + style_loss
