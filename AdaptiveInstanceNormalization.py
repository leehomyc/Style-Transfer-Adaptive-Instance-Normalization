"""Implement the adaptive instance normalization layer."""
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class AdaptiveInstanceNormalization(torch.nn.Module):
    def __init__(self):
        super(AdaptiveInstanceNormalization, self).__init__()

        self.gradInput = None
        self.eps = 1e-5
        self.nOutput = 512
        self.batchSize = -1
        self.bn = None

    def forward(self, input):
        """
        Implement the AdaIN layer. The layer applies batch normalization on the content input using the standard
        deviation and mean computed using the style input.

        :param input: a PyTorch tensor of 2x3x128x128 consisting of the content and the style.
        :return: the output of ? as the batch normalized content.
        """
        content, style = input[0], input[1]

        hc, wc = content.size()[1], content.size()[2]
        hs, ws = style.size()[1], style.size()[2]

        content = content.unsqueeze(0)
        style = style.unsqueeze(0)
        style_view = style.view(1, self.nOutput, hs * ws)
        target_std = torch.std(style_view, 2, unbiased=False).view(-1)
        target_mean = torch.mean(style_view, 2, keepdim=False).view(-1)

        if self.bn is None:
            self.bn = nn.BatchNorm2d(self.nOutput, self.eps, affine=False)

        self.bn.weight = Parameter(target_std.data)
        self.bn.bias = Parameter(target_mean.data)
        content_view = content.view(1, self.nOutput, hc, wc)
        self.bn.train()
        output = self.bn(content_view).view_as(content)
        return output

    def backward(self, grad_output):
        """Not implemented."""
        self.gradInput = None
        return self.gradInput
