#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


### YOUR CODE HERE for part 1h
class Highway(nn.Module):
    """
    Highway network as specified in eq. 8-10 in pdf
    with skip-connection and dynamic gate.
    """

    def __init__(self, word_embed_size: torch.Tensor):
        super().__init__()

        self.proj = nn.Linear(in_features=word_embed_size, out_features=word_embed_size, bias=True)
        self.gate = nn.Linear(in_features=word_embed_size, out_features=word_embed_size, bias=True)

    def forward(self, x_conv_out: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of highway network.
        :param x_conv_out: torch.Tensor output of convolution layer (batch_size, word_embed_size)
        :return x_highway: torch.Tensor output of highway network (batch_size, word_embed_size)
        """
        x_proj = F.relu(self.proj(x_conv_out))
        x_gate = F.sigmoid(self.gate(x_conv_out))
        x_highway = x_gate * x_proj + (1 - x_gate) * x_conv_out
        return x_highway

### END YOUR CODE
