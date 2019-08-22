#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


### YOUR CODE HERE for part 1i
class CNN(nn.Module):

    def __init__(self, char_embed_size, num_filters, max_word_length, kernel_size=5):
        super().__init__()

        self.conv = nn.Conv1d(in_channels=char_embed_size,
                              out_channels=num_filters,
                              kernel_size=kernel_size,
                              bias=True)
        self.max_pool = nn.MaxPool1d(kernel_size=max_word_length - kernel_size + 1)

    def forward(self, x_embed):
        x_conv = self.conv(x_embed)
        x_conv_out = self.max_pool(F.relu(x_conv)).squeeze()
        return x_conv_out

### END YOUR CODE
