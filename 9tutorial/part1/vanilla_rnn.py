################################################################################
# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#


################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np

################################################################################

"""
seq_length: 
The length of the input sequences. In the context of RNNs, 
this is the number of time steps that the network will unroll or process. 
Each time step corresponds to one input from the sequence.

input_dim: 
The dimensionality of the input data at each time step. 
For example, if you are processing sequences of vectors where each vector has 10 elements, 
input_dim would be 10.

num_hidden: 
The number of hidden units in the RNN. This parameter defines the size of the hidden state 
of the RNN, which is a key factor in determining the capacity of the network to 
capture information from the input sequence.

num_classes: 
The number of classes for the output. This is typically used to specify the size of the 
output layer of the network, which corresponds to the number of target classes in a 
classification task.

"""
class VanillaRNN(nn.Module):

    def __init__(
        self, seq_length, input_dim, num_hidden, num_classes, batch_size, device="cpu"
    ):
        super(VanillaRNN, self).__init__()
        self.seq_length = seq_length
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.device = device
        self.batch_size = batch_size

        self.W_hx = torch.nn.Parameter(torch.randn(num_hidden, input_dim))
        self.W_hh = torch.nn.Parameter(torch.randn(num_hidden, num_hidden))
        self.W_ph = torch.nn.Parameter(torch.randn(num_classes, num_hidden))
        self.b_h = torch.nn.Parameter(torch.randn(num_hidden))
        self.b_p = torch.nn.Parameter(torch.randn(num_classes))

    def forward(self, x):
        h = torch.zeros(self.batch_size, self.num_hidden).to(self.device)
        for t in range(self.seq_length):
            x_t = x[:, t:t+1] # preserve 2 dimensions
            # Calculate the output as described in equation (1)
            h = torch.tanh(x_t @ self.W_hx.t() + h @ self.W_hh.t() + self.b_h)
        # Calculate the digit prediction for the last step
        output = h @ self.W_ph.t() + self.b_p
        return output