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

################################################################################

class LSTM(nn.Module):

    def __init__(
        self, seq_length, input_dim, num_hidden, num_classes, batch_size, device="cpu"
    ):
        super(LSTM, self).__init__()
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.device = device
        self.batch_size = batch_size

        # Input modulation gate
        self.W_gx = torch.nn.Parameter(torch.randn(num_hidden, input_dim))
        self.W_gh = torch.nn.Parameter(torch.randn(num_hidden, num_hidden))
        self.b_g = torch.nn.Parameter(torch.randn(num_hidden))
        # Input gate
        self.W_ix = torch.nn.Parameter(torch.randn(num_hidden, input_dim))
        self.W_ih = torch.nn.Parameter(torch.randn(num_hidden, num_hidden))
        self.b_i = torch.nn.Parameter(torch.randn(num_hidden))
        # Forget gate
        self.W_fx = torch.nn.Parameter(torch.randn(num_hidden, input_dim))
        self.W_fh = torch.nn.Parameter(torch.randn(num_hidden, num_hidden))
        self.b_f = torch.nn.Parameter(torch.randn(num_hidden))
        # Output gate
        self.W_ox = torch.nn.Parameter(torch.randn(num_hidden, input_dim))
        self.W_oh = torch.nn.Parameter(torch.randn(num_hidden, num_hidden))
        self.b_o = torch.nn.Parameter(torch.randn(num_hidden))
        # Linear 
        self.W_ph = torch.nn.Parameter(torch.randn(num_classes, num_hidden))
        self.b_p = torch.nn.Parameter(torch.randn(num_classes))

    def forward(self, x):
        c = torch.zeros(self.batch_size, self.num_hidden).to(self.device) # Cell state
        h = torch.zeros(self.batch_size, self.num_hidden).to(self.device)
         
        for t in range(self.seq_length):
            x_t = x[:, t:t+1] # preserve 2 dimensions

            # Input modulation gate
            g = torch.tanh(x_t @ self.W_gx.t() + h @ self.W_gh.t() + self.b_g)
            # Input gate
            i = torch.sigmoid(x_t @ self.W_ix.t() + h @ self.W_ih.t() + self.b_i)
            # Forget gate
            f = torch.sigmoid(x_t @ self.W_fx.t() + h @ self.W_fh.t() + self.b_f)
            # Output gate
            o = torch.sigmoid(x_t @ self.W_ox.t() + h @ self.W_oh.t() + self.b_o)
            # Calculate next cell state
            c = g * i + c * f
            # Calculate next hidden state
            h = (torch.tanh(c) * o).to(self.device)
            # Calculate output
            p = torch.einsum('ch,bh->bc',self.W_ph, h) + self.b_p
        
        output = p

        return output