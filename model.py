#!/usr/bin/env python
# coding=utf-8

import math
import torch as th
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import Variable
import common

class DNN(nn.Module):
    def __init__(self, input_size, hidden_layer, hidden_size, num_classes, 
            batchnorm=True, dropout=0.0, activate='relu'):
        super(DNN, self).__init__()
        assert hidden_layer
        assert activate in ['relu', 'sigmoid']
        layer = [nn.Linear(input_size, hidden_size), ]
        # dropout => activate => batchnorm 
        for i in range(hidden_layer):
            if dropout != 0.0:
                layer.append(nn.Dropout(dropout))
            layer.append(nn.ReLU() if activate == 'relu' else nn.Sigmoid())
            if batchnorm:
                layer.append(nn.BatchNorm1d(hidden_size))
            layer.append(nn.Linear(hidden_size, num_classes) \
                if i == hidden_layer - 1 else \
                nn.Linear(hidden_size, hidden_size)    
            )
        self.dnn = nn.Sequential(*layer) 

    def forward(self, x):
        x = self.dnn(x)
        return x

class BatchNormRNN(nn.Module):
    def __init__(self, input_size, output_size, rnn_type=nn.GRU, \
            dropout=0.0, residual=False):
        super(BatchNormRNN, self).__init__()
        self.batch_norm = nn.BatchNorm1d(input_size)
        self.residual   = residual
        self.inner_rnn  = rnn_type(
            input_size=input_size,
            hidden_size=output_size,
            batch_first=True,
            dropout=dropout
        )
    
    def forward(self, x):
        if self.residual:
            residual = x
        n, t = x.size(0), x.size(1)
        x = x.contiguous().view(n * t, -1)
        # first batch_norm then rnn
        x = self.batch_norm(x)
        # restore the RNN input shape(3D)
        x = x.view(n, t, -1)
        x, _ = self.inner_rnn(x)
        if self.residual:
            x = x + residual
        return x

rnn_type_map = {
    'lstm': nn.LSTM,
    'gru':  nn.GRU,
    'rnn':  nn.RNN
}

class RNN(nn.Module):
    def __init__(self, input_size, hidden_layer, hidden_size, num_classes, 
            rnn_type='gru', dropout=0.0, residual=False):
        super(RNN, self).__init__()
        assert rnn_type in ['lstm', 'gru', 'rnn']
        inner_rnn = rnn_type_map[rnn_type]
        layer = [BatchNormRNN(input_size, hidden_size, \
                rnn_type=inner_rnn, dropout=dropout), ]
        for i in range(hidden_layer):
            layer.append(BatchNormRNN(hidden_size, hidden_size, \
                    rnn_type=inner_rnn, dropout=dropout, residual=residual))
        self.mrnn = nn.Sequential(*layer)
        self.conn = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, num_classes)
        ) 

    def forward(self, x):
        x = self.mrnn(x)
        n, t = x.size(0), x.size(1)
        x = x.contiguous().view(n * t, -1)
        x = self.conn(x)
        return x


class CNN(nn.Module):
    def __init__(self, time_step, feat_dim, num_maps,
                pooling_size, filter_size, conn_dim,
                num_classes):
        """
            This CNN only contains 1 convolution layer with 1 channel input, 
            convoluting along frequent axies and several fully connect layer 
            following. 
        """
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=num_maps,
                kernel_size=(time_step, filter_size)
            ),
            nn.ReLU(),
        )
        size_after_conv = common.get_conv_output_size(feat_dim, 0, filter_size, 1)
        self.maxpool = nn.MaxPool1d(kernel_size=pooling_size)
        size_after_pool = common.get_conv_output_size(size_after_conv, 0, pooling_size, pooling_size) 
        # implement by DNN, or using nn.Sequential as followings
        # self.conn    = nn.Sequential(
        #    nn.Linear(num_maps * size_after_pool, conn_dim),
        #    nn.Dropout(p=0.2),
        #    nn.ReLU(),
        #    nn.BatchNorm1d(conn_dim),
        #    nn.Linear(conn_dim, num_classes)
        #)
        self.conn       = DNN(num_maps * size_after_pool, 1, conn_dim, num_classes, dropout=0.2)

    def forward(self, x):
        x = x.view(x.size(0), 1, x.size(1), x.size(2))
        x = self.conv(x)
        batch_size, num_channel = x.size(0), x.size(1)
        x = x.view(batch_size, num_channel, x.size(2) * x.size(3))
        x = self.maxpool(x)
        x = x.view(x.size(0), x.size(1) * x.size(2))
        y = self.conn(x)
        return y

"""
Implement of 
    Peddinti, V., Povey, D., & Khudanpur, S. (2015). 
    A time delay neural network architecture for efficient modeling of long temporal contexts. In INTERSPEECH (pp. 3214-3218). Figure.1

batch_size  = 10
featurn_dim = 40
time_step   = 23

feature_in  = Variable(th.rand(batch_size, time_step, featurn_dim))
tdlayer1 = model.TDLinear([-2, 2], featurn_dim, 850, sub_sampling=False)
tdlayer2 = model.TDLinear([-1, 0, 2], 850, 850)
tdlayer3 = model.TDLinear([-3, 0, 3], 850, 850)
tdlayer4 = model.TDLinear([-7, 0, 2], 850, 850)
tdlayer5 = model.TDLinear([0], 850, 1300)
layer1_out  = func.relu(tdlayer1(feature_in))
layer2_out  = func.relu(tdlayer2(layer1_out))
layer3_out  = func.relu(tdlayer3(layer2_out))
layer4_out  = func.relu(tdlayer4(layer3_out))
layer5_out  = func.relu(tdlayer5(layer4_out))

or
tdnn = nn.Sequential(
    model.TDLinear([-2, 2], featurn_dim, 850, sub_sampling=False),
    nn.ReLU(),
    model.TDLinear([-1, 0, 2], 850, 850),
    nn.ReLU(),
    model.TDLinear([-3, 0, 3], 850, 850),
    nn.ReLU(),
    model.TDLinear([-7, 0, 2], 850, 850),
    nn.ReLU(),
    model.TDLinear([0], 850, 1300)
)
"""
class TDLinear(nn.Module):
    def __init__(self, context, feat_dim, output_dim, bias=True, sub_sampling=True):
        super(TDLinear, self).__init__()
        self.feat_dim   = feat_dim
        self.output_dim = output_dim
        self.context_info = context if sub_sampling else range(context[0], context[1] + 1)
        self.register_buffer('sample_context', Variable(th.LongTensor([index - min(context) for index in self.context_info])))
        self.input_dim  = self.sample_context.size(0) * self.feat_dim
        self.context_width = max(context) - min(context) + 1
        self.weight = nn.Parameter(th.Tensor(self.output_dim, self.input_dim))
        if bias:
            self.bias = nn.Parameter(th.Tensor(self.output_dim))
        else:
            self.register_buffer('bias', None)
        self.init_weight()
    
    def init_weight(self):
        """
            init parameter same as Linear layer, cause I implement TDNN using 
            Linear transform instead of 1D convolution
        """
        stdev = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdev, stdev)
        if self.bias is not None:
            self.bias.data.uniform_(-stdev, stdev)

    def forward(self, x):
        """
            x: input as batch_size x time_step x feature_dim
                note: we will shape as batch_size x N as feed of Linear transform
            return: batch x context x output_dim 
            forward:
                NOTE: needed to optimized, the way following computes all the frames which contains
                someone having no contribution to the results of current batch, it's not necessary to
                compute them in fact.
                for stride in range(output_frames):
                    # batch_size x sub_sample_context x feature_dim
                    features_sub = th.index_select(x, 1, stride + self.sample_context) 
                    # batch_size x input_dim
                    features_in  = features_sub.view(batch_size, self.input_dim)
                    # feed in Linear layer
                    features_out.append(func.linear(features_in, self.weight, self.bias))
        """
        input_size = x.size()
        assert len(input_size) == 3
        batch_size, time_step, feature_dim = input_size
        assert feature_dim == self.feat_dim
        output_frames = time_step - self.context_width + 1
        # apply linear transform
        features_out = [func.linear(th.index_select(x, 1, stride + self.sample_context).view(batch_size, self.input_dim), \
                self.weight, self.bias) for stride in range(output_frames)]
        features_out = th.cat(features_out, 0)
        return features_out.view(batch_size, output_frames, self.output_dim)

    def __repr__(self):
        return "TDLinear (feature_dim = {feature_dim}, time_index = {context}" \
                " {input_dim} -> {output_dim})".format(feature_dim=self.feat_dim,
                context=self.context_info, input_dim=self.input_dim, output_dim=self.output_dim)
