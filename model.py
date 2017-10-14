#!/usr/bin/env python
# coding=utf-8

import math
import torch as th
import torch.nn as nn


class DNN(nn.Module):
    def __init__(self, input_size, hidden_layer, hidden_size, num_classes, 
            batchnorm=True, dropout=0.0, activate='relu'):
        super(DNN, self).__init__()
        assert hidden_layer
        assert activate in ['relu', 'sigmoid']
        layer = [nn.Linear(input_size, hidden_size), ]
        # dropout => activate => batchnorm 
        for i in range(hidden_layer + 1):
            if dropout != 0.0:
                layer.append(nn.Dropout(dropout))
            layer.append(nn.ReLU() if activate == 'relu' else nn.Sigmoid())
            if batchnorm:
                layer.append(nn.BatchNorm1d(hidden_size))
            layer.append(nn.Linear(hidden_size, num_classes) \
                if i == hidden_layer else \
                nn.Linear(hidden_size, hidden_size)    
            )
        self.dnn = nn.Sequential(*layer) 

    def forward(self, x):
        x = self.dnn(x)
        return x

class BatchNormRNN(nn.Module):
    def __init__(self, input_size, output_size, rnn_type=nn.GRU, dropout=0.0):
        super(BatchNormRNN, self).__init__()
        self.batch_norm = nn.BatchNorm1d(input_size)
        self.inner_rnn  = rnn_type(
            input_size=input_size,
            hidden_size=output_size,
            batch_first=True,
            dropout=dropout
        )
    
    def forward(self, x):
        # print 'i BN: {}'.format(x.shape)
        n, t = x.size(0), x.size(1)
        x = x.contiguous().view(n * t, -1)
        # first batch_norm then rnn
        x = self.batch_norm(x)
        x = x.view(n, t, -1)
        x, _ = self.inner_rnn(x)
        # print 'o BN: {}'.format(x.shape)
        return x

rnn_type_map = {
    'lstm': nn.LSTM,
    'gru':  nn.GRU,
    'rnn':  nn.RNN
}

class RNN(nn.Module):
    def __init__(self, input_size, hidden_layer, hidden_size, num_classes, 
            rnn_type='gru', dropout=0.0):
        super(RNN, self).__init__()
        assert rnn_type in ['lstm', 'gru', 'rnn']
        inner_rnn = rnn_type_map[rnn_type]
        layer = [BatchNormRNN(input_size, hidden_size, rnn_type=inner_rnn, dropout=dropout), ]
        for i in range(hidden_layer):
            layer.append(BatchNormRNN(hidden_size, hidden_size, rnn_type=inner_rnn, dropout=dropout))
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

def get_output_size(input_size, padding_size, kernel_size, stride):
    return int(math.floor((input_size + 2 * padding_size - (kernel_size - 1) - 1) / stride + 1))

class CNN(nn.Module):
    def __init__(self, time_step, feat_dim, num_maps,
                pooling_size, filter_size, conn_dim,
                num_classes):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=num_maps,
                kernel_size=(time_step, filter_size)
            ),
            nn.ReLU(),
        )
        size_after_conv = get_output_size(feat_dim, 0, filter_size, 1)
        self.maxpool = nn.MaxPool1d(kernel_size=pooling_size)
        size_after_pool = get_output_size(size_after_conv, 0, pooling_size, pooling_size) 
        self.conn    = nn.Sequential(
            nn.Linear(num_maps * size_after_pool, conn_dim),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.BatchNorm1d(conn_dim),
            nn.Linear(conn_dim, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), 1, x.size(1), x.size(2))
        x = self.conv(x)
        batch_size, num_channel = x.size(0), x.size(1)
        x = x.view(batch_size, num_channel, x.size(2) * x.size(3))
        x = self.maxpool(x)
        x = x.view(x.size(0), x.size(1) * x.size(2))
        y = self.conn(x)
        return y

