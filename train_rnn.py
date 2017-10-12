#!/usr/bin/env python
# coding=utf-8
# wujian@17.10.10

import argparse
import os
import numpy as np
import torch as th
import torch.nn as nn
import torch.utils.data as data

from data.seq_dataset import THCHS30
from model import RNN
from torch.autograd import Variable

# RNN config
input_size      = 40
hidden_size     = 512
hidden_layer    = 3
dropout         = 0.2

def cross_validate(epoch, nnet, test_dataset, test_loader):
    pos_frames = 0.0
    for (test_feats, test_labels) in test_loader:
        test_inputs  = Variable(test_feats.cuda())
        test_targets = Variable(test_labels.view(test_labels.size(0) * test_labels.size(1)).cuda())
        _, predict_classes = th.max(nnet(test_inputs), 1)
        pos_frames += float((predict_classes == test_targets).cpu().data.sum())
    accuracy = pos_frames / test_dataset.num_frames 
    print('epoch {}: accracy = {}'.format(epoch + 1, accuracy))

def make_dir(pdir):
    if not os.path.exists(pdir):
        os.makedirs(pdir)

def train(args):
    make_dir(args.checkout_dir)
    assert args.num_classes
    # nnet 
    nnet = RNN(input_size, hidden_layer, hidden_size, args.num_classes, dropout=dropout)
    print(nnet)
    nnet.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = th.optim.Adam(nnet.parameters(), lr=0.001)

    train_dataset = THCHS30(root=args.data_dir, data_type='train')
    train_loader  = data.DataLoader(dataset=train_dataset, batch_size=128,
                                    shuffle=True)

    test_dataset = THCHS30(root=args.data_dir, data_type='test')
    test_loader  = data.DataLoader(dataset=test_dataset, batch_size=128,
                                    shuffle=True)
    
    cross_validate(-1, nnet, test_dataset, test_loader)    
    for epoch in range(args.num_epochs):
        for index, (feats, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            labels = labels.view(labels.size(0) * labels.size(1))
            inputs  = Variable(feats.cuda())
            targets = Variable(labels.cuda())
            outputs = nnet(inputs)
            celoss  = criterion(outputs, targets)
            celoss.backward()
            optimizer.step()
        cross_validate(epoch, nnet, test_dataset, test_loader)    
        th.save(nnet, os.path.join(args.checkout_dir, 'rnn.{}.pkl'.format(epoch + 1)))
        # th.save(nnet.state_dict(), os.path.join(args.checkout_dir, 'rnn.{}.pkl'.format(epoch + 1)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Training ML-RNN on TIMIT'
    )
    parser.add_argument('-i', '--data-dir', 
        dest='data_dir', type=str,
        default='egs',
        help='directory of dataset'
    )
    parser.add_argument('-o', '--checkout-dir', 
        dest='checkout_dir', type=str,
        default='rnn',
        help='directory to save RNN model'
    )
    parser.add_argument('-e', '--num-epochs', 
        dest='num_epochs', type=int,
        default=10,
        help='num of epochs to training on the dataset'
    )
    parser.add_argument('-c', '--num-classes', 
        dest='num_classes', type=int,
        default=-1,
        help='num of pdfs to predict/classify'
    )
    args = parser.parse_args()
    print(args)
    train(args)
