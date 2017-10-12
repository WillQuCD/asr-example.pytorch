#!/usr/bin/env python
# coding=utf-8
# wujian@17.10.5

import os
import argparse
import numpy as np
import torch as th
import torch.nn as nn

from data.dataset import THCHS30
from data.dataset import SpliceDev
from model import DNN
from model import CNN
from torch.autograd import Variable

import data.kaldi_reader as reader
import torch.utils.data as data


# CNN config
feat_dim        = 40
left_context    = 5
right_context   = 5
filter_size     = 10
num_maps        = 128
pooling_size    = 6
conn_dim        = 512

def cross_validate(epoch, nnet, test_dataset, test_loader):
    pos_frames = 0.0
    for (test_feats, test_labels) in test_loader:
        test_inputs  = Variable(test_feats.cuda())
        test_targets = Variable(test_labels.cuda())
        _, predict_classes = th.max(nnet(test_inputs), 1)
        pos_frames += float((predict_classes == test_targets).cpu().data.sum())
    accuracy = pos_frames / test_dataset.num_frames 
    print('epoch {}: accracy = {}'.format(epoch + 1, accuracy))

def train(args):
    assert args.num_classes
    nnet = CNN(left_context + right_context + 1, feat_dim, num_maps, pooling_size,
            filter_size, conn_dim, args.num_classes)
    print(nnet)
    nnet.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = th.optim.Adam(nnet.parameters(), lr=0.001)

    train_dataset = THCHS30(root=args.data_dir, data_type='train', left_context=left_context,
            right_context=right_context, model_type='cnn')
    train_loader  = data.DataLoader(dataset=train_dataset, batch_size=128,
                                    shuffle=True, num_workers=6)

    test_dataset = THCHS30(root=args.data_dir, data_type='test', left_context=left_context,
            right_context=right_context, model_type='cnn')
    test_loader  = data.DataLoader(dataset=test_dataset, batch_size=128,
                                    shuffle=True, num_workers=6)

    cross_validate(-1, nnet, test_dataset, test_loader)    
    for epoch in range(args.num_epochs):
        for index, (feats, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs  = Variable(feats.cuda())
            targets = Variable(labels.cuda())
            outputs = nnet(inputs)
            celoss  = criterion(outputs, targets)
            celoss.backward()
            optimizer.step()
        cross_validate(epoch, nnet, test_dataset, test_loader)    
        # th.save(nnet, 'mdl/dnn.{}.mdl'.format(epoch + 1))
        th.save(nnet.state_dict(), os.path.join(args.checkout_dir, 'cnn.{}.pkl'.format(epoch + 1)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Training CNN on assigned dataset'
    )
    parser.add_argument('-i', '--data-dir', 
        dest='data_dir', type=str,
        default='egs',
        help='directory of source dataset'
    )
    parser.add_argument('-o', '--checkout-dir', 
        dest='checkout_dir', type=str,
        default='cnn',
        help='directory to save CNN model'
    )
    parser.add_argument('-e', '--epoch', 
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
