#!/usr/bin/env python
# coding=utf-8
# wujian@17.10.5

import argparse
import numpy as np
import torch as th
import torch.nn as nn

from data.dataset import THCHS30
from model import CNN
from torch.autograd import Variable

import torch.utils.data as data
import common

logger = common.get_logger()

# CNN config
filter_size     = 8
num_maps        = 256
pooling_size    = 6
conn_dim        = 512

def cross_validate(epoch, nnet, test_loader, tot_frames):
    pos_frames = common.cross_validate(nnet, test_loader)
    logger.info('epoch {:2}: accracy = {:.4f}'.format(epoch + 1, pos_frames / tot_frames))

def train(args):
    assert args.num_classes
    common.make_dir(args.checkout_dir)
    nnet = CNN(args.left_context + args.right_context + 1, args.feat_dim, num_maps, pooling_size,
            filter_size, conn_dim, args.num_classes)
    print(nnet)
    nnet.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = th.optim.Adam(nnet.parameters(), lr=args.learning_rate)

    train_dataset = THCHS30(root=args.data_dir, data_type='train', left_context=left_context,
            right_context=right_context, model_type='cnn')
    train_loader  = data.DataLoader(dataset=train_dataset, batch_size=args.min_batch,
                                    shuffle=True, num_workers=6)

    test_dataset = THCHS30(root=args.data_dir, data_type='test', left_context=left_context,
            right_context=right_context, model_type='cnn')
    test_loader  = data.DataLoader(dataset=test_dataset, batch_size=args.min_batch,
                                    shuffle=True, num_workers=6)

    cross_validate(-1, nnet, test_dataset, test_loader) 
    for epoch in range(args.num_epochs):
        common.train_one_epoch(nnet, criterion, optimizer, train_loader)
        cross_validate(epoch, nnet, test_dataset, test_loader) 
        th.save(nnet, common..join_path(args.checkout_dir, 'cnn.{}.pkl'.format(epoch + 1)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""Trains a simple CNN acoustic model using CE loss function""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler='resolve',
        parents=[common.get_default_parser()])
    args = parser.parse_args()
    print(args)
    train(args)
