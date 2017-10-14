#!/usr/bin/env python
# coding=utf-8
# wujian@17.10.14

import os
import math
import logging
import argparse

import torch as th
from torch.autograd import Variable

def make_dir(pdir):
    if not os.path.exists(pdir):
        os.makedirs(pdir)

def join_path(prefix, name):
    return os.path.join(prefix, name)

def get_conv_output_size(input_size, padding_size, kernel_size, stride):
    return int(math.floor((input_size + 2 * padding_size - (kernel_size - 1) - 1) / stride + 1))

def get_logger():
    logger      = logging.getLogger("scripts")
    logger.setLevel(logging.INFO) 
    log_fmt     = "%(filename)s[%(lineno)d] %(asctime)s %(levelname)s: %(message)s",
    date_fmt    = "%Y-%M-%d %T"
    logging.basicConfig(level=logging.INFO, format=log_fmt, datefmt=date_fmt)
    #handler = logging.StreamHandler()  
    #handler.setLevel(logging.INFO) 
    #formatter =logging.Formatter(
    #    fmt='%(filename)s[%(lineno)d] %(asctime)s %(levelname)s: %(message)s',
    #    datefmt="%Y-%M-%d %T"
    #)
    #handler.setFormatter(formatter)
    #logger.addHandler(handler)
    return logger

def cross_validate(nnet, test_loader, is_rnn=False):
    pos_frames = 0.0
    for (test_feats, test_labels) in test_loader:
        test_inputs  = Variable(test_feats.cuda())
        test_targets = Variable(test_labels.view(test_labels.size(0) * test_labels.size(1)).cuda()) \
            if is_rnn else Variable(test_labels.cuda())
        _, predict_classes = th.max(nnet(test_inputs), 1)
        pos_frames += float((predict_classes == test_targets).cpu().data.sum())
    return pos_frames

def train_one_epoch(nnet, criterion, optimizer, train_loader, is_rnn=False):
    for index, (feats, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        if is_rnn:
            labels = labels.view(labels.size(0) * labels.size(1))
        inputs  = Variable(feats.cuda())
        targets = Variable(labels.cuda())
        outputs = nnet(inputs)
        celoss  = criterion(outputs, targets)
        celoss.backward()
        optimizer.step()


def get_default_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-i', '--data-dir', 
                        dest='data_dir', type=str, default='egs',
                        help="""directory of the training dataset""")
    parser.add_argument('-o', '--checkout-dir', 
                        dest='checkout_dir', type=str, default='mdl',
                        help="""directory to save checkout of the model""")
    parser.add_argument('-e', '--num-epochs', 
                        dest='num_epochs', type=int, default=10,
                        help="""the value of the epochs to training on the dataset""")
    parser.add_argument('-c', '--num-classes', 
                        dest='num_classes', type=int, default=0,
                        help="""the total number of pdfs to predict/classify""")
    parser.add_argument('-s', '--learning-rate', 
                        dest='learning_rate', type=float, default=0.001,
                        help="""the initial learning rate to training the model""")
    parser.add_argument('-m', '--min-batch', 
                        dest='min_batch', type=int, default=128,
                        help="""size of min-batch used in SGD training""")
    parser.add_argument('-l', '--left-context', 
                        dest='left_context', type=int, default=0,
                        help="""left context[in frames] for the inputs of the neural networks""")
    parser.add_argument('-r', '--right-context', 
                        dest='right_context', type=int, default=0,
                        help="""right context[in frames] for the inputs of the neural networks""")
    parser.add_argument('-d', '--feat-dim', 
                        dest='feat_dim', type=int, default=40,
                        help="""dimention of the input features""")
    return parser
