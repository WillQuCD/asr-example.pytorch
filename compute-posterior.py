#!/usr/bin/env python
# coding=utf-8
# wujian@17.10.8

import sys
import argparse
import numpy as np
import torch as th
import torch.nn as nn

from model import DNN
from torch.autograd import Variable

import data.kaldi_reader as reader


def compute_posteriors(args, src_fd, dst_fd):
    # nnet.load_state_dict(th.load(args.model))
    nnet = th.load(args.model)
    nnet.eval()
    nnet.cpu()
    context = args.left_context + args.right_context + 1
    feats_dim = args.feats_dim

    # CNN: input spliced, and shape as 2D
    # RNN: input no splice(as a utterance), shape as batch, timesteps, feats_dim
    # DNN: input spliced
    while True: 
        key = reader.read_key(src_fd)
        if not key:
            break
        mat = reader.read_common_mat(src_fd)
        time_steps, splice_size = mat.shape
        assert splice_size == context * feats_dim 

        if args.model_type == 'rnn':
            mat = mat.reshape(1, time_steps, splice_size)
        if args.model_type == 'cnn':
            mat = mat.reshape(time_steps, context, feats_dim)
        feats_in = Variable(th.from_numpy(mat))
        posterior = nnet(feats_in)
        reader.write_key(dst_fd, key)
        reader.write_common_mat(dst_fd, posterior.data.numpy())


if __name__ == '__main__':
    sys.stderr.write(' '.join(sys.argv) + "\n")
    parser = argparse.ArgumentParser(
        description="Compute posteriors of the pytorch Neural Network"
    )
    parser.add_argument("model", help="path of the pytorch model") 
    parser.add_argument("feats", help="path of the feeding features") 
    parser.add_argument("posts", help="path of the output posteriors") 
    parser.add_argument('-d', '--feats-dim', 
        dest='feats_dim', type=int,
        default=40,
        help='dimention of the input features'
    )
    parser.add_argument('-l', '--left-context', 
        dest='left_context', type=int,
        default=0,
        help='left context of the neural networt'
    )
    parser.add_argument('-r', '--right-context', 
        dest='right_context', type=int,
        default=0,
        help='right context of the neural networt'
    )
    parser.add_argument('-t', '--model-type', 
        dest='model_type', type=str,
        default='dnn',
        help='type of the neural network[dnn/cnn/rnn]'
    )
    args = parser.parse_args()
    assert args.model_type in ['dnn', 'cnn', 'rnn']

    src_fd = open(args.feats, 'rb') if args.feats != '-' else sys.stdin
    dst_fd = open(args.posts, 'wb') if args.posts != '-' else sys.stdout
    compute_posteriors(args, src_fd, dst_fd)
    dst_fd.close()
    src_fd.close()
