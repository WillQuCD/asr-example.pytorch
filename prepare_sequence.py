#!/usr/bin/env python
# coding=utf-8
# wujian@17.10.6

import argparse
import sys
import os

import h5py as hp
import data.kaldi_reader as reader

class Sequencer(object):
    def __init__(self, steps, delay, outdir, prefix):
        self.steps = steps
        self.delay = delay
        dir_prefix = os.path.join(outdir, prefix)
        self.feats_h5f = hp.File('{}_inputs.h5'.format(dir_prefix), 'w')
        self.label_h5f = hp.File('{}_labels.h5'.format(dir_prefix), 'w')
    
    def __del__(self):
        self.label_h5f.close()
        self.feats_h5f.close()

    def process_utt(self, key, feats, label):
        num_frames, feats_dim = feats.shape
        assert num_frames == label.size
        seq_num  = int((num_frames - 1 - self.delay) / self.steps)
        for i in range(seq_num):
            label_base = i * self.steps
            feats_base = label_base + self.delay
            assert label_base + self.steps < label.size
            self.write_binary(
                '{key}-{sid}'.format(key=key, sid=i), 
                feats[feats_base: feats_base + self.steps], 
                label[label_base: label_base + self.steps]
            )

    def write_binary(self, key, feats, label):
        assert feats.shape[0] == label.size
        self.feats_h5f.create_dataset(key, data=feats, 
            dtype='float32')
        self.label_h5f.create_dataset(key, data=label, 
            dtype='int32')

def main(args):
    sequencer = Sequencer(args.timesteps, args.timedelay, args.out_dir, args.prefix)
    with open(args.feats, 'rb') as feats_ark, open(args.label, 'rb') as label_ark:
        while True:
            feats_key, label_key = reader.read_key(feats_ark), \
                    reader.read_key(label_ark)
            assert feats_key == label_key
            if not feats_key:
                break
            feats_mat, label_vec = reader.read_common_mat(feats_ark), \
                    reader.read_common_int_vec(label_ark)
            sequencer.process_utt(feats_key, feats_mat, label_vec)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepare sequence data for RNN trainning and write in hdf5 format')
    parser.add_argument('feats', type=str, 
                        help="""input feats in kaldi *.ark format""")
    parser.add_argument('label', type=str,
                        help="""input label according to the feats feeded before""")
    parser.add_argument('-t', '--timesteps', 
                        dest='timesteps', type=int, default=20,
                        help="""length of utterance segment""")
    parser.add_argument('-d', '--timedelay', 
                        dest='timedelay', type=int, default=5,
                        help="""time delay of the labels""")
    parser.add_argument('-o', '--output-dir', 
                        dest='out_dir', type=str, default='',
                        help="""direction of the output file""")
    parser.add_argument('-p', '--prefix', 
                        dest='prefix', type=str, default='',
                        help="""the prefix of the output file""")
    args = parser.parse_args()
    print(args)
    main(args)

    
