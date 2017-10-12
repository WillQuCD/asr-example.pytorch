#!/usr/bin/env python
# coding=utf-8
# wujian@17.10.5

import os 
import torch as th
import numpy as np
import h5py as hp
import torch.utils.data as data


class THCHS30(data.Dataset):
    def __init__(self, root, data_type):
        if data_type not in ['train', 'dev', 'test']:
            raise SystemExit('data_type could only be \'train/dev/test\'')
        inputs = os.path.join(root, '{}_inputs.h5'.format(data_type))
        labels = os.path.join(root, '{}_labels.h5'.format(data_type))
        self.inputs_h5 = hp.File(inputs)
        self.labels_h5 = hp.File(labels)
        assert len(self.inputs_h5) == len(self.labels_h5)
        self.keys = []
        for key in self.labels_h5:
            self.keys.append(key)
        assert len(self.keys) == len(self.labels_h5)
        self.size = len(self.keys)
        self.num_frames = self.size * 20
        print('Load {num} sequence / {nframes} frames in {model} file'.format(
            num=self.size, nframes=self.num_frames, model=data_type)
        )

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        key = self.keys[index]
        return self.inputs_h5[key][:], self.labels_h5[key][:].astype('int64') 

def test():
    test_dataset = THCHS30(root='../egs', data_type='test')
    test_loader  = data.DataLoader(dataset=test_dataset, 
                                    batch_size=128,
                                    shuffle=True)
    train_dataset = THCHS30(root='../egs', data_type='train')
    train_loader  = data.DataLoader(dataset=train_dataset, 
                                    batch_size=128,
                                    shuffle=True)
    for feats, labels in test_loader:
        print labels

if __name__ == '__main__':
    test()

