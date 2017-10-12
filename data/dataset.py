#!/usr/bin/env python
# coding=utf-8
# wujian@17.10.5

import os 
import torch as th
import numpy as np
import torch.utils.data as data
import kaldi_reader as reader

class SpliceDev():
    def __init__(self, left_context, right_context):
        self.left_context  = left_context
        self.context = left_context + right_context + 1

    def splice_frames(self, feats, index):
        num_frames, feats_dim = feats.shape
        assert index >= 0 and index < num_frames
        splice_vec = np.zeros(self.context * feats_dim, dtype=np.float32)

        for t in range(self.context):
            offset = index + t - self.left_context
            if offset < 0:
                offset = 0
            if offset >= num_frames:
                offset = num_frames - 1
            splice_vec[t * feats_dim: (t + 1) * feats_dim] = feats[offset]
        return splice_vec

    def splice_feats(self, feats):
        num_frames, feats_dim = feats.shape
        splice_mat = np.zeros([num_frames, self.context * feats_dim], dtype=np.float32)
        for t in range(num_frames):
            vec = self.splice_frames(feats, t)
            splice_mat[t, :] = vec
        return splice_mat


class THCHS30(data.Dataset):
    def __init__(self, root, data_type, left_context=4, right_context=4, model_type='dnn'):
    
        if data_type not in ['train', 'dev', 'test']:
            raise SystemExit('data_type could only be \'train/dev/test\'')
        if model_type not in ['dnn', 'cnn']:
            raise SystemExit('model_type could only be \'dnn/cnn\'')
        feats  = os.path.join(root, '{}_inputs.ark'.format(data_type))
        labels = os.path.join(root, '{}_labels.ark'.format(data_type))
        self.model_type = model_type
        self.feats_dim = 0
        self.dataset = []
        self.num_frames = 0
        self.idx_bounds = [0, ]
        self.splice_dev = SpliceDev(left_context, right_context)

        # self.left_context, self.right_context = left_context, right_context
        # self.context_size = 1 + self.left_context + self.right_context
    
        with open(feats) as feats_ark, open(labels) as labels_ark:
            while True:
                feats_key, label_key = reader.read_key(feats_ark), reader.read_key(labels_ark)

                if feats_key is None and label_key is None:
                    break
                # print('feats: {} <=> labels: {}'.format(feats_key, label_key))
                assert feats_key == label_key
                feats_mat, label_vec = reader.read_common_mat(feats_ark), reader.read_common_int_vec(labels_ark)
                utt_frames, cur_feats_dim = feats_mat.shape
                
                if self.feats_dim == 0:
                    assert utt_frames
                    self.feats_dim = cur_feats_dim
                else:
                    assert self.feats_dim == cur_feats_dim 

                assert utt_frames == label_vec.shape[0]
                self.dataset.append({'feats': feats_mat, 'label': label_vec})
                self.num_frames += utt_frames
                # [0, t1, t2, ...]
                self.idx_bounds.append(self.num_frames)

        print('load {num_frames} frames from {ark}'.format(num_frames=self.num_frames, ark=feats))

    def index_to_uttid(self, index):

        for i, bound in enumerate(self.idx_bounds):
            if index < bound:
                return i - 1
        return -1

    def splice_frames(self, index):

        # splice_feats = np.zeros(self.context_size * self.feats_dim, dtype=np.float32)
        utt_id = self.index_to_uttid(index)
        assert utt_id != -1

        utt_feats, utt_label = self.dataset[utt_id]['feats'], self.dataset[utt_id]['label']
        frame_id = index - self.idx_bounds[utt_id]
        
        splice_feats = self.splice_dev.splice_frames(utt_feats, frame_id)
        
        if self.model_type == 'cnn':
            splice_feats = splice_feats.reshape(self.splice_dev.context, self.feats_dim)

        return splice_feats, utt_label[frame_id]

    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):
        return self.splice_frames(index)
                

def test_dataset():
    test_dataset = THCHS30(root='../egs', data_type='test')
    test_loader  = data.DataLoader(dataset=test_dataset, 
                                    batch_size=128,
                                    shuffle=True)
    train_dataset = THCHS30(root='../egs', data_type='train')
    train_loader  = data.DataLoader(dataset=train_dataset, 
                                    batch_size=128,
                                    shuffle=True)

    for feats, labels in test_loader:
        print feats
        print labels

def test_splice_dev():
    feats_egs = np.ones([5, 5])
    for i in range(5):
        feats_egs[i] = feats_egs[i] * i
    splice_dev = SpliceDev(2, 2)
    print splice_dev.splice_feats(feats_egs)

if __name__ == '__main__':
    test_dataset()
