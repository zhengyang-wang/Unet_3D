import numpy as np
import h5py
import random
import os

class H53DDataLoader(object):

    def __init__(self, data_dir, patch_size, valid_sub_id, overlap_stepsize, aug_flip=False, aug_rotate=False):
        data_files = []
        data_files.append(h5py.File(os.path.join(data_dir, 'data.h5'), 'r'))
        
        if aug_flip:
            data_files.append(h5py.File(os.path.join(data_dir, 'data_flip1.h5'), 'r'))
            data_files.append(h5py.File(os.path.join(data_dir, 'data_flip2.h5'), 'r'))
            data_files.append(h5py.File(os.path.join(data_dir, 'data_flip3.h5'), 'r'))
        
        self.aug_rotate = aug_rotate
        if aug_rotate:
            data_files.append(h5py.File(os.path.join(data_dir, 'data_rotate1_2.h5'), 'r'))
            data_files.append(h5py.File(os.path.join(data_dir, 'data_rotate2_2.h5'), 'r'))
            data_files.append(h5py.File(os.path.join(data_dir, 'data_rotate3_2.h5'), 'r'))
            # different shapes
            data_files.append(h5py.File(os.path.join(data_dir, 'data_rotate1_1.h5'), 'r'))
            data_files.append(h5py.File(os.path.join(data_dir, 'data_rotate1_3.h5'), 'r'))
            data_files.append(h5py.File(os.path.join(data_dir, 'data_rotate2_1.h5'), 'r'))
            data_files.append(h5py.File(os.path.join(data_dir, 'data_rotate2_3.h5'), 'r'))
            data_files.append(h5py.File(os.path.join(data_dir, 'data_rotate3_1.h5'), 'r'))
            data_files.append(h5py.File(os.path.join(data_dir, 'data_rotate3_3.h5'), 'r'))

        self.num_files = len(data_files)
        inputs = [np.array(data_files[i]['X']) for i in range(self.num_files)]
        labels = [np.array(data_files[i]['Y']) for i in range(self.num_files)]

        self.t_n, self.t_d, self.t_h, self.t_w, self.t_c = inputs[0].shape
        self.d, self.h, self.w = patch_size, patch_size, patch_size

        self.valid_sub_id = valid_sub_id-1 # leave-one-out cross validation 1-10
        mask = np.ones(self.t_n, dtype=bool)
        mask[self.valid_sub_id] = False
        self.train_inputs = [inputs[i][mask] for i in range(self.num_files)]
        self.train_labels = [labels[i][mask] for i in range(self.num_files)]
        
        self.valid_inputs, self.valid_labels = inputs[0][self.valid_sub_id], labels[0][self.valid_sub_id]
        self.prepare_validation(overlap_stepsize)
        self.num_of_valid_patches = len(self.patches_ids)
        self.valid_patch_id = 0

    def next_batch(self, batch_size):
        batches_ids = set()
        while len(batches_ids) < batch_size:
            i = random.randint(0, self.num_files-1)
            n = random.randint(0, self.t_n-2)
            d = random.randint(0, self.t_d-self.d)
            h = random.randint(0, self.t_h-self.h)
            w = random.randint(0, self.t_w-self.w)
            if ((not self.aug_rotate) or (self.aug_rotate and i < self.num_files-6)):
                batches_ids.add((i, n, d, h, w))
            elif (i >= self.num_files-6 and i < self.num_files-4):
                batches_ids.add((i, n, h, d, w))
            elif (i >= self.num_files-4 and i < self.num_files-2):
                batches_ids.add((i, n, w, h, d))
            elif (i >= self.num_files-2):
                batches_ids.add((i, n, d, w, h))

        input_batches = []
        label_batches = []
        for i, n, d, h, w in batches_ids:
            input_batches.append(self.train_inputs[i][n, d:d+self.d, h:h+self.h, w:w+self.w, :])
            label_batches.append(self.train_labels[i][n, d:d+self.d, h:h+self.h, w:w+self.w])
        inputs = np.stack(input_batches, axis=0)
        labels = np.stack(label_batches, axis=0)
        return inputs, labels

    def prepare_validation(self, overlap_stepsize):
        self.patches_ids = []
        self.drange = list(range(0, self.t_d-self.d+1, overlap_stepsize))
        self.hrange = list(range(0, self.t_h-self.h+1, overlap_stepsize))
        self.wrange = list(range(0, self.t_w-self.w+1, overlap_stepsize))
        if (self.t_d-self.d) % overlap_stepsize != 0:
            self.drange.append(self.t_d-self.d)
        if (self.t_h-self.h) % overlap_stepsize != 0:
            self.hrange.append(self.t_h-self.h)
        if (self.t_w-self.w) % overlap_stepsize != 0:
            self.wrange.append(self.t_w-self.w)
        for d in self.drange:
            for h in self.hrange:
                for w in self.wrange:
                    self.patches_ids.append((d, h, w))
        
    def reset(self):
        self.valid_patch_id = 0

    def valid_next_batch(self):
        input_batches = []
        label_batches = []
        # self.num_of_valid_patches = len(self.patches_ids)
        d, h, w = self.patches_ids[self.valid_patch_id]
        input_batches.append(self.valid_inputs[d:d+self.d, h:h+self.h, w:w+self.w, :])
        label_batches.append(self.valid_labels[d:d+self.d, h:h+self.h, w:w+self.w])
        inputs = np.stack(input_batches, axis=0)
        labels = np.stack(label_batches, axis=0)
        self.valid_patch_id += 1
        if self.valid_patch_id == self.num_of_valid_patches:
            self.reset()
        return inputs, labels, (d, h, w)
