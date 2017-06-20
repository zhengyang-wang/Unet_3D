"""
Get necessary numbers for data pre-processing. Run only once and copy the outputs to the generate_h5.py
"""

import nibabel as nib
import numpy as np
import os

# arguments
margin = 16 # training_patch_size / 2
raw_data_dir = '../data'

def cut_edge(data, keep_margin):
    '''
    function that cuts zero edge
    '''
    D, H, W, C = data.shape
    D_s, D_e = 0, D-1
    H_s, H_e = 0, H-1
    W_s, W_e = 0, W-1

    while D_s < D:
        if data[D_s].sum() != 0:
            break
        D_s += 1
    while D_e > D_s:
        if data[D_e].sum() != 0:
            break
        D_e -= 1
    while H_s < H:
        if data[:,H_s].sum() != 0:
            break
        H_s += 1
    while H_e > H_s:
        if data[:,H_e].sum() != 0:
            break
        H_e -= 1
    while W_s < W:
        if data[:,:,W_s].sum() != 0:
            break
        W_s += 1
    while W_e > W_s:
        if data[:,:,W_e].sum() != 0:
            break
        W_e -= 1

    if keep_margin != 0:
        D_s = max(0, D_s - keep_margin)
        D_e = min(D-1, D_e + keep_margin)
        H_s = max(0, H_s - keep_margin)
        H_e = min(H-1, H_e + keep_margin)
        W_s = max(0, W_s - keep_margin)
        W_e = min(W-1, W_e + keep_margin)
    
    return int(D_s), int(D_e), int(H_s), int(H_e), int(W_s), int(W_e)

def get_cut_size(data_path):
    list_D_s, list_D_e, list_H_s, list_H_e, list_W_s, list_W_e = [], [], [], [], [], []

    for i in range(1, 10+1):
        subject_name = 'subject-%d-' % i
        f = os.path.join(data_path, subject_name+'T1.hdr')
        img = nib.load(f)
        inputs_tmp = img.get_data()
        D_s, D_e, H_s, H_e, W_s, W_e = cut_edge(inputs_tmp, margin)
        list_D_s.append(D_s)
        list_D_e.append(D_e)
        list_H_s.append(H_s)
        list_H_e.append(H_e)
        list_W_s.append(W_s)
        list_W_e.append(W_e)

    min_D_s, max_D_e, min_H_s, max_H_e, min_W_s, max_W_e = \
    min(list_D_s), max(list_D_e), min(list_H_s), max(list_H_e), min(list_W_s), max(list_W_e)
    
    print("new size: ", min_D_s, max_D_e, min_H_s, max_H_e, min_W_s, max_W_e)
    return min_D_s, max_D_e, min_H_s, max_H_e, min_W_s, max_W_e

def get_mean(data_path):
    mean1_per_sub = []
    mean2_per_sub = []
    cut_mean1_per_sub = []
    cut_mean2_per_sub = []

    D_s, D_e, H_s, H_e, W_s, W_e = get_cut_size(data_path)
    cut_size = (D_e-D_s+1) * (H_e-H_s+1) * (W_e-W_s+1)
    # print(cut_size)

    for i in range(1, 10+1):
        subject_name = 'subject-%d-' % i
        f_T1 = os.path.join(data_path, subject_name+'T1.hdr')
        img_T1 = nib.load(f_T1)
        f_T2 = os.path.join(data_path, subject_name+'T2.hdr')
        img_T2 = nib.load(f_T2)

        inputs_tmp_T1 = img_T1.get_data()
        inputs_tmp_T2 = img_T2.get_data()

        D, H, W, C = inputs_tmp_T1.shape
        original_size = D * H * W

        sum1 = inputs_tmp_T1.sum()
        sum2 = inputs_tmp_T2.sum()

        mean1_per_sub.append(sum1 / original_size)
        mean2_per_sub.append(sum2 / original_size)

        cut_mean1_per_sub.append(sum1 / cut_size)
        cut_mean2_per_sub.append(sum2 / cut_size)

    print(mean1_per_sub)
    print(mean2_per_sub)
    print(cut_mean1_per_sub)
    print(cut_mean2_per_sub)
    print("T1 mean: ", np.mean(mean1_per_sub))
    print("T2 mean: ", np.mean(mean2_per_sub))
    print("T1 cut mean: ", np.mean(cut_mean1_per_sub))
    print("T2 cut mean: ", np.mean(cut_mean2_per_sub))

if __name__ == '__main__':
    get_mean(raw_data_dir)