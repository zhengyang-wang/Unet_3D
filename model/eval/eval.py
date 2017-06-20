import numpy as np
import h5py
from DiceRatio import dice_ratio
from HausdorffDistance import ModHausdorffDist

results_filename = '../result/results190000_sub10_overlap4.npy'
print(results_filename)
data_filename = '/tempspace/zwang6/infant_brain/h5_data/data.h5'
valid_sub_id = 10

def one_hot(inputs):
    '''
    convert label (d,h,w) to one-hot label (d,h,w,num_class)
    '''
    num_class = np.max(inputs) + 1
    return np.eye(num_class)[inputs]

def get_locs(inputs):
    '''
    get the location (d,h,w) where the entry is 1
    '''
    return np.stack(np.where(inputs==1)).T

param = 50

# load results
results = np.load(results_filename)
pred_labels = np.argmax(results, axis=3)
pred_labels_one_hot = one_hot(pred_labels)

# load labels
data_file = h5py.File(data_filename, 'r')
labels = np.array(data_file['Y'])
valid_labels = labels[valid_sub_id-1]
_D, _H, _W = valid_labels.shape
print('D: %d, H: %d, W: %d' % (_D, _H, _W))
valid_labels_one_hot = one_hot(valid_labels)

# separate each class
csf_preds = pred_labels_one_hot[:,:,:,1]
csf_labels = valid_labels_one_hot[:,:,:,1]

gm_preds = pred_labels_one_hot[:,:,:,2]
gm_labels = valid_labels_one_hot[:,:,:,2]

wm_preds = pred_labels_one_hot[:,:,:,3]
wm_labels = valid_labels_one_hot[:,:,:,3]

# evaluate dice ratio
print('--->CSF Dice Ratio:', dice_ratio(csf_preds, csf_labels))
print('--->GM Dice Ratio:', dice_ratio(gm_preds, gm_labels))
print('--->WM Dice Ratio:', dice_ratio(wm_preds, wm_labels))

# # build point sets for MHD
# csf_preds_locs = get_locs(csf_preds)
# csf_labels_locs = get_locs(csf_labels)

# gm_preds_locs = get_locs(gm_preds)
# gm_labels_locs = get_locs(gm_labels)

# wm_preds_locs = get_locs(wm_preds)
# wm_labels_locs = get_locs(wm_labels)

# # evaluate MHD
# print('--->CSF MHD:', ModHausdorffDist(csf_preds_locs, csf_labels_locs))
# print('--->GM MHD:', ModHausdorffDist(gm_preds_locs, gm_labels_locs))
# print('--->WM MHD:', ModHausdorffDist(wm_preds_locs, wm_labels_locs))