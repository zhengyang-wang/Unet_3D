import numpy as np
import h5py
from DiceRatio import dice_ratio
from HausdorffDistance import ModHausdorffDist

results_filename = '../result/results215000_sub10_overlap4.npy'
print(results_filename)
data_filename = '/tempspace/zwang6/infant_brain/h5_data_SA/data.h5'
valid_sub_id = 10

def one_hot(inputs):
    '''
    convert label (d,h,w) to one-hot label (d,h,w,num_class)
    '''
    num_class = np.max(inputs) + 1
    return np.eye(num_class)[inputs]

def MHD_3D(preds, labels):
    D, H, W = labels.shape
    preds_d = np.array([preds[:, i, j] for i in range(H) for j in range(W)])
    preds_h = np.array([preds[i, :, j] for i in range(D) for j in range(W)])
    preds_w = np.array([preds[i, j, :] for i in range(D) for j in range(H)])
    labels_d = np.array([labels[:, i, j] for i in range(H) for j in range(W)])
    labels_h = np.array([labels[i, :, j] for i in range(D) for j in range(W)])
    labels_w = np.array([labels[i, j, :] for i in range(D) for j in range(H)])

    MHD_d = ModHausdorffDist(preds_d, labels_d)[0]
    MHD_h = ModHausdorffDist(preds_h, labels_h)[0]
    MHD_w = ModHausdorffDist(preds_w, labels_w)[0]
    ret = np.mean([MHD_d, MHD_h, MHD_w])

    print('--->MHD d:', MHD_d)
    print('--->MHD h:', MHD_h)
    print('--->MHD w:', MHD_w)
    print('--->avg:', ret)

    return ret

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
csf_dr = dice_ratio(csf_preds, csf_labels)
gm_dr = dice_ratio(gm_preds, gm_labels)
wm_dr = dice_ratio(wm_preds, wm_labels)
print('--->CSF Dice Ratio:', csf_dr)
print('--->GM Dice Ratio:', gm_dr)
print('--->WM Dice Ratio:', wm_dr)
print('--->avg:', np.mean([csf_dr, gm_dr, wm_dr]))

# evaluate MHD
csf_mhd = MHD_3D(csf_preds, csf_labels)
gm_mhd = MHD_3D(gm_preds, gm_labels)
wm_mhd = MHD_3D(wm_preds, wm_labels)
print('--->CSF MHD:', csf_mhd)
print('--->GM MHD:', gm_mhd)
print('--->WM MHD:', wm_mhd)
print('--->avg:', np.mean([csf_mhd, gm_mhd, wm_mhd]))
