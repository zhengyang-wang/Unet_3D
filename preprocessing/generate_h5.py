import nibabel as nib
import numpy as np
import os
# import matplotlib.pyplot as plt
import h5py

# copy from results of calculate_mean.py
MEAN = np.array((57.9034515187, 69.5360292781), dtype=np.float32) # 31.6906890728, 38.0572250084
SIZE = (0, 143, 9, 191, 69, 215) # (0, 143, 0, 195, 0, 255)

# arguments
SUBSTRACT_MEAN = False
AUGMENT = True
NORMALIZE = True

data_path = '../data'
target_path = '../h5_data_AMN'
'''    
h5_data_[S][A][M][N]
    S: mean substraction
    A: data augmentation
    M: keep margins
    N: normalize
'''

def flip(inputs, labels, axis):
    '''
    axis : integer. Axis in array, which entries are reversed.
    '''
    return np.flip(inputs, axis), np.flip(labels, axis)

def rotate(inputs, labels, num_of_rots, axes):
    '''
    num_of_rots : integer. Number of times the array is rotated by 90 degrees.
    axes : (2,) array_like. The array is rotated in the plane defined by the axes. Axes must be different.
    '''
    return np.rot90(inputs, num_of_rots, axes), np.rot90(labels, num_of_rots, axes)

def convert_labels(labels):
    '''
    function that converts 0:background / 10:CSF / 150:GM / 250:WM to 0/1/2/3
    '''
    D, H, W, C = labels.shape
    for d in range(D):
        for h in range(H):
            for w in range(W):
                for c in range(C):
                    if labels[d,h,w,c] == 10:
                        labels[d,h,w,c] = 1
                    elif labels[d,h,w,c] == 150:
                        labels[d,h,w,c] = 2
                    elif labels[d,h,w,c] == 250:
                        labels[d,h,w,c] = 3

def build_h5_dataset(data_path, target_path):
    '''
    Build HDF5 Image Dataset.
    '''
    new_shape = (SIZE[1]-SIZE[0]+1, SIZE[3]-SIZE[2]+1, SIZE[5]-SIZE[4]+1)
    print(new_shape)

    d_imgshape = (10, new_shape[0], new_shape[1], new_shape[2], 2)
    d_labelshape = (10, new_shape[0], new_shape[1], new_shape[2])

    d_imgshape_r1 = (10, new_shape[1], new_shape[0], new_shape[2], 2)
    d_labelshape_r1 = (10, new_shape[1], new_shape[0], new_shape[2])

    d_imgshape_r2 = (10, new_shape[2], new_shape[1], new_shape[0], 2)
    d_labelshape_r2 = (10, new_shape[2], new_shape[1], new_shape[0])

    d_imgshape_r3 = (10, new_shape[0], new_shape[2], new_shape[1], 2)
    d_labelshape_r3 = (10, new_shape[0], new_shape[2], new_shape[1])

    # data after cutting
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    dataset = h5py.File(os.path.join(target_path, "data.h5"), 'w')
    dataset.create_dataset('X', d_imgshape, dtype='f')
    dataset.create_dataset('Y', d_labelshape, dtype='i')

    if AUGMENT:
        # data after cutting, with flipping in first dim
        dataset_f1 = h5py.File(os.path.join(target_path, "data_flip1.h5"), 'w')
        dataset_f1.create_dataset('X', d_imgshape, dtype='f')
        dataset_f1.create_dataset('Y', d_labelshape, dtype='i')

        # data after cutting, with flipping in second dim
        dataset_f2 = h5py.File(os.path.join(target_path, "data_flip2.h5"), 'w')
        dataset_f2.create_dataset('X', d_imgshape, dtype='f')
        dataset_f2.create_dataset('Y', d_labelshape, dtype='i')

        # data after cutting, with flipping in third dim
        dataset_f3 = h5py.File(os.path.join(target_path, "data_flip3.h5"), 'w')
        dataset_f3.create_dataset('X', d_imgshape, dtype='f')
        dataset_f3.create_dataset('Y', d_labelshape, dtype='i')

        # data after cutting, with rotating k=1 axes=(0,1)
        dataset_r1_1 = h5py.File(os.path.join(target_path, "data_rotate1_1.h5"), 'w')
        dataset_r1_1.create_dataset('X', d_imgshape_r1, dtype='f')
        dataset_r1_1.create_dataset('Y', d_labelshape_r1, dtype='i')

        # data after cutting, with rotating k=2 axes=(0,1)
        dataset_r1_2 = h5py.File(os.path.join(target_path, "data_rotate1_2.h5"), 'w')
        dataset_r1_2.create_dataset('X', d_imgshape, dtype='f')
        dataset_r1_2.create_dataset('Y', d_labelshape, dtype='i')

        # data after cutting, with rotating k=3 axes=(0,1)
        dataset_r1_3 = h5py.File(os.path.join(target_path, "data_rotate1_3.h5"), 'w')
        dataset_r1_3.create_dataset('X', d_imgshape_r1, dtype='f')
        dataset_r1_3.create_dataset('Y', d_labelshape_r1, dtype='i')

        # data after cutting, with rotating k=1 axes=(0,2)
        dataset_r2_1 = h5py.File(os.path.join(target_path, "data_rotate2_1.h5"), 'w')
        dataset_r2_1.create_dataset('X', d_imgshape_r2, dtype='f')
        dataset_r2_1.create_dataset('Y', d_labelshape_r2, dtype='i')

        # data after cutting, with rotating k=2 axes=(0,2)
        dataset_r2_2 = h5py.File(os.path.join(target_path, "data_rotate2_2.h5"), 'w')
        dataset_r2_2.create_dataset('X', d_imgshape, dtype='f')
        dataset_r2_2.create_dataset('Y', d_labelshape, dtype='i')

        # data after cutting, with rotating k=3 axes=(0,2)
        dataset_r2_3 = h5py.File(os.path.join(target_path, "data_rotate2_3.h5"), 'w')
        dataset_r2_3.create_dataset('X', d_imgshape_r2, dtype='f')
        dataset_r2_3.create_dataset('Y', d_labelshape_r2, dtype='i')

        # data after cutting, with rotating k=1 axes=(1,2)
        dataset_r3_1 = h5py.File(os.path.join(target_path, "data_rotate3_1.h5"), 'w')
        dataset_r3_1.create_dataset('X', d_imgshape_r3, dtype='f')
        dataset_r3_1.create_dataset('Y', d_labelshape_r3, dtype='i')

        # data after cutting, with rotating k=2 axes=(1,2)
        dataset_r3_2 = h5py.File(os.path.join(target_path, "data_rotate3_2.h5"), 'w')
        dataset_r3_2.create_dataset('X', d_imgshape, dtype='f')
        dataset_r3_2.create_dataset('Y', d_labelshape, dtype='i')

        # data after cutting, with rotating k=3 axes=(1,2)
        dataset_r3_3 = h5py.File(os.path.join(target_path, "data_rotate3_3.h5"), 'w')
        dataset_r3_3.create_dataset('X', d_imgshape_r3, dtype='f')
        dataset_r3_3.create_dataset('Y', d_labelshape_r3, dtype='i')

    for i in range(10):
        subject_name = 'subject-%d-' % (i+1)
        f_T1 = os.path.join(data_path, subject_name+'T1.hdr')
        img_T1 = nib.load(f_T1)
        f_T2 = os.path.join(data_path, subject_name+'T2.hdr')
        img_T2 = nib.load(f_T2)
        f_l = os.path.join(data_path, subject_name+'label.hdr')
        img_l = nib.load(f_l)

        # combine T1 and T2 as 2 channels
        inputs_tmp_T1 = img_T1.get_data()
        inputs_tmp_T2 = img_T2.get_data()
        inputs = np.concatenate((inputs_tmp_T1,inputs_tmp_T2), axis=3)

        labels = img_l.get_data()
        convert_labels(labels)

        inputs = inputs[SIZE[0]:SIZE[1]+1,SIZE[2]:SIZE[3]+1,SIZE[4]:SIZE[5]+1,:].astype('float32')
        labels = labels[SIZE[0]:SIZE[1]+1,SIZE[2]:SIZE[3]+1,SIZE[4]:SIZE[5]+1,:].reshape(new_shape)

        # substract MEAN
        if SUBSTRACT_MEAN:
            inputs -= MEAN

        if NORMALIZE:
            inputs /= 1000.

        dataset['X'][i] = inputs
        dataset['Y'][i] = labels

        if AUGMENT:
            inputs_f1, labels_f1 = flip(inputs, labels, axis=0)
            dataset_f1['X'][i] = inputs_f1
            dataset_f1['Y'][i] = labels_f1

            inputs_f2, labels_f2 = flip(inputs, labels, axis=1)
            dataset_f2['X'][i] = inputs_f2
            dataset_f2['Y'][i] = labels_f2

            inputs_f3, labels_f3 = flip(inputs, labels, axis=2)
            dataset_f3['X'][i] = inputs_f3
            dataset_f3['Y'][i] = labels_f3

            inputs_r1_1, labels_r1_1 = rotate(inputs, labels, 1, axes=(0,1))
            dataset_r1_1['X'][i] = inputs_r1_1
            dataset_r1_1['Y'][i] = labels_r1_1

            inputs_r1_2, labels_r1_2 = rotate(inputs, labels, 2, axes=(0,1))
            dataset_r1_2['X'][i] = inputs_r1_2
            dataset_r1_2['Y'][i] = labels_r1_2

            inputs_r1_3, labels_r1_3 = rotate(inputs, labels, 3, axes=(0,1))
            dataset_r1_3['X'][i] = inputs_r1_3
            dataset_r1_3['Y'][i] = labels_r1_3

            inputs_r2_1, labels_r2_1 = rotate(inputs, labels, 1, axes=(0,2))
            dataset_r2_1['X'][i] = inputs_r2_1
            dataset_r2_1['Y'][i] = labels_r2_1

            inputs_r2_2, labels_r2_2 = rotate(inputs, labels, 2, axes=(0,2))
            dataset_r2_2['X'][i] = inputs_r2_2
            dataset_r2_2['Y'][i] = labels_r2_2

            inputs_r2_3, labels_r2_3 = rotate(inputs, labels, 3, axes=(0,2))
            dataset_r2_3['X'][i] = inputs_r2_3
            dataset_r2_3['Y'][i] = labels_r2_3

            inputs_r3_1, labels_r3_1 = rotate(inputs, labels, 1, axes=(1,2))
            dataset_r3_1['X'][i] = inputs_r3_1
            dataset_r3_1['Y'][i] = labels_r3_1

            inputs_r3_2, labels_r3_2 = rotate(inputs, labels, 2, axes=(1,2))
            dataset_r3_2['X'][i] = inputs_r3_2
            dataset_r3_2['Y'][i] = labels_r3_2

            inputs_r3_3, labels_r3_3 = rotate(inputs, labels, 3, axes=(1,2))
            dataset_r3_3['X'][i] = inputs_r3_3
            dataset_r3_3['Y'][i] = labels_r3_3

# tmp = inputs_tmp_T1[D_s:D_e+1,H_s:H_e+1,W_s:W_e+1,:]
# plt.imshow(tmp)
# plt.colorbar()
# plt.show()

if __name__ == '__main__':
    build_h5_dataset(data_path, target_path)
