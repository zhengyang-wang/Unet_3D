# 3D Unet with VoxelDCL and DTN
## Configure the network
All network hyperparameters are configured in main.py.

#### Training

max_step: how many iterations or steps to train

test_interval: how many steps to perform a mini test or validation

save_interval: how many steps to save the model

summary_interval: how many steps to save the summary

#### Data

data_dir: data directory

train_data: h5 file for training

valid_data: h5 file for validation

test_data: h5 file for testing

batch: batch size

channel: input image channel number

height, width: height and width of input image

#### Debug

logdir: where to store log

modeldir: where to store saved models

sampledir: where to store predicted samples, please add a / at the end for convinience

model_name: the name prefix of saved models

reload_step: where to return training

test_step: which step to test or predict

random_seed: random seed for tensorflow

#### Network architecture

network_depth: how deep of the U-Net including the bottom layer

class_num: how many classes. Usually number of classes plus one for background

start_channel_num: the number of channel for the first conv layer


1. batch to 1
2. overlap_stepsize to an integer between 1-32
3. test_step