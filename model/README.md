# 3D Unet with VoxelDCL and DTN

## Configure the network
All network hyperparameters are configured in main.py.

#### Training

max_step: how many iterations or steps to train

save_interval: how many steps to save the model

summary_interval: how many steps to save the summary

learning_rate: learning rate for SGD training

#### Data

data_dir: data directory

aug_flip: whether to use flip-augmented data to generate 3 extra datasets

aug_rotate: whether to use rotation-augmented data to generate 9 extra datasets

validation_id: which subject to use as validation data (1-10)

overlap_stepsize: overlap stepsize when testing or prediction

patch_size: training patch size

batch: batch size (>1 if training, =1 if testing)

#### Debug

logdir: where to store log

modeldir: where to store saved models

savedir: where to store predicted results

model_name: the name prefix of saved models

reload_step: where to return training

test_step: which step to test or predict

random_seed: random seed for tensorflow

#### Network architecture

network_depth: how deep of the U-Net including the bottom layer

class_num: how many classes. Usually number of classes plus one for background

start_channel_num: the number of channel for the first conv layer

conv_name: choose traditional convolution or ipixel convolution to do downsampling

deconv_name: choose traditional deconvolution, pixel deconvolution or ipixel deconvolution to do upsampling

action: if pixel_dcl or ipixel_dcl is used, choose the way how feature maps are combined inside the layer

#### Dense Transformer Networks

add_dtn: add DTN or not

dtn_location: where to add the DTN

control_points_ratio: the DTN input size / control_points


## Training, Testing and Evaluation

#### Start training

After configure the network, we can start to train. Run
```
python main.py
```

#### Training process visualization

We employ tensorboard to visualize the training process.
```
tensorboard --logdir=logdir/
```

#### Testing and prediction

Fill the overlap_stepsize, test_step and change batch to 1 in configure. Then run
```
python main.py --option='test'
```

If you want to make predictions, run
```
python main.py --option='predict'
```

#### Prediction evaluation and visualization

You can use eval/eval.py to evaluate the results through Dice Ratio and Modified Hausdorff Distance.

To visualize the results, use eval/show_results.py.

## Use VoxelDCL or DTN

You can copy the utils/ to your own project to use VoxelDCL or DTN. Use this project as an example.
