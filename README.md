# 3D Unet Equipped with Advanced Deep Learning Methods
Created by [Zhengyang Wang](http://www.eecs.wsu.edu/~zwang6/) at Washington State University.

This project was presented as a poster (please find it in this repository) in BioImage Informatics Conference 2017.

## Introduction
This repository includes a 3D version of Unet equipped with 2 advanced deep learning methods: VoxelDCL (derived from [PixelDCL](https://arxiv.org/abs/1705.06820)) and [Dense Transformer Networks](https://arxiv.org/abs/1705.08881).

The preprocessing code and data input interface is for our dataset introduced below. To apply this model on other 3D segmentation datasets, you only need to change preprocessing code and data_reader.py.

## Citation
If using this code, please cite our paper.
```
@article{gao2017pixel,
  title={Pixel Deconvolutional Networks},
  author={Hongyang Gao and Hao Yuan and Zhengyang Wang and Shuiwang Ji},
  journal={arXiv preprint arXiv:1705.06820},
  year={2017}
}
```
```
@article{li2017dtn,
  title={Dense Transformer Networks},
  author={Jun Li and Yongjun Chen and Lei Cai and Ian Davidson and Shuiwang
Ji},
  journal={arXiv preprint arXiv:1705.08881},
  year={2017}
}
```

## Dataset
The dataset is from UNC and currently not available to the public. Basically, it is composed of multi-modality isointense infant brain MR images (3D) of 10 subjects. Each subject has two 3D images (T1WI and T2WI) with a manually 3D segmentation label.

It is an important step in brain development study to perform automatic segmentation of infant brain magnetic resonance (MR) images into white matter (WM), grey matter (GM) and cerebrospinal fluid (CSF) regions. This task is especially challenging in the isointense stage (approximately 6-8 months of age) when WM and GM exhibit similar levels of intensities in MR images.

## System requirement
#### Programming language
Python 3.5+

#### Python Packages
tensorflow-gpu (GPU), numpy, h5py, nibabel
