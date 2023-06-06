# PointCMP: Contrastive Mask Prediction for Self-supervised Learning on Point Cloud Videos (CVPR2023)

## Introduction
In this paper, we propose a contrastive mask prediction (PointCMP) framework for self-supervised learning on point cloud videos. Specifically, our PointCMP employs a two-branch structure to achieve simultaneous learning of both local and global spatio-temporal information. On top of this two-branch structure, a mutual similarity based augmentation module is developed to synthesize hard samples at the feature level. 

## Installation
The code is tested with Python 3.7.12, PyTorch 1.7.1, GCC 9.4.0, and CUDA 10.2.
Compile the CUDA layers for [PointNet++](http://arxiv.org/abs/1706.02413):
```
cd modules
python setup.py install
```

## Related Repositories  
We thank the authors of related repositories:
1. PSTNet: https://github.com/hehefan/Point-Spatio-Temporal-Convolution
2. P4Transformer: https://github.com/hehefan/P4Transformer

