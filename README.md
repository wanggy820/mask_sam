# MedSAM
This is the official repository for Mask-SAM: Segment Anything base on mask attention.



## Installation
1. Create a virtual environment `conda create -n masksam python=3.10 -y` and activate it `conda activate masksam`
2. Install [Pytorch 2.0](https://pytorch.org/get-started/locally/)
3. `git clone https://github.com/wanggy820/mask_sam`
4. Enter the MaskSAM folder `cd mask_sam` and run `pip install -e .`


## Get Started
Download the [model checkpoint](https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN?usp=drive_link) and place it at e.g., `work_dir/MedSAM/medsam_vit_b`

We provide three ways to quickly test the model on your images

1. Command line


## Model Training

### Data preprocessing

Download [MaskSAM checkpoint](https://pan.baidu.com/s/1GGkBEwOf76wmz0mAEb7HEA) Extracted code: 28xn and place it at `MaskSAM/work_dir/Thyroid_tn3k_fold0/sam_best.pth` .

Download the demo [dataset](https://drive.google.com/file/d/1reHyY5eTZ5uePXMVMzFOq5j3eFOSp50F/view?usp=sharing) and unzip it to `data/FLARE22Train/`.


The model was trained on one A100. Please use the slurm script to start the training process.

### Training on one GPU

```bash
python MaskSAM/train.py
```


## Acknowledgements
- We highly appreciate all the challenge organizers and dataset owners for providing the public dataset to the community.
- We thank Meta AI for making the source code of [segment anything](https://github.com/facebookresearch/segment-anything) publicly available.
- We also thank Alexandre Bonnet for sharing this great [blog](https://encord.com/blog/learn-how-to-fine-tune-the-segment-anything-model-sam/)


## Reference

```
@article{Mask-SAM,
  title={Thyroid nodule ultrasound image segmentation based on mask attention and segment anything},
  author={Wang, Zhao}
}
```

| 模型名称 | 代码地址                                                 | 论文   |
|-------|------------------------------------------------------|------|
| SAM  | https://github.com/facebookresearch/segment-anything | https://scontent-hkg4-1.xx.fbcdn.net/v/t39.2365-6/10000000_900554171201033_1602411987825904100_n.pdf?_nc_cat=100&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=OmDpjYX2c7AAb6zDL3f&_nc_ht=scontent-hkg4-1.xx&oh=00_AfAb9mHKqMHPpgVILiETJsHihg0nd1PU7MIOw7sEvv3vCQ&oe=6622BAA7 |
|MedSAM| https://github.com/bowang-lab/MedSAM                 |https://www.nature.com/articles/s41467-024-44824-z|
|Medical-SAM-Adapter| https://github.com/KidsWithTokens/Medical-SAM-Adapter |https://arxiv.org/abs/2304.12620|
|nnet| https://github.com/milesial/Pytorch-UNet|https://arxiv.org/pdf/1505.04597v1.pdf|
|U2net|                                                      |https://arxiv.org/pdf/2005.09007.pdf|


