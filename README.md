# SSLChange

![Author](https://img.shields.io/badge/Author-MarsZYT-orange.svg)

This is a PyTorch implementation of the paper [SSLChange: A Self-supervised Change Detection Framework Based on Domain Adaptation](https://arxiv.org/abs/2405.18224)

<br>
If you find our project useful in you own research, please cite our paper below.

```
@Article{zhao2024sslchange,
      title={SSLChange: A Self-supervised Change Detection Framework Based on Domain Adaptation}, 
      author={Yitao Zhao and Turgay Celik and Nanqing Liu and Feng Gao and Heng-Chao Li},
      year={2024},
      journal={arXiv:2405.18224},
}
```

* Our manuscript has been submitted to IEEE TGRS and is under review. 

## Architecture Overview
The overview of our proposed SSLChange pre-training framework for Remote Sensing Change Decetion tasks.  
<p align="center">
      <img src="https://github.com/MarsZhaoYT/SSLChange/blob/main/imgs/SSLChange.jpg", width=800>
</p>

## Catalog
- [x] Visualization demo
- [x] Dependencies
- [ ] Domain Adapter Training
- [ ] SSLChange Pre-training
- [ ] Downstream Finetuning

## Visualization demo


## Dependencies
* Linux (Recommended) or Windows
* Python 3.8+
* Pytorch 1.8.0 or higher
* CUDA 10.1 or higher


## Code Usage
### `1. Domain Adapter Training`
* A **Domain Adapter** needs to be trained to serve as an auto-augmenter in the subsequent SSLChange Pre-training.
* The training target of Domain Adapter is to project the T1 samples into T2 domain style without change image content.
* The architecture could be **ANY** Image-to-Image Translation Algorithms. 
<p align="center">
      <img src="https://github.com/MarsZhaoYT/SSLChange/blob/main/imgs/DA.jpg", width=250>
</p>

* Here we take [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) with stable performance as a example to train the Domain Adapter. 

**Step 1. Dataset Preparation for DA Training:** Only the training set of CDD dataset is used for DA training, and the no label images are involved in the training. 
```
CDD
├── /train/
│  ├── /A/
│  │  ├── 00000.jpg
│  │  └── 00001.jpg
│  │  └── ......
│  ├── /B/
│  │  ├── 00000.jpg
│  │  └── 00001.jpg
│  │  └── ......
```
**Step 2. Train the Domain Adapter.** (train.py file in [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix))

```shell
python train.py --dataroot datasets/CDD/train/ --name your_project 
```

**Step 3. SSLChange Pre-training Dataset Generation.** (test.py file in [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix))
```shell
python test.py --dataroot datasets/CDD/train/ --name your_project --model cycle_gan --direction AtoB
```

**⭐️Some generated samples:**
  
_Original T1 images:_
<p align="center">
      <img src="https://github.com/MarsZhaoYT/SSLChange/blob/main/imgs/00000_real_A.png", width=128>
      <img src="https://github.com/MarsZhaoYT/SSLChange/blob/main/imgs/00002_real_A.png", width=128>
      <img src="https://github.com/MarsZhaoYT/SSLChange/blob/main/imgs/00011_real_A.png", width=128>
      <img src="https://github.com/MarsZhaoYT/SSLChange/blob/main/imgs/00022_real_A.png", width=128>
</p>     

_Generated Pseudo T2 images:_
<p align="center">
      <img src="https://github.com/MarsZhaoYT/SSLChange/blob/main/imgs/00000_fake_B.png", width=128>
      <img src="https://github.com/MarsZhaoYT/SSLChange/blob/main/imgs/00002_fake_B.png", width=128>
      <img src="https://github.com/MarsZhaoYT/SSLChange/blob/main/imgs/00011_fake_B.png", width=128>
      <img src="https://github.com/MarsZhaoYT/SSLChange/blob/main/imgs/00022_fake_B.png", width=128>
</p>

### `2. SSLChange Pre-training`
### `2.1 Dataset for Pre-training`

```
GenCDD
├── /train/
│  ├── /A/
│  │  ├── 00000.jpg
│  │  └── 00001.jpg
│  │  └── ......
│  ├── /B/
│  │  ├── 00000.jpg
│  │  └── 00001.jpg
│  │  └── ......
```

### `3. Downstream Finetuning`

```
CDD
├── /train/
│  ├── /A/
│  │  ├── 00000.jpg
│  │  └── 00001.jpg
│  │  └── ......
│  ├── /B/
│  │  ├── 00000.jpg
│  │  └── 00001.jpg
│  │  └── ......
│  ├── /OUT/
│  │  ├── 00000.jpg
│  │  └── 00001.jpg
│  │  └── ......
```

Thanks for your attention on our work. The codes will be published after our paper is accepted. 

