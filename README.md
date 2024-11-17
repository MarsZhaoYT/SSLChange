# SSLChange

![Author](https://img.shields.io/badge/Author-MarsZYT-orange.svg)

This is a PyTorch implementation of the paper [SSLChange: A Self-supervised Change Detection Framework Based on Domain Adaptation](https://arxiv.org/abs/2405.18224)

<p align="center">
      <img src="https://github.com/MarsZhaoYT/SSLChange/blob/main/imgs/paradigm.jpg", width=650>
</p>

<br>

## 🖊 Citation
If you find our project useful in you own research, please consider cite our paper below.

```
@ARTICLE{zhap2024sslchange,
  author={Zhao, Yitao and Celik, Turgay and Liu, Nanqing and Gao, Feng and Li, Heng-Chao},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={SSLChange: A Self-Supervised Change Detection Framework Based on Domain Adaptation}, 
  year={2024},
  volume={62},
  number={},
  pages={1-14},
  doi={10.1109/TGRS.2024.3489615}}
```

* 📩 11/1/2024 Our manuscript has been accepted by IEEE TGRS. 

## 🌏 Architecture Overview
The overview of our proposed SSLChange pre-training framework for Remote Sensing Change Decetion tasks.  
<p align="center">
      <img src="https://github.com/MarsZhaoYT/SSLChange/blob/main/imgs/SSLChange.jpg", width=800>
</p>

## 📗 Catalog
- [x] Visualization Demo
- [x] Dependencies
- [x] Domain Adapter Training
- [x] SSLChange Pre-training
- [x] Downstream Finetuning

## 🎨 Visualization Demo
The visualization results of baselines w/o and w/ SSLChange on CDD-series dataset.

<p align="center">
      <img src="https://github.com/MarsZhaoYT/SSLChange/blob/main/imgs/visual_cdd.png", width=1000>
</p>

## 💼 Dependencies
* Linux (Recommended) or Windows
* Python 3.8+
* Pytorch 1.8.0 or higher
* CUDA 10.1 or higher


## 🕹 Code Usage
### `1. Domain Adapter Training`
* A **Domain Adapter** needs to be trained to serve as an auto-augmenter in the subsequent SSLChange Pre-training.
* The training target of Domain Adapter is to project the T1 samples into T2 domain style without change image content.
* The architecture could be **ANY** Image-to-Image Translation Algorithms. 
<p align="center">
      <img src="https://github.com/MarsZhaoYT/SSLChange/blob/main/imgs/DA.jpg", width=250>
</p>

* Here we take [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) with stable performance as an example to train the Domain Adapter. 

**📂 Step 1. Dataset Preparation for DA Training.** <br>
Only the training set of CDD dataset is used for DA training, and no label images are involved in the training.
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
**🔥 Step 2. Train the Domain Adapter.** (train.py file in [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix))

```shell
python train.py --dataroot datasets/CDD/train/ --name YOUR_PROJECT 
```

**🎞 Step 3. SSLChange Pre-training Dataset Generation.** (test.py file in [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix))
```shell
python test.py --dataroot datasets/CDD/train/ --name YOUR_PROJECT --model cycle_gan --direction AtoB
```

   **⭐️Some generated samples of GenCDD dataset:**
  
_Original T1 images:_
<p align="center">
      <img src="https://github.com/MarsZhaoYT/SSLChange/blob/main/imgs/00000_real_A.png", width=128>
      <img src="https://github.com/MarsZhaoYT/SSLChange/blob/main/imgs/00002_real_A.png", width=128>
      <img src="https://github.com/MarsZhaoYT/SSLChange/blob/main/imgs/00011_real_A.png", width=128>
      <img src="https://github.com/MarsZhaoYT/SSLChange/blob/main/imgs/00022_real_A.png", width=128>
</p>     

_Generated Pseudo T2 images in GenCDD dataset:_
<p align="center">
      <img src="https://github.com/MarsZhaoYT/SSLChange/blob/main/imgs/00000_fake_B.png", width=128>
      <img src="https://github.com/MarsZhaoYT/SSLChange/blob/main/imgs/00002_fake_B.png", width=128>
      <img src="https://github.com/MarsZhaoYT/SSLChange/blob/main/imgs/00011_fake_B.png", width=128>
      <img src="https://github.com/MarsZhaoYT/SSLChange/blob/main/imgs/00022_fake_B.png", width=128>
</p>

____
### `2. SSLChange Pre-training`
Perform the SSLChange Pre-training with the Generated GenCDD dataset.

**📂 Step 1. Dataset Preparation for SSLChange Pre-training.** <br>
Only the training set of GenCDD dataset is used for SSLChange Pre-training.
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

**🔥 Step 2. Label-free Pre-training of SSLChange Framework.** <br>
Only the training set of GenCDD dataset is used for SSLChange Pre-training.

```shell
cd SSLChange
python train.py --dataroot ./datasets/GenCCD/train --name YOUR_PROJECT --model sslchange --gpu_ids 0 --simsiam_aug \
                --batch_size 8 --contrastive_head sslchange_head 
```

We release our pre-trained SSLChange weights on GenCDD dataset in [Google Drive](https://drive.google.com/file/d/1CuZMjxFB51JTCpWe2wFXDrKyvqVytf4i/view?usp=drive_link), and [BaiduYunPan](https://pan.baidu.com/s/1WErgErI6V9WeSMQ6XBnx9A) (code: scpt).


____
### `3. Downstream Finetuning`

<p align="center">
      <img src="https://github.com/MarsZhaoYT/SSLChange/blob/main/imgs/ft.png", width=500>
</p>

**📂 Step 1. Dataset Preparation for SSLChange Pre-training.** <br>
The whole portion of CDD dataset is used for downstream supervised finetuning.

```
CDD
├── /train/
│  ├── /A/
│  ├── /B/
│  ├── /OUT/
├── /test/
│  ├── /A/
│  ├── /B/
│  ├── /OUT/
├── /val/
│  ├── /A/
│  ├── /B/
│  ├── /OUT/
```

**🎮  Step 2. Pre-trained Weight Transferring.** <br>
Create a new dir to store the pre-trained SSLChange weights file.
```shell
cd Transfer-Model
mkdir pretrained_models
mkdir pretrained_models/PRETRAINED_PROJECT
cp -r ../SSLChange/checkpoint/YOUR_PROJECT/ ../Transfer-Model/pretrained_models/PRETRAINED_PROJECT/
```

**🔥  Step 3. Downstream Finetuning.** <br>
Take the finetuning for [SNUNet-CD](https://github.com/likyoo/Siam-NestedUNet) as an example.

```shell
python main_finetune.py --dataset_dir datasets/CDD --name YOUR_FTINETUNE_PROJECT \
                        --pretrained_model PRETRAINED_PROJECT/latest_net_SimSiam.pth \
                        --gpu_ids 0 --head_type sslchange_head --classifier_name SNUNet --batch_size 4
```

**✔  Step 4. Testing.** <br>

```shell
python eval.py --dataset_dir datasets/CDD --name YOUR_FTINETUNE_PROJECT --classifier_name SNUNet --gpu_ids 0
```


## 💡 Acknowledgement

We are grateful to those who kindly share their codes, which we referenced in our implementation. 

* [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
* [SNUNet-CD](https://github.com/likyoo/Siam-NestedUNet)
* [SimSiam](https://github.com/facebookresearch/simsiam)
* [CDD Dataset](https://paperswithcode.com/dataset/cdd-dataset-season-varying)
* [LEVIR-CD Dataset](https://paperswithcode.com/sota/change-detection-on-levir-cd)
