# SSLChange
Pytorch codes for SSLChange Framework

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

## 1. Domain Adapter Training
First, a **Domain Adapter** requires to be trained which will serve as an auto-augmenter in the subsequent SSLChange Pre-training. 

<br>
Considering the natural existence of two temporal domains in RSCD tasks, the Domain Adapter is utilized to project the T1 samples into T2 domain without change the image content. 
<p align="center">
      <img src="https://github.com/MarsZhaoYT/SSLChange/blob/main/imgs/DA.jpg", width=280>
</p>

### Dataset for DA
```
CDD
├── /train/
│  ├── /A/
│  │  ├── 00000.jpg
│  │  └── 00001.jpg
│  │  └── 00002.jpg
│  │  └── ......
│  ├── /B/
│  │  ├── 00000.jpg
│  │  └── 00001.jpg
│  │  └── 00002.jpg
│  │  └── ......
│  ├── /OUT/
│  │  ├── 00000.jpg
│  │  └── 00001.jpg
│  │  └── 00002.jpg
│  │  └── ......
```
## 2. SSLChange Pre-training
### Dataset for Pre-training

## 3. Downstream Finetuning

Thanks for your attention on our work. The codes will be published after our paper is accepted. 

