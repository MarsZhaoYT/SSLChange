import os
import torch.utils.data as data
from PIL import Image
from utils import transforms as tr


'''
Load all training and validation data paths
'''
def full_path_loader(data_dir):
    # 循环获取训练数据集目录，路径为 CDD/train/A/
    train_data = [i for i in os.listdir(data_dir + '/train/A/') if not
    i.startswith('.')]
    train_data.sort()   # 对读取到的路径进行排序

    # 循环获取验证数据集目录，路径为 CDD/val/A/
    valid_data = [i for i in os.listdir(data_dir + '/val/A/') if not
    i.startswith('.')]
    valid_data.sort()   # 对读取到的路径进行排序

    train_label_paths = []
    val_label_paths = []
    # 加载训练和验证label影像的路径
    for img in train_data:
        train_label_paths.append(data_dir + '/train/OUT/' + img)
    for img in valid_data:
        val_label_paths.append(data_dir + '/val/OUT/' + img)


    train_data_path = []
    val_data_path = []
    # 加载训练和验证数据的路径，其中上级目录和文件名分布存储，格式：['dataset/CDD/train', '00000.jpg']
    for img in train_data:
        train_data_path.append([data_dir + '/train/', img])
    for img in valid_data:
        val_data_path.append([data_dir + '/val/', img])

    # 创立字典，将训练集和验证集路径配对存入字典
    # 格式：
    # {{'image': ['dataset/CDD/train', '00000.jpg'],
    #   'label': ['dataset/CDD/train', '00000.jpg']}}
    train_dataset = {}
    val_dataset = {}
    for cp in range(len(train_data)):
        train_dataset[cp] = {'image': train_data_path[cp],
                         'label': train_label_paths[cp]}
    for cp in range(len(valid_data)):
        val_dataset[cp] = {'image': val_data_path[cp],
                         'label': val_label_paths[cp]}


    return train_dataset, val_dataset


'''
Load all testing data paths
'''
def full_test_loader(data_dir):
    # 循环获取测试数据集的目录，路径为 CDD/test/A/
    test_data = [i for i in os.listdir(data_dir + '/test/A/') if not
                    i.startswith('.')]
    test_data.sort()    # 对读取到的路径进行排序

    # 加载 test_data目录下的label影像路径
    test_label_paths = []
    for img in test_data:
        test_label_paths.append(data_dir + '/test/OUT/' + img)

    # 加载 test_data目录下的img影像路径
    test_data_path = []
    for img in test_data:
        test_data_path.append([data_dir + '/test/', img])

    # 创建字典，将测试集的影像和label路径配对存入字典
    test_dataset = {}
    for cp in range(len(test_data)):
        test_dataset[cp] = {'image': test_data_path[cp],
                           'label': test_label_paths[cp]}

    return test_dataset

'''
用于加载CDD数据集的加载器。
参数:
    img_path: (str) 单张影像数据路径
    label_path: (str) 单张label数据路径
    aug: (bool) 标记进行数据扩增的类型(train_transform/test_transform)
'''
def cdd_loader(img_path, label_path, aug):
    # 获取指定数据集目录和文件名
    dir = img_path[0]
    name = img_path[1]

    # 指定T1和T2数据集中影像文件的路径，读取并存储进字典
    img1 = Image.open(dir + 'A/' + name).resize([256, 256])
    img2 = Image.open(dir + 'B/' + name).resize([256, 256])
    label = Image.open(label_path).resize([256, 256])
    sample = {'image': (img1, img2), 'label': label}

    if aug:
        # 对训练集进行train_transforms
        sample = tr.train_transforms(sample)
    else:
        # 对测试集进行test_transforms
        sample = tr.test_transforms(sample)

    # dataloader返回三张影像
    return sample['image'][0], sample['image'][1], sample['label']


class CDDloader(data.Dataset):

    def __init__(self, full_load, aug=False):

        self.full_load = full_load  # 初始化存储路径的数据集
        self.loader = cdd_loader    # 初始化cdd_loader
        self.aug = aug

    def __getitem__(self, index):

        img_path, label_path = self.full_load[index]['image'], self.full_load[index]['label']

        return self.loader(img_path,
                           label_path,
                           self.aug)

    def __len__(self):
        return len(self.full_load)
