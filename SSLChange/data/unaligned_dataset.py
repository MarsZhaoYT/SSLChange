import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torchvision.transforms as transforms
from util.gaussian_blur import GaussianBlur


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        # opt.dataroot 即存储trainA/trainB/testA/testB的上层目录
#         self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
#         self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        
        self.dir_A = os.path.join(opt.dataroot, 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, 'B')  # create a path '/path/to/data/trainB'

        # 获取目录下所有图像文件的排序路径
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'

        # 数据集size
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        btoA = self.opt.direction == 'BtoA'

        # 根据转换方向获取输入和输出通道数
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image

        # 进行transform，并根据通道数确定是否进行灰度化
        '''
        Compose(
                Resize(size=[286, 286], interpolation=PIL.Image.BICUBIC)
                RandomCrop(size=(256, 256), padding=None)
                RandomHorizontalFlip(p=0.5)
                ToTensor()
                Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                )
        '''
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

        # Augmentation for SimSiam
        self.simsiam_aug = opt.simsiam_aug
        if self.simsiam_aug:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            self.simsiam_transform = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.RandomResizedCrop(256, scale=(0.2, 1.)),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # not strengthened
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        # 按照index取出A_path
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range

        # opt.serial_batches默认为True，则A/B按照相同的index取，即为同样的文件
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        # 否则对B随机选取index
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        # 读取图像张量
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        # apply image transformation、
        # Lamda() + Normalize(0.5, 0.5, 0.5) + ToTensor()
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        # 首先根据opt.simsiam_aug参数判断是否需要进行数据扩增
        # (1) 不指定--simsiam_aug：使用上面transform_A/B
        # (2) 指定--simsiam_aug：直接对A图像进行两次变换
        if self.simsiam_aug:
            A = self.simsiam_transform(A_img)
            B = self.simsiam_transform(A_img)
        # print('-------- SimSiam Augmentation Performed! --------')

        # 共有4个返回值
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
