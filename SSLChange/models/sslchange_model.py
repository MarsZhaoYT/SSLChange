import torch
import torch.nn as nn
import itertools
from util.image_pool import ImagePool
from models.base_model import BaseModel
import models.networks as networks
import SSLChange.models.sslchange_net as sslchange_net
from torch.optim import lr_scheduler
from .sslchange_net import *


class SSLChangeModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout

        return parser

    def __init__(self, opt):

        BaseModel.__init__(self, opt)

        self.opt = opt

        if not opt.no_double_loss:
            self.full_loss_names = ['sslchange', 'rec', 'total']
        else:
            self.full_loss_names = ['sslchange', 'total']
                               
        if self.isTrain:
            self.model_names = ['SSLChange']
        else:
            self.model_names = ['SSLChange']


        # ----------------- 构建转换网络 ---------------------
        self.netSSLChange = sslchange_net.ContrastiveNet(head_type=opt.contrastive_head)
        self.netSSLChange = networks.init_net(net=self.netSSLChange, init_type='kaiming')

        # -------- 3.损失函数的基础定义、学习率，优化器 --------
        self.criterionCos = nn.CosineSimilarity(dim=1)
        self.criterionRec = nn.L1Loss()
        self.init_lr = opt.lr_sslchange * opt.batch_size / 256
        
        # -------定义优化器------
        self.optimizer = torch.optim.SGD(self.netSSLChange.parameters(), self.init_lr, momentum=0.9, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=opt.n_epochs, eta_min=0)

        self.optimizers.append(self.optimizer)
   
    def set_input(self, input):
        """
        # 继承自base_model()中同名的函数 set_input()
        # 提取dataloader中的数据，将input解包为real_A和real_B以及文件路径
        Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'

        # 解包input中的real_A和real_B，即为真实训练样本
        self.batch_A = input['A' if AtoB else 'B'].to(self.device)   #  [b, 3, 256, 256]
        self.batch_B = input['B' if AtoB else 'A'].to(self.device)   #  [b, 3, 256, 256]

        # 如果方向为AtoB，则只获取A_paths（test中使用）
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def get_scheduler(self, optimizer, opt, lr_policy):
            """Return a learning rate scheduler
            Parameters:
                optimizer          -- the optimizer of the network
                args (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                                    opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
            For 'linear', we keep the same learning rate for the first <opt.niter> epochs
            and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
            For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
            See https://pytorch.org/docs/stable/optim.html for more details.
            """
            epochs = opt.n_epochs + opt.n_epochs_decay
            if lr_policy == 'linear':
                def lambda_rule(epoch):
                    lr_l = 1.0 - epoch / float(epochs + 1)
                    return lr_l
                scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
            elif lr_policy == 'step':
                step_size = epochs // 3
                # args.lr_decay_iters
                scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
            else:
                return NotImplementedError('learning rate policy [%s] is not implemented', lr_policy)
            return scheduler

    def forward(self):
        self.f_A, self.p_A, self.z_A, self.f_B, self.p_B, self.z_B = self.netSSLChange(self.batch_A, self.batch_B)
                
    def backward_SSLChange(self):
        '''
        pseudo label和 Predicted CD map进行loss计算
        '''
        if self.opt.contrastive_head == 'sslchange_head' or self.opt.contrastive_head == 'Attn_sslchange_head':
            self.loss_sslchange_global = -0.5 * (self.criterionCos(self.p_A[0], self.z_B[0]).mean() + self.criterionCos(self.p_B[0], self.z_A[0]).mean())
            self.loss_sslchange_local = -0.5 * (self.criterionCos(self.p_A[1], self.z_B[1]).mean() + self.criterionCos(self.p_B[1], self.z_A[1]).mean())
            self.loss_sslchange = (self.loss_sslchange_global + self.loss_sslchange_local).mean()

        else:
            self.loss_sslchange = -0.5 * (self.criterionCos(self.p_A, self.z_B).mean() + self.criterionCos(self.p_B, self.z_A).mean())
        
        self.loss_total = self.loss_sslchange 
        self.loss_total.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # 执行forward()中的步骤
        self.forward()  # compute fake images and reconstruction images.

        # 打开netCDNet的梯度
        self.set_requires_grad(self.netSSLChange, True)

        self.optimizer.zero_grad()    # 将优化器梯度置0
        self.backward_SSLChange()   # 计算CDNet的loss并反向传播
        self.optimizer.step()     # 更新optimizer参数
        self.scheduler.step()    # 更新optimizer学习策略
