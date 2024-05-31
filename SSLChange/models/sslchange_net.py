import torch
import torch.nn as nn
import torchvision.models as models
from models.resunet18_pro import ResUNet18_Pro
import numpy as np


# --------------- Encoder ---------------

class TwoLayerConv(nn.Module):
    def __init__(self, in_dim, middle_dim):
        super().__init__()

        self.predictor = nn.Sequential(
            nn.Conv2d(in_dim, middle_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_dim, in_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_dim)
            # nn.Flatten(),
            # nn.Linear(131072, 128)
        ).cuda()

    def forward(self, x):
        p = self.predictor(x.cuda())
        
        return p


# ------------- Projector & Predictor -------------

class SSLChangeProjector(nn.Module):
    def __init__(self, in_dim=196608, proj_dim=2048, pool=False):
        super(SSLChangeProjector, self).__init__()
        if not pool:
            # 用于对卷积输入降维
            if in_dim != proj_dim:
                self.proj = nn.Sequential( 
                    nn.Flatten(1),          # [b, 3, 256, 256] -> [b, 196608]
                    nn.Linear(in_dim, proj_dim, bias=False),      # [b, 196608] -> [b, 2048]
                    nn.BatchNorm1d(proj_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(proj_dim, proj_dim, bias=False),
                    nn.BatchNorm1d(proj_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(proj_dim, proj_dim, bias=False)
                ).cuda()

            # 用于向量输入
            elif in_dim == proj_dim:
                self.proj = nn.Sequential(
                    nn.Flatten(1),
                    nn.Linear(in_dim, proj_dim, bias=False),      # [b, 2048] -> [b, 2048]
                    nn.BatchNorm1d(proj_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(proj_dim, proj_dim, bias=False),
                    nn.BatchNorm1d(proj_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(proj_dim, proj_dim, bias=False)
                ).cuda()
        else: 
            self.proj = nn.Sequential(
                nn.AdaptiveAvgPool2d(32), 
                nn.Flatten(1),          # [b, 16, 32, 32] -> [b, 16384]
                nn.Linear(in_dim, proj_dim, bias=False),      # [b, 16384] -> [b, 2048]
                nn.BatchNorm1d(proj_dim),
                nn.ReLU(inplace=True),
                nn.Linear(proj_dim, proj_dim, bias=False),
                nn.BatchNorm1d(proj_dim),
                nn.ReLU(inplace=True),
                nn.Linear(proj_dim, proj_dim, bias=False)
            ).cuda()

    def forward(self, x):
        x = self.proj(x)
        return x


class SSLChangePredictor(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048):
        super(SSLChangePredictor, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        ).cuda()
        self.layer2 = nn.Linear(hidden_dim, out_dim, bias=False).cuda()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class ChannelAttention(nn.Module):
    '''
    通道注意力模块CBAM,ratio用于控制MLP中间层的压缩倍率
    '''

    def __init__(self, in_ch, ratio=4):
        super(ChannelAttention, self).__init__()
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_ch, in_ch // ratio, 1, bias=False).cuda()
        self.relu1 = nn.ReLU().cuda()
        self.fc2 = nn.Conv2d(in_ch // ratio, in_ch, 1, bias=False).cuda()
        self.sigmoid = nn.Sigmoid().cuda()

    def forward(self, x):
        # avg_pool -> conv(in, in/16, 1) -> relu -> conv(in/16, in, 1)
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))

        # max_pool -> conv(in, in/16, 1) -> relu -> conv(in/16, in, 1)
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out).cuda()


class SpatialAttention(nn.Module):
    """
    Tensor [b, c, h, w] -> [b, 1, h, w]
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False).cuda()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # channel维度求均值
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # channel维度求最大值
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 二者concat成2通道特征
        x = torch.cat([avg_out, max_out], dim=1)
        # conv降维并求sigmoid归一化
        x = self.conv1(x)
        return self.sigmoid(x).cuda()


# ------------- Main Network -------------------

class ContrastiveNet(nn.Module):
    '''
    Define the architecture of the prediction head after feature extraction from CDNet.
    '''
    def __init__(self, head_type='sslchange_head'):
        super(ContrastiveNet, self).__init__()
        
        # 根据参数type选择分割头网络结构
        self.type = head_type

        # 此处针对type做判断，初始化不同结构对应的网络层
        if self.type == 'sslchange_head':
            self.backbone = ResUNet18_Pro()
            self.projector_local = TwoLayerConv(in_dim=3, middle_dim=16)
            self.predictor_local = TwoLayerConv(in_dim=3, middle_dim=16)

            self.projector_global = SSLChangeProjector(in_dim=196608)
            self.predictor_global = SSLChangePredictor()
        
        elif self.type == 'sslchange_head_Spa':
            self.backbone = ResUNet18_Pro()
            self.projector = TwoLayerConv(in_dim=3, middle_dim=16)
            self.predictor = TwoLayerConv(in_dim=3, middle_dim=16)
        
        elif self.type == 'sslchange_head_Cha':
            self.backbone = ResUNet18_Pro()
            self.projector = SSLChangeProjector(in_dim=196608)
            self.predictor = SSLChangePredictor()
        
        elif self.type == 'Attn_sslchange_head':
            self.backbone = ResUNet18_Pro(expansion=True)

            self.projector_local = ChannelAttention(in_ch=16)
            self.predictor_local = SpatialAttention()

            self.projector_global = SSLChangeProjector(in_dim=16384, pool=True)
            self.predictor_global = SSLChangePredictor()

    def forward(self, fpn_A, fpn_B):      
        if self.type == 'sslchange_head':
            real_A = fpn_A  # [b, 3, 256, 256]  real_A
            fake_B = fpn_B  # [b, 3, 256, 256]  fake_B

            f_A = self.backbone(real_A)                             # [4, 3, 256, 256]
            z_A_global = self.projector_global(f_A)                 # [4, 2048]
            p_A_global = self.predictor_global(z_A_global)          # [4, 2048]
            
            z_A_local = self.projector_local(f_A)                   # [4, 3, 256, 256]
            p_A_local = self.predictor_local(z_A_local)             # [4, 3, 256, 256]


            f_B = self.backbone(fake_B)                             # [4, 3, 256, 256]
            z_B_global = self.projector_global(f_B)                 # [4, 2048]
            p_B_global = self.predictor_global(z_B_global)          # [4, 2048]
            
            z_B_local = self.projector_local(f_B)                   # [4, 3, 256, 256]
            p_B_local = self.predictor_local(z_B_local)             # [4, 3, 256, 256]

            return f_A, (z_A_global.detach(), z_A_local.detach()), (p_A_global, p_A_local), f_B, (z_B_global.detach(), z_B_local.detach()), (p_B_global, p_B_local)

        elif self.type == 'sslchange_head_Spa':
            real_A = fpn_A  # [b, 3, 256, 256]  real_A
            fake_B = fpn_B  # [b, 3, 256, 256]  fake_B

            f_A = self.backbone(real_A)           # [4, 3, 256, 256]
            z_A = self.projector(f_A)             # [4, 3, 256, 256]
            p_A = self.predictor(z_A)             # [4, 3, 256, 256]

            f_B = self.backbone(fake_B)           # [4, 3, 256, 256]
            z_B = self.projector(f_B)             # [4, 3, 256, 256]
            p_B = self.predictor(z_B)             # [4, 3, 256, 256]

            return f_A, z_A.detach(), p_A, f_B, z_B.detach(), p_B
        
        elif self.type == 'sslchange_head_Cha':
            real_A = fpn_A  # [b, 3, 256, 256]  real_A
            fake_B = fpn_B  # [b, 3, 256, 256]  fake_B

            f_A = self.backbone(real_A)           # [4, 3, 256, 256]
            z_A = self.projector(f_A)             # [4, 2048]
            p_A = self.predictor(z_A)             # [4, 2048]

            f_B = self.backbone(fake_B)           # [4, 3, 256, 256]
            z_B = self.projector(f_B)             # [4, 2048]
            p_B = self.predictor(z_B)             # [4, 2048]

            return f_A, z_A.detach(), p_A, f_B, z_B.detach(), p_B

        elif self.type == 'Attn_sslchange_head':
            real_A = fpn_A  # [b, 3, 256, 256]  real_A
            fake_B = fpn_B  # [b, 3, 256, 256]  fake_B

            f_A = self.backbone(real_A)                             # [4, 16, 256, 256]
            z_A_global = self.projector_global(f_A)                 # [4, 2048]
            p_A_global = self.predictor_global(z_A_global)          # [4, 2048] 

            z_A_local = f_A * self.projector_local(f_A)             # [4, 16, 256, 256]
            p_A_local = z_A_local * self.predictor_local(z_A_local) + f_A  # [4, 16, 256, 256]

            f_B = self.backbone(fake_B)                             # [4, 16, 256, 256]
            z_B_global = self.projector_global(f_B)                 # [4, 2048]
            p_B_global = self.predictor_global(z_B_global)          # [4, 2048]
            
            z_B_local = f_B * self.projector_local(f_B)             # [4, 16, 256, 256]
            p_B_local = z_B_local * self.predictor_local(z_B_local) + f_B  # [4, 16, 256, 256]

            return f_A, (z_A_global.detach(), z_A_local.detach()), (p_A_global, p_A_local), f_B, (z_B_global.detach(), z_B_local.detach()), (p_B_global, p_B_local)
