import torch
import torch.nn as nn
import torchvision.models as models


class UpSampling2x(nn.Module):
    def __init__(self, in_ch):
        super(UpSampling2x, self).__init__()

        self.up2x = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2).cuda()

    def forward(self, x):
        return self.up2x(x)


class ResBlock(nn.Module):
    def __init__(self, in_ch):
        super(ResBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        ).cuda()

    def forward(self, x):
        x = x + self.block(x)
        return x


class ResUNet18_Pro(nn.Module):
    def __init__(self, expansion=False):
        super(ResUNet18_Pro, self).__init__()

        self.encoder = models.resnet18(pretrained=False, zero_init_residual=True).cuda()

        self.feat_indice = [2, 4, 5, 6, 7]
        self.feat_channels = [64, 64, 128, 256, 512]

        self.bottle = nn.Sequential(
            ResBlock(self.feat_channels[4]),
            ResBlock(self.feat_channels[4]),
            ResBlock(self.feat_channels[4]),
        )

        # ----------- f4 ------------
        self.decoder_f4 = nn.Sequential(
            UpSampling2x(self.feat_channels[4]),    # 最底层的上采样模块
            ResBlock(self.feat_channels[4]),
            nn.Conv2d(self.feat_channels[4], self.feat_channels[4] // 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True)
        ).cuda()
        self.reduction_f3 = nn.Sequential(
            nn.Conv2d(self.feat_channels[4], self.feat_channels[3], kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.feat_channels[3]),
            nn.ReLU(inplace=True)
        ).cuda()

        # ----------- f3 -------------
        self.decoder_f3 = nn.Sequential(
            UpSampling2x(self.feat_channels[3]),
            ResBlock(self.feat_channels[3]),
            nn.Conv2d(self.feat_channels[3], self.feat_channels[3] // 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True)
        ).cuda()
        self.reduction_f2 = nn.Sequential(
            nn.Conv2d(self.feat_channels[3], self.feat_channels[2], kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.feat_channels[2]),
            nn.ReLU(True)
        ).cuda()

        # ------------- f2 ------------
        self.decoder_f2 = nn.Sequential(
            UpSampling2x(self.feat_channels[2]),
            ResBlock(self.feat_channels[2]),
            nn.Conv2d(self.feat_channels[2], self.feat_channels[2] // 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True)
        ).cuda()
        self.reduction_f1 = nn.Sequential(
            nn.Conv2d(self.feat_channels[2], self.feat_channels[1], kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.feat_channels[1]),
            nn.ReLU(True)
        ).cuda()

        # ------------ f1 -------------
        self.decoder_f1 = nn.Sequential(
            UpSampling2x(self.feat_channels[1]),
            ResBlock(self.feat_channels[1])
        ).cuda()

        if expansion:
            # 如果使用扩张，则输出维度为16（为了适应CBAM的下采样ratio=4）
            self.up_conv_blk = nn.Sequential(
                UpSampling2x(self.feat_channels[0]*2),
                nn.Conv2d(self.feat_channels[0]*2, 16, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(16),
                nn.ReLU(True)
            ).cuda()
        else:
            self.up_conv_blk = nn.Sequential(
                UpSampling2x(self.feat_channels[0]*2),
                nn.Conv2d(self.feat_channels[0]*2, 3, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(3),
                nn.ReLU(True)
            ).cuda()

    def forward(self, x):

        down_feats = []     # 下采样特征集合
        for i, module in enumerate(self.encoder.children()):
            x = module(x)
            if i in self.feat_indice:
                down_feats.append((i, x))   # list中存入元组(特征索引下标，特征张量)
            if i == self.feat_indice[-1]:
                break

        bottom_feat = down_feats[-1][1]     #   取出最底层的特征[b, 512, 8, 8]

        # 将底层特征输入resblock
        bottom_feat = self.bottle(bottom_feat)

        up_f4 = self.decoder_f4(bottom_feat)    # [b, 512, 8, 8] -> [b, 512, 16, 16] ->[b, 256, 16, 16]
        concat_f3 = torch.cat((down_feats[3][1], up_f4), dim=1)    # [b, 256*2, 16, 16]
        re_f3 = self.reduction_f3(concat_f3)    # [b, 512, 16, 16] ->[b, 256, 16, 16]

        up_f2 = self.decoder_f3(re_f3)  # [256, 16, 16] -> [256, 32, 32] -> [128, 32, 32]
        concat_f2 = torch.cat((down_feats[2][1], up_f2), dim=1)    # [128*2, 32, 32]
        re_f2 = self.reduction_f2(concat_f2)    # [256, 32, 32] -> [128, 32, 32]

        up_f1 = self.decoder_f2(re_f2)  # [128, 32, 32] -> [128, 64, 64] -> [64, 64, 64]
        concat_f1 = torch.cat((down_feats[1][1], up_f1), dim=1)    # [64*2, 64, 64]
        re_f1 = self.reduction_f1(concat_f1)    # [128, 64, 64] -> [64, 64, 64]

        up_f0 = self.decoder_f1(re_f1)  # [64, 64, 64] -> [64, 128, 128]
        concat_f0 = torch.cat((down_feats[0][1], up_f0), dim=1)    # [128, 128, 128]

        out = self.up_conv_blk(concat_f0)   # [128, 128, 128] -> [128, 256, 256] -> [3, 256, 256]

        return out


if __name__ == '__main__':
    input = torch.ones(2, 3, 256, 256)
    resunet_18 = ResUNet18_Pro()

    out = resunet_18(input)
    print(out.size())

