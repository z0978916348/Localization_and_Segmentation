from torch.nn import Module, Conv2d, ConvTranspose2d
from torch import cat
from .components import Residual_Unit, ResidualBlock, cropping


class ResUnet(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        self.resunit1l = Residual_Unit(64, 64, padding=1)
        self.resblock2l = ResidualBlock(64, 128, f_stride=2, padding=1)
        self.resblock3l = ResidualBlock(128, 256, f_stride=2, padding=1)
        self.resbridge = ResidualBlock(256, 512, f_stride=2, padding=1)

        self.up3 = ConvTranspose2d(512, 256, kernel_size=3, stride=2)
        self.up2 = ConvTranspose2d(256, 128, kernel_size=3, stride=2)
        self.up1 = ConvTranspose2d(128, 64, kernel_size=3, stride=2)

        self.resblock3r = ResidualBlock(512, 256, padding=1)
        self.resblock2r = ResidualBlock(256, 128, padding=1)
        self.resblock1r = ResidualBlock(128, 64, padding=1)

        self.final = Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.resunit1l(x)
        l1 = x

        x = self.resblock2l(x)
        l2 = x

        x = self.resblock3l(x)
        l3 = x

        x = self.resbridge(x)

        x = self.up3(x)
        x = cropping(x, l3)
        x = cat([l3, x], dim=1)
        x = self.resblock3r(x)

        x = self.up2(x)
        x = cropping(x, l2)
        x = cat([l2, x], dim=1)
        x = self.resblock2r(x)

        x = self.up1(x)
        x = cropping(x, l1)
        x = cat([l1, x], dim=1)
        x = self.resblock1r(x)

        x = self.final(x)
        return x

""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .resunet_parts import *


class ResidualUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(ResidualUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = ResidualDoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.down5 = Down(512, 512)
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(768, 256, bilinear)
        self.up3 = Up(384, 128, bilinear)
        self.up4 = Up(192, 64, bilinear)
        self.up5 = Up(96, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        logits = self.outc(x)
        
        return logits
