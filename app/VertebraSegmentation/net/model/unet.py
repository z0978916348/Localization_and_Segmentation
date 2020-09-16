from torch.nn import Module, MaxPool2d, Conv2d
import torch
from .components import Double_Conv2d, DeConv2D, cropping


class Unet(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double1l = Double_Conv2d(in_channels, 64, padding=0)

        self.double2l = Double_Conv2d(64, 128, padding=0)
        self.double3l = Double_Conv2d(128, 256, padding=0)
        self.double4l = Double_Conv2d(256, 512, padding=0)
        self.doubleb = Double_Conv2d(512, 1024, padding=0)

        self.maxpooling = MaxPool2d(kernel_size=2, stride=2)

        self.up1 = DeConv2D(1024, 512)
        self.up2 = DeConv2D(512, 256)
        self.up3 = DeConv2D(256, 128)
        self.up4 = DeConv2D(128, 64)

        self.double1r = Double_Conv2d(1024, 512, padding=0)
        self.double2r = Double_Conv2d(512, 256, padding=0)
        self.double3r = Double_Conv2d(256, 128, padding=0)
        self.double4r = Double_Conv2d(128, 64, padding=0)

        self.final = Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        l1 = self.double1l(x)
        x = self.maxpooling(l1)

        l2 = self.double2l(x)
        x = self.maxpooling(l2)

        l3 = self.double3l(x)
        x = self.maxpooling(l3)

        l4 = self.double4l(x)
        x = self.maxpooling(l4)

        x = self.doubleb(x)

        x = self.up1(x)
        l4 = cropping(l4, x)
        x = torch.cat([l4, x], dim=1)
        # x = cropping(l4, x)
        x = self.double1r(x)

        x = self.up2(x)
        l3 = cropping(l3, x)
        x = torch.cat([l3, x], dim=1)
        # x = cropping(l3, x)
        x = self.double2r(x)

        x = self.up3(x)
        l2 = cropping(l2, x)
        x = torch.cat([l2, x], dim=1)
        # x = cropping(l2, x)
        x = self.double3r(x)

        x = self.up4(x)
        l1 = cropping(l1, x)
        x = torch.cat([l1, x], dim=1)
        # x = cropping(l1, x)
        x = self.double4r(x)

        x   = self.final(x)
        return x


""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits