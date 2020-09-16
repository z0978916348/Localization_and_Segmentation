from torch.nn import Sequential, Conv2d, ConvTranspose2d, ReLU, BatchNorm2d, Module, functional
import torch.nn.functional as F
import torch

# y > x
def padding(x, y):
    if x.shape == y.shape:
        return x
    else:
        s2 = y.shape[2] - x.shape[2]
        s3 = y.shape[3] - x.shape[3]
        return functional.pad(x, (0, s3, 0, s2))


# x > y
def cropping(x, y):
    if x.shape == y.shape:
        return x
    else:
        # diffY = x.size()[2] - y.size()[2]
        # diffX = x.size()[3] - y.size()[3]

        # y = F.pad(y, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # return torch.cat([x, y], dim=1)
        return x[:, :, :y.shape[2], :y.shape[3]]


def Double_Conv2d(in_channels, out_channels, padding=0):
    return Sequential(
        Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=padding),
        ReLU(inplace=True),
        
        Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=padding),
        ReLU(inplace=True),
    )


def DeConv2D(in_channels, out_channels):
    return Sequential(
        ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        ReLU(inplace=True),
    )


def Residual_Unit(in_channels, out_channels, stride=1, padding=0):
    return Sequential(
        BatchNorm2d(in_channels),
        ReLU(inplace=True),
        Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=padding),
    )


class ResidualBlock(Module):
    def __init__(self, in_channels, out_channels, f_stride=1, padding=0):
        super().__init__()
        self.ru1 = Residual_Unit(in_channels=in_channels, out_channels=out_channels, stride=f_stride, padding=padding)
        self.ru2 = Residual_Unit(in_channels=out_channels, out_channels=out_channels, padding=padding)

    def forward(self, x):
        x = self.ru1(x)
        residual = x
        x = self.ru2(x)
        x += residual
        return x

