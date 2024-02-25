# As described in ResNet-paper https://arxiv.org/pdf/1512.03385.pdf
# and PyTorch https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

import torch
from torch import nn


class ResBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride, activation, shortcut=None):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU() if activation == "ReLU" else nn.GELU()
        self.relu2 = nn.ReLU() if activation == "ReLU" else nn.GELU()
        self.relu3 = nn.ReLU() if activation == "ReLU" else nn.GELU()
        self.shortcut = shortcut

            
    def forward(self, x):
        Fx = self.relu1( self.bn1( self.conv_1(x) ) )
        Fx = self.relu2( self.bn2( self.conv_2(Fx) ) )
        if self.shortcut is not None:
            x = self.shortcut(x)
        y = self.relu3( Fx + x )
        return y


class ResNet(nn.Module):
    def __init__(self, block, layers, activation="ReLU", first_conv=False, first_maxpool=False, in_channels=3, zero_init_residual=False):
        super().__init__()
        self.channels = 64
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, self.channels, kernel_size=7, stride=2, padding=3, bias=False) if first_conv else \
                nn.Conv2d(in_channels, self.channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if first_maxpool else nn.MaxPool2d(kernel_size=1, stride=1, padding=0), 
            nn.BatchNorm2d(self.channels),
            nn.ReLU() if activation == "ReLU" else nn.GELU()
        )
        self.conv_2 = self.build_layer(block, layers[0], stride=1, activation=activation, channels=64) 
        self.conv_3 = self.build_layer(block, layers[1], stride=2, activation=activation, channels=128) 
        self.conv_4 = self.build_layer(block, layers[2], stride=2, activation=activation, channels=256) 
        self.conv_5 = self.build_layer(block, layers[3], stride=2, activation=activation, channels=512) 
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        # Two initialization tricks from PyTorch 
        # https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
        #########################
        for modl in self.modules():
            if isinstance(modl, nn.Conv2d):
                nn.init.kaiming_normal_(modl.weight, mode="fan_out", nonlinearity="relu" if activation=="ReLU" else "leaky_relu")
            elif isinstance(modl, nn.BatchNorm2d):
                nn.init.constant_(modl.weight, 1)
                nn.init.constant_(modl.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for modl in self.modules():
                if isinstance(modl, BasicBlock) and modl.bn2.weight is not None:
                    nn.init.constant_(modl.bn2.weight, 0)  # type: ignore[arg-type]
        ##################################################
    
    def build_layer(self, block, count_layers, stride, activation, channels):
        in_channels = self.channels
        if stride > 1 or channels * block.expansion != in_channels:
            shortcut =nn.Sequential(
                nn.Conv2d(in_channels, channels * block.expansion, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(channels * block.expansion)
            )
        else:
            shortcut = None
        convx_1 = [
            block(
                in_channels=in_channels, 
                out_channels=channels*block.expansion, 
                shortcut=shortcut, 
                stride=stride, 
                activation=activation
            )
        ]
        in_channels = channels * block.expansion
        convx_rest = [
            block(
            in_channels=in_channels, 
            out_channels=channels, 
            shortcut=None, 
            stride=1, 
            activation=activation
            ) 
        ] * (count_layers - 1) 
        convx = convx_1 + convx_rest
        convx = nn.Sequential(*convx)
        self.channels = in_channels
        return convx

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.avgpool(x)
        return x


def resnet18(in_channels=3, activation="ReLU"):
    return ResNet(
        ResBlock, 
        layers=[2,2,2,2], 
        activation=activation, 
        first_conv=False, 
        first_maxpool=False, 
        in_channels=in_channels
    )

def resnet34(in_channels=3, activation="ReLU"):
    return ResNet(
        ResBlock, 
        layers=[3,4,6,3], 
        activation=activation, 
        first_conv=False, 
        first_maxpool=False, 
        in_channels=in_channels
    )

class Projector()

class ResSCECLR(nn.Module):
    def __init__(self, backbone, in_channels=3, activation="ReLU")
        super.__init__()
        if backbone == "resnet18":
            self.mixer = resnet18(in_channels, activation)
        elif backbone == "resnet34":
            self.mixer = resnet34(in_channels, activation)
        else: raise ValueError("Invalid backbone name")
    
    
if __name__ == "__main__":
    x = torch.rand((2, 3, 28, 28))
    res = resnet34()
    print(res)
    y = res(x)
    print(y.shape)

    
        