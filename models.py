# As described in ResNet-paper https://arxiv.org/pdf/1512.03385.pdf
# and PyTorch https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

import torch
from torch import nn
import inspect


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
            #nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if first_maxpool else nn.MaxPool2d(kernel_size=1, stride=1),
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
                #print(modl, "init")
            elif isinstance(modl, nn.BatchNorm2d):
                nn.init.constant_(modl.weight, 1)
                nn.init.constant_(modl.bias, 0)
                #print(modl, "init")

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for modl in self.modules():
                if isinstance(modl, ResBlock) and modl.bn2.weight is not None:
                    nn.init.constant_(modl.bn2.weight, 0)  # type: ignore[arg-type]
        ##################################################
    
    def build_layer(self, block, count_layers, stride, activation, channels):
        in_channels = self.channels
        if stride > 1 or channels * block.expansion != in_channels:
            shortcut = nn.Sequential(
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
        x = torch.flatten(x, start_dim=1)  # (N,....) -> (N, E)
        return x


def resnet_x(depth, **kwargs):
    resnets ={
        18: (ResNet(ResBlock, layers=[2,2,2,2],**kwargs), 512),
        34: (ResNet(ResBlock, layers=[3,4,6,3],**kwargs), 512),
    }
    return resnets[depth]


class QProjector(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, activation="ReLU",  norm_layer=True, hidden_mlp=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_features, bias=False),
            nn.BatchNorm1d(hidden_features) if norm_layer else nn.Identity(),
            nn.ReLU() if activation == "ReLU" else nn.GELU(),
            nn.Linear(in_features=hidden_features, out_features=out_features)
        ) if hidden_mlp else nn.Sequential(nn.Linear(in_features=in_features, out_features=out_features))
        # Store values for easy recreation in method change_model(...)
        for k,v in locals().items():
            if k!='self': setattr(self, k, v)

    def forward(self, x):
        x = self.mlp(x)
        return x

    @staticmethod
    def create_new_model(cls, new_out_features):
        return QProjector(
            out_features=new_out_features,
            in_features=cls.in_features,
            hidden_features=cls.hidden_features,
            activation=cls.activation,
            norm_layer=cls.norm_layer,
            hidden_mlp=cls.hidden_mlp
        )


class ResSCECLR(nn.Module):
    def __init__(self,
                 backbone_depth,
                 in_channels=3,
                 activation="ReLU",
                 zero_init_residual=False,
                 mlp_hidden_features=1024,
                 mlp_outfeatures=128,
                 norm_mlp_layer=True,
                 hidden_mlp=True
                 ):
        super().__init__()
        resnet, mlp_in_features = resnet_x(
            backbone_depth,
            in_channels=in_channels,
            activation=activation,
            zero_init_residual=zero_init_residual,
        )
        self.mixer = resnet
        self.qprojector = QProjector(
            in_features=mlp_in_features,
            hidden_features=mlp_hidden_features,
            out_features=mlp_outfeatures,
            activation=activation,
            norm_layer=norm_mlp_layer,
            hidden_mlp=hidden_mlp
        )

    def forward(self, x):
        hidden_feats = self.mixer(x)  # (N, H)
        latent_feats = self.qprojector(hidden_feats)  # (N, Z)
        return latent_feats, hidden_feats


# Mutate model from t-SimCNE https://arxiv.org/pdf/2210.09879.pdf
def change_model(model, device, projection_dim=2, freeze_layer=None, change_layer=None):
    if change_layer == "last":
        in_features = model.qprojector.mlp[-1].weight.shape[1]
        model.qprojector.mlp[-1] = nn.Linear(in_features=in_features, out_features=projection_dim).to(device)
        nn.init.normal_(model.qprojector.mlp[-1].weight, std=1.0)
        # TODO different inits?
    elif change_layer == "mlp":
        model.qprojector = QProjector.create_new_model(model.qprojector, projection_dim).to(device)
    else:
        pass

    if freeze_layer == "all":
        model.requires_grad_(False)
    elif freeze_layer == "mixer":
        model.mixer.requires_grad_(False)
        model.qprojector.mlp.requires_grad_(True)
    elif freeze_layer == "keeplast":
        model.requires_grad_(False)
        model.qprojector.mlp[-1].requires_grad_(True)
    else:
        model.requires_grad_(True)

    return model


def build_model_from_hparams(hparams):
    valid_hparams =  list(inspect.signature(ResSCECLR).parameters.keys())
    hparams = {name: value for name, value in hparams.items() if name in valid_hparams}
    #print(hparams)
    return ResSCECLR(**hparams)


if __name__ == "__main__":
    x = torch.rand((2, 3, 28, 28))
    res = ResSCECLR(backbone_depth=34)
    print(res)
    y1, y2 = res(x)
    print(y1.shape, y2.shape)

    
        