import torch
import torch.nn as nn

#CA
class CA(nn.Module):
    def __init__(self, in_channels,ratio = 16):
        super(CA, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=in_channels // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=in_channels // ratio, out_features=in_channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        z = self.excitation(y).view(b, c, 1, 1, 1)
        return x * z.expand_as(x)

#RCAU
class RCAU(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(RCAU, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels,out_channels=out_channels,kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv3d(in_channels=in_channels,out_channels=out_channels,kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv3d(in_channels=3*in_channels,out_channels=out_channels,kernel_size=1, bias=True)
        self.ca = CA(in_channels=in_channels)


    def forward(self, x):
        x1 = self.conv1(self.relu(x))
        x2 = self.conv2(self.relu(x1))
        x_cffm = torch.cat([x,x1,x2],1)
        x3 = self.conv3(x_cffm)
        x_ca = self.ca(x3)
        out = x+x_ca
        return out


# RB
class RB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RB, self).__init__()
        self.RCAU_1 = RCAU(in_channels, out_channels)
        self.RCAU_2 = RCAU(in_channels, out_channels)
        self.RCAU_3 = RCAU(in_channels, out_channels)
        self.RCAU_4 = RCAU(in_channels, out_channels)
        self.RCAU_5 = RCAU(in_channels, out_channels)
    def forward(self, x):
        rcau1 = self.RCAU_1(x)+x
        rcau2 = self.RCAU_2(rcau1)+x
        rcau3 = self.RCAU_3(rcau2)+x
        rcau4 = self.RCAU_4(rcau3)+x
        rcau5 = self.RCAU_5(rcau4)+x
        return rcau5

class DRRCAN(nn.Module):
    def __init__(self):
        super(DRRCAN, self).__init__()
        self.conv_0 = nn.Conv3d(in_channels=1,out_channels=32,kernel_size=3, padding=1, bias=True)

        self.RB_1 = RB(in_channels=32, out_channels=32)
        self.RB_2 = RB(in_channels=32, out_channels=32)

        self.conv_out = nn.Conv3d(in_channels=32, out_channels=1, kernel_size=3, padding=1, bias=True)

    def forward(self, x_input):
        x = self.conv_0(x_input)
        x = self.RB_1(x)
        x = self.RB_2(x)
        x = self.conv_out(x)
        x +=x_input #x=x+x+input
        return x

    def my_weights_init(self):
        """init the weight for a network"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                    m.weight.data,
                    a=0,
                    mode="fan_in",
                    nonlinearity="relu"
                )
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.normal_(
                    m.weight.data,
                    mean=0,
                    std=0.001
                )
                if m.bias is not None:
                    m.bias.data.zero_()

