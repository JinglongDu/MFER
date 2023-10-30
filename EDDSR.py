import torch
from torch import nn
class DenseBlock(nn.Module):
    def __init__(self):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv3d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv3d(in_channels=48, out_channels=24, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv3d(in_channels=72, out_channels=24, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5 = nn.Conv3d(in_channels=96, out_channels=24, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x0 = self.relu(self.conv1(x))
        x1 = self.relu(self.conv2(x0))
        x_add1 = [x0,x1]
        x_add1 = torch.cat(x_add1,1)
        x2 = self.relu(self.conv3(x_add1))
        x_add2 = [x_add1,x2]
        x_add2 = torch.cat(x_add2,1)
        x3 = self.relu(self.conv4(x_add2))
        x_add3 = [x_add2,x3]
        x_add3 = torch.cat(x_add3,1)
        x4 = self.relu(self.conv5(x_add3))

        return x4


class EDDSR(nn.Module):
    def __init__(self):
        super(EDDSR, self).__init__()
        self.conv_input = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=24, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
        )
        self.Block1 = DenseBlock()
        self.Block2 = DenseBlock()
        self.Block3 = DenseBlock()
        self.Block4 = DenseBlock()
        self.deconv = nn.ConvTranspose3d(in_channels=120, out_channels=1, kernel_size=8, stride=2, padding=3, bias=False)


    def forward(self, x):
        x0 = self.conv_input(x)
        x1 = self.Block1(x0)
        x2 = self.Block2(x1)
        x3 = self.Block3(x2)
        x4 = self.Block4(x3)
        x_add = [x0,x1,x2,x3,x4]
        x_add = torch.cat(x_add, 1)
        x_out = self.deconv(x_add)
        return x_out

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