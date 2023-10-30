import torch
import torch.nn as nn
class LSRN(nn.Module):
    def __init__(self):
        super(LSRN, self).__init__()
        self.conv_0 = nn.Conv3d(in_channels=1,out_channels=32,kernel_size=3, padding=1, bias=True)

        self.conv_1 = nn.Conv3d(in_channels=32, out_channels=48, kernel_size=3, padding=1, bias=True)
        self.conv_2 = nn.Conv3d(in_channels=48, out_channels=48, kernel_size=3, padding=1, bias=True)

        self.conv_3 = nn.Conv3d(in_channels=48, out_channels=24, kernel_size=3, padding=1, bias=True)
        self.conv_4 = nn.Conv3d(in_channels=24, out_channels=24, kernel_size=3, padding=1, bias=True)
        self.conv_5 = nn.Conv3d(in_channels=24, out_channels=24, kernel_size=3, padding=1, bias=True)


        self.conv_6 = nn.Conv3d(in_channels=24, out_channels=12, kernel_size=3, padding=1, bias=True)
        self.conv_7 = nn.Conv3d(in_channels=12, out_channels=12, kernel_size=3, padding=1, bias=True)

        self.compression = nn.Conv3d(in_channels=224, out_channels=32, kernel_size=1, bias=True)

        self.conv_out1 = nn.Conv3d(in_channels=32, out_channels=16, kernel_size=3, padding=1, bias=True)
        self.conv_out2 = nn.Conv3d(in_channels=16, out_channels=1, kernel_size=3, padding=1, bias=True)

        self.lrelu = nn.LeakyReLU(inplace=True)

    def forward(self, x_input):
        x0 = self.lrelu(self.conv_0(x_input))

        conv_1 = self.lrelu(self.conv_1(x0))
        conv_2 = self.lrelu(self.conv_2(conv_1))
        conv_3 = self.lrelu(self.conv_3(conv_2))
        conv_4 = self.lrelu(self.conv_4(conv_3))
        conv_5 = self.lrelu(self.conv_5(conv_4))
        conv_6 = self.lrelu(self.conv_6(conv_5))
        conv_7 = self.lrelu(self.conv_7(conv_6))

        x_out = [x0,conv_1,conv_2,conv_3,conv_4,conv_5,conv_6,conv_7]
        x_out = torch.cat(x_out,1)
        x_out = self.lrelu(self.compression(x_out))#!!!

        x_out = x0+x_out
        x_out = self.lrelu(self.conv_out1(x_out))
        x_out = self.conv_out2(x_out)
        return x_out

    def my_weights_init(self):
        """init the weight for a network"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                    m.weight.data,
                    a=0,
                    mode="fan_in",
                    nonlinearity="leaky_relu"
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

