import torch
import torch.nn as nn

# Channel Attention
class CALayer(nn.Module):
    def __init__(self, input_channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc1=nn.Conv3d(input_channel,input_channel//reduction,1,bias=True)
        self.relu=nn.ReLU()
        self.fc2=nn.Conv3d(input_channel//reduction,input_channel,1,bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out=self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out=self.fc2(self.relu(self.fc1(self.max_pool(x))))
        y = avg_out+max_out
        y = self.sigmoid(y)
        return x * y

# Spatial Attention
class SALayer(nn.Module):
    def __init__(self, input_channel, output_channels, reduction=1):
        super(SALayer, self).__init__()
        self.sa_conv = nn.Sequential(
            nn.Conv3d(in_channels=input_channel,
                      out_channels=output_channels,
                      kernel_size=3,
                      padding=1,
                      bias=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y=torch.cat([avg_out,max_out],dim=1)
        y = self.sa_conv(y)
        y = self.sigmoid(y)
        return x * y


#triple Mixed Convolution(MC)
class Ml_Conv(nn.Module):
    def __init__(self, input_channels, out_channels=32):
        super(Ml_Conv, self).__init__()
        # 3D standard convolution
        self.ml_conv = nn.Sequential(
            nn.Conv3d(in_channels=input_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      groups=1,
                      bias=True,
                      ),
        )
        # 3D dilated convolution
        self.ml_dilated = nn.Sequential(
            nn.Conv3d(in_channels=input_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      dilation=2,
                      stride=1,
                      padding=2,
                      groups=1,
                      bias=True,
                      ),
        )
        # 3D deconvolution
        self.ml_deconv = nn.Sequential(
            nn.ConvTranspose3d(in_channels=input_channels,
                               out_channels=32,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               groups=1,
                               bias=True,
                               ),
        )
        # SR upsampling—>Multi-level reconstruction
        self.de_conv_sr = nn.Sequential(
            nn.ConvTranspose3d(in_channels=32,
                               out_channels=1,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=True,
                               ),
        )
        self.sa = SALayer(input_channel=2, output_channels=1, reduction=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        ml_conv = self.ml_conv(x)
        ml_dilated = self.ml_dilated(x)
        ml_deconv = self.ml_deconv(x)
        ml_out = self.relu(ml_conv + ml_dilated + ml_deconv)
        ml_sr = self.de_conv_sr(ml_out)
        ml_sr = self.sa(ml_sr)
        ml_sr = self.relu(ml_sr)
        return ml_out,ml_sr


class MFER(nn.Module):
    def __init__(self):
        super(MFER, self).__init__()
        #IFE
        self.conv_input = nn.Sequential(
            nn.Conv3d(in_channels=1,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      groups=1,
                      bias=True,
                      ),
            nn.ReLU(),
        )

        #CSR
        self.cross_rl = nn.Sequential(
            nn.ConvTranspose3d(in_channels=1,
                               out_channels=1,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=True,
                               )
        )

        # MFE
        self.Ml_Conv_1 = Ml_Conv(input_channels=64, out_channels=32)
        self.Ml_Conv_2 = Ml_Conv(input_channels=32, out_channels=32)
        self.Ml_Conv_3 = Ml_Conv(input_channels=32, out_channels=32)
        self.Ml_Conv_4 = Ml_Conv(input_channels=32, out_channels=32)
        self.Ml_Conv_5 = Ml_Conv(input_channels=32, out_channels=32)
        self.Ml_Conv_6 = Ml_Conv(input_channels=32, out_channels=32)
        self.Ml_Conv_7 = Ml_Conv(input_channels=32, out_channels=32)
        self.Ml_Conv_8 = Ml_Conv(input_channels=32, out_channels=32)
        self.Ml_Conv_9 = Ml_Conv(input_channels=32, out_channels=32)
        self.Ml_Conv_10 = Ml_Conv(input_channels=32, out_channels=32)
        self.Ml_Conv_11 = Ml_Conv(input_channels=32, out_channels=32)
        self.Ml_Conv_12 = Ml_Conv(input_channels=32, out_channels=32)
        self.Ml_Conv_13 = Ml_Conv(input_channels=32, out_channels=32)
        self.Ml_Conv_14 = Ml_Conv(input_channels=32, out_channels=32)
        self.Ml_Conv_15 = Ml_Conv(input_channels=32, out_channels=32)
        self.Ml_Conv_16 = Ml_Conv(input_channels=32, out_channels=32)
        self.Ml_Conv_17 = Ml_Conv(input_channels=32, out_channels=32)
        self.Ml_Conv_18 = Ml_Conv(input_channels=32, out_channels=32)
        self.Ml_Conv_19 = Ml_Conv(input_channels=32, out_channels=32)

        #MRec
        self.ca = CALayer(input_channel=19, reduction=16)
        self.conv_out = nn.Sequential(
            nn.Conv3d(in_channels=19,
                      out_channels=1,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      groups=1,
                      bias=True,
                      )
        )


    def forward(self, x_input):

        x = self.conv_input(x_input)
        cross_rl = self.cross_rl(x_input)
        x ,up1 = self.Ml_Conv_1(x)
        x ,up2 = self.Ml_Conv_2(x)
        x ,up3 = self.Ml_Conv_3(x)
        x ,up4 = self.Ml_Conv_4(x)
        x ,up5 = self.Ml_Conv_5(x)
        x ,up6 = self.Ml_Conv_6(x)
        x ,up7 = self.Ml_Conv_7(x)
        x ,up8 = self.Ml_Conv_8(x)
        x ,up9 = self.Ml_Conv_9(x)
        x ,up10 = self.Ml_Conv_10(x)
        x ,up11 = self.Ml_Conv_11(x)
        x ,up12 = self.Ml_Conv_12(x)
        x ,up13 = self.Ml_Conv_13(x)
        x ,up14 = self.Ml_Conv_14(x)
        x ,up15 = self.Ml_Conv_15(x)
        x ,up16 = self.Ml_Conv_16(x)
        x ,up17 = self.Ml_Conv_17(x)
        x ,up18 = self.Ml_Conv_18(x)
        x ,up19 = self.Ml_Conv_19(x)

        x_up_all = [up1, up2, up3, up4, up5,
                    up6, up7, up8, up9, up10,
                    up11, up12, up13, up14, up15,
                    up16, up17, up18, up19]
        x_up = torch.cat(x_up_all, 1)

        x_ca = self.ca(x_up)
        x_sr = self.conv_out(x_ca)

        return x_sr+cross_rl

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


