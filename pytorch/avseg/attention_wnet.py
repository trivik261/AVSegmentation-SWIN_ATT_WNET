import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)

class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, filters=[32, 64, 128, 256, 512]):
        super(AttentionUNet, self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.Conv1 = ConvBlock(in_channels, filters[0])
        self.Conv2 = ConvBlock(filters[0], filters[1])
        self.Conv3 = ConvBlock(filters[1], filters[2])
        self.Conv4 = ConvBlock(filters[2], filters[3])
        self.Conv5 = ConvBlock(filters[3], filters[4])

        self.Up4 = UpConv(filters[4], filters[3])
        self.Att4 = AttentionGate(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv4 = ConvBlock(filters[4], filters[3])

        self.Up3 = UpConv(filters[3], filters[2])
        self.Att3 = AttentionGate(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv3 = ConvBlock(filters[3], filters[2])

        self.Up2 = UpConv(filters[2], filters[1])
        self.Att2 = AttentionGate(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv2 = ConvBlock(filters[2], filters[1])

        self.Up1 = UpConv(filters[1], filters[0])
        self.Att1 = AttentionGate(F_g=filters[0], F_l=filters[0], F_int=filters[0]//2)
        self.Up_conv1 = ConvBlock(filters[1], filters[0])

        self.Conv_1x1 = nn.Conv2d(filters[0], out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        e1 = self.Conv1(x)
        e2 = self.Maxpool(e1)
        e2 = self.Conv2(e2)
        e3 = self.Maxpool(e2)
        e3 = self.Conv3(e3)
        e4 = self.Maxpool(e3)
        e4 = self.Conv4(e4)
        e5 = self.Maxpool(e4)
        e5 = self.Conv5(e5)

        d4 = self.Up4(e5)
        x4 = self.Att4(g=d4, x=e4)
        d4 = torch.cat((x4, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x3 = self.Att3(g=d3, x=e3)
        d3 = torch.cat((x3, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x2 = self.Att2(g=d2, x=e2)
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Up1(d2)
        x1 = self.Att1(g=d1, x=e1)
        d1 = torch.cat((x1, d1), dim=1)
        d1 = self.Up_conv1(d1)

        out = self.Conv_1x1(d1)
        return out

class AttentionWNet(nn.Module):
    def __init__(self, in_channels=3, out_channels1=2, out_channels2=3, filters=[32, 64, 128, 256, 512]):
        super(AttentionWNet, self).__init__()
        # First UNet predicts vessels (out_channels1)
        self.unet1 = AttentionUNet(in_channels=in_channels, out_channels=out_channels1, filters=filters)
        # Second UNet takes concatenation of original input and OUT1, predicts A/V (out_channels2)
        self.unet2 = AttentionUNet(in_channels=in_channels + out_channels1, out_channels=out_channels2, filters=filters)

    def forward(self, x):
        # Forward through first UNet
        out1_logits = self.unet1(x)
        out1_probs = torch.softmax(out1_logits, dim=1)
        
        # Concatenate original input with first output
        x2 = torch.cat([x, out1_probs], dim=1)
        
        # Forward through second UNet
        out2_logits = self.unet2(x2)
        out2_probs = torch.softmax(out2_logits, dim=1)
        
        return out1_probs, out2_probs
