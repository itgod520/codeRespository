from __future__ import absolute_import, print_function
import torch
from torch import nn

from models import MemModule
from .cbam import CBAMBlock

class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleFeatureExtractor, self).__init__()

        self.conv1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv3d(in_channels, out_channels, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        out1x1 = self.conv1x1(x)
        out3x3 = self.conv3x3(x)
        out5x5 = self.conv5x5(x)

        # return torch.cat([out1x1, out3x3, out5x5], dim=1)
        return out1x1+out3x3+out5x5   


    
class SeparableConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(SeparableConv3d, self).__init__()
        self.depthwise_conv = nn.Conv3d(in_channels, in_channels, kernel_size, groups=in_channels, padding=kernel_size//2, stride=stride)
        self.pointwise_conv = nn.Conv3d(in_channels, out_channels, 1, stride=1)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x
    
    
    
    
class AutoEncoderCov3DMem(nn.Module):
    def __init__(self, chnum_in, mem_dim, shrink_thres=0.0025):
        super(AutoEncoderCov3DMem, self).__init__()
        print('AutoEncoderCov3DMem')
        self.chnum_in = chnum_in
        feature_num = 128
        feature_num_2 = 96
        feature_num_x2 = 256
        self.encoder = nn.Sequential(
            SeparableConv3d(self.chnum_in, feature_num_2, kernel_size=3, stride=(1, 2, 2)),
            nn.BatchNorm3d(feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),
            SeparableConv3d(feature_num_2, feature_num, kernel_size=3, stride=(2, 2, 2)),
            nn.BatchNorm3d(feature_num),
            nn.LeakyReLU(0.2, inplace=True),
            SeparableConv3d(feature_num, feature_num_x2, kernel_size=3, stride=(2, 2, 2)),
            nn.BatchNorm3d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),
            SeparableConv3d(feature_num_x2, feature_num_x2, kernel_size=3, stride=(2, 2, 2)),
            nn.BatchNorm3d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        
        self.MSFE=MultiScaleFeatureExtractor(256,256)
        self.cbam=CBAMBlock(channel=256,reduction=16,kernel_size=7)
        # self.mem_rep = MemModule(mem_dim=2000, fea_dim=feature_num_x2, shrink_thres =0.0025)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(feature_num_x2, feature_num_x2, kernel_size=3, stride=(2, 2, 2), padding=(1, 1, 1),
                               output_padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(feature_num_x2, feature_num, kernel_size=3, stride=(2, 2, 2), padding=(1, 1, 1),
                               output_padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(feature_num, feature_num_2, kernel_size=3, stride=(2, 2, 2), padding=(1, 1, 1),
                               output_padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),
            # SeparableConv3d(feature_num_2, self.chnum_in, kernel_size=3, stride=(1, 2, 2),,output_padding=(0, 1, 1)),
            nn.ConvTranspose3d(feature_num_2, self.chnum_in, kernel_size=3, stride=(1, 2, 2), padding=(1, 1, 1),
                               output_padding=(0, 1, 1)),
        )

    def forward(self, x):
        f = self.encoder(x)
#         print("the shape of the output from the encoder", f.shape)
        # res_mem = self.mem_rep(f)
        # f = res_mem['output']
        # att = res_mem['att']
        
        # out_cbam_res=f
        
        # out_msfe_res=f
        
        f=self.cbam(f)
        #链接残差
        # f+=out_cbam_res
        
        # out_msfe_res=f
        
        # out_msfe_res=self.MSFE(out_msfe_res)
        #链接残差
        # f+=out_msfe_res
        
        
        # f=self.cbam(out4)
        output = self.decoder(f)
        return output
