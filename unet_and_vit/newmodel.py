import torch.nn as nn
import torch
# import torch
from vit.vit_pytorch.simple_vit import SimpleViT

from models.memory_module import MemModule


# import models.mdl1

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x



class AE(nn.Module):
    def __init__(self,):

        super(AE,self).__init__()

        self.encode1=DoubleConv(16,64)
        self.encode2=DoubleConv(64,128)
        self.encode3=DoubleConv(128,256)
        self.bottleneck=DoubleConv(256,512)

        self.simplevit = SimpleViT(
            image_size=32,
            patch_size=8,
            num_classes=1000,
            dim=512  * 8 * 8,
            depth=6,
            heads=16,
            mlp_dim=2048,
            channels=512
        )

        self.decode1=DoubleConv(512,256)
        self.decode2=DoubleConv(256,128)
        self.decode3=DoubleConv(128,64)
        self.decode4=DoubleConv(64,16)


        self.fusion=nn.Sequential(
            nn.Conv2d(32,16,kernel_size=(3,3),stride=(1,1))
        )

    def forward(self,x1,x2):

        ######
        output1=self.encode1(x1)
        output2=self.encode2(output1)
        output3=self.encode3(output2)
        output4=self.bottleneck(output3)


        # mid_out=self.simplevit(output4)



        out1=self.decode1(output4)
        out2=self.decode2(out1)
        out3=self.decode3(out2)
        recon1=self.decode4(out3)
        # recon1=self.decode5(out4)



        ###########

        output1 = self.encode1(x2)
        output2 = self.encode2(output1)
        output3 = self.encode3(output2)
        output4 = self.bottleneck(output3)

        mid_out = self.simplevit(output4)

        out1 = self.decode1(mid_out)
        out2 = self.decode2(out1)
        out3 = self.decode3(out2)
        recon2 = self.decode4(out3)
        # recon2 = self.decode5(out4)

        merge=torch.cat([recon1,recon2],dim=1)

        last_recon=self.fusion(merge)

        return {'recon1':recon1,'recon2':recon2,'last_recon':last_recon}






class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.downs2 = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNet
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in features:
            self.downs2.append(DoubleConv(in_channels, feature))
            in_channels = feature

        #         self.latent=nn.Sequential(

        #             nn.Conv2d(16, 64, kernel_size=3, padding=1),
        #             nn.BatchNorm2d(64),
        #             nn.ReLU(inplace=True),
        #             nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #             nn.BatchNorm2d(64),
        #             nn.ReLU(inplace=True),
        #             nn.MaxPool2d(kernel_size=2, stride=2),

        #             nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #             nn.BatchNorm2d(128),
        #             nn.ReLU(inplace=True),
        #             nn.Conv2d(128, 128, kernel_size=3, padding=1),
        #             nn.BatchNorm2d(128),
        #             nn.ReLU(inplace=True),
        #             nn.MaxPool2d(kernel_size=2, stride=2),

        #             nn.Conv2d(128, 256, kernel_size=3, padding=1),
        #             nn.BatchNorm2d(256),
        #             nn.ReLU(inplace=True),
        #             nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #             nn.BatchNorm2d(256),
        #             nn.ReLU(inplace=True),
        #             nn.MaxPool2d(kernel_size=2, stride=2),

        #             nn.Conv2d(256, 512, kernel_size=3, padding=1),
        #             nn.BatchNorm2d(512),
        #             nn.ReLU(inplace=True),
        #             nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #             nn.BatchNorm2d(512),
        #             nn.ReLU(inplace=True),
        #             nn.MaxPool2d(kernel_size=2, stride=2),

        #         )

        self.simplevit = SimpleViT(
            image_size=16,
            patch_size=4,
            num_classes=1000,
            dim=256 * 4 * 4 * 4,
            depth=6,
            heads=16,
            mlp_dim=2048,
            channels=1024
        )
        ### 设置估计

        # Up part of UNet
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        # skip_connections2=[]
        input = x
        # Down part of UNet
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottom part of UNet
        x = self.bottleneck(x)

        x = self.simplevit(x)

        # Up part of UNet
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            # skip_connection = skip_connections[len(skip_connections) - idx // 2 - 1]
            # x = torch.cat((x, skip_connection), dim=1)
            # x = self.ups[idx + 1](x)

        # Final layer
        output1 = self.final_conv(x)

        return {'output': output1}




if __name__ == '__main__':
    model=AE().cuda()
    x=torch.rand(1,16,256,256).cuda()
    y=model(x,x)

    print(y.shape)
