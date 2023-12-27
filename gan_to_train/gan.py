from __future__ import absolute_import, print_function
import torch
from torch import nn

from models import MemModule

class AutoEncoderCov3DG(nn.Module):
    def __init__(self, chnum_in):
        super(AutoEncoderCov3DG, self).__init__()
        print('AutoEncoderCov3DMem')
        self.chnum_in = chnum_in
        feature_num = 128
        feature_num_2 = 96
        feature_num_x2 = 256
        self.encoder1 = nn.Sequential(
            nn.Conv3d(self.chnum_in, feature_num_2, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(feature_num_2, feature_num, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(feature_num, feature_num_x2, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(feature_num_x2, feature_num_x2, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(feature_num_x2, feature_num_x2, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1),
                               output_padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(feature_num_x2, feature_num, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1),
                               output_padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(feature_num, feature_num_2, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1),
                               output_padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(feature_num_2, self.chnum_in, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1),
                               output_padding=(0, 1, 1))
        )

        self.encoder2=nn.Sequential(
            nn.Conv3d(self.chnum_in, feature_num_2, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(feature_num_2, feature_num, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(feature_num, feature_num_x2, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(feature_num_x2, feature_num_x2, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        input_img=x
        f = self.encoder1(x)
        latent_i=f
        f=self.decoder(f)
        output=f
        # print("output:",output.shape)
        f=self.encoder2(f)
        latent_o=f

        # output = self.decoder(f)
        return {'real_img':input_img,'fake_img': output,'latent_i':latent_i,'latent_o':latent_o}



class AutoEncoderCov3DD(nn.Module):
    def __init__(self, chnum_in,):
        super(AutoEncoderCov3DD, self).__init__()
        print('AutoEncoderCov3DMem')
        self.chnum_in = chnum_in
        feature_num = 128
        feature_num_2 = 96
        feature_num_x2 = 256
        self.encoder1 = nn.Sequential(
            nn.Conv3d(self.chnum_in, feature_num_2, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),
            (nn.Conv3d(feature_num_2, feature_num, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))),
            nn.BatchNorm3d(feature_num),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(feature_num, feature_num_x2, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(feature_num_x2, feature_num_x2, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),

        )
        self.liner=nn.Sequential(
            nn.Linear(256 * 2* 16 * 16, 512),
            nn.Linear(512, 128),
            nn.Linear(128, 16),
            nn.Softmax(dim=1)
        )


    def forward(self, x):
        input_img=x
        f = self.encoder1(x)
        feature=f
        f=f.view(-1,1 * 256 * 2 * 16 * 16)

        p=self.liner(f)





        return {'pre_p':p,'latent':feature}





if __name__ == '__main__':
    model=AutoEncoderCov3DG(1)
    x=torch.randn(4,1,16,256,256)
    y=model(x)
    print(y['latent_i'].shape)
    print(y['latent_o'].shape)
    print(y['real_img'].shape)
    print(y['fake_img'].shape)
    loss=nn.MSELoss()
    print(loss(y['latent_i'],y['latent_o']))

    modeld=AutoEncoderCov3DD(1)
    p=modeld(x)['pre_p']

    print("p:",p.shape)
    real_label = torch.ones(size=(4,16), dtype=torch.float32)
    fake_label = torch.zeros(size=(4, 16), dtype=torch.float32)
    print(real_label)
    print(fake_label)
    loss_bec=nn.BCELoss()
    # loss_bec(p,fake_label)

    print(loss_bec(p,fake_label))
    print(loss_bec(p, real_label))
