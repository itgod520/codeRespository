import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.nn.init as init
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
# from torch.utils.tensorboard import SummaryWriter
import cv2
import math
from collections import OrderedDict
import copy
import time
import data.utils as data_utils
import utils.eval as eval_utils
# import data.utils as data_utils
import models.loss as loss
import utils
from models import AutoEncoderCov3D, AutoEncoderCov3DG,AutoEncoderCov3DD
# from models.att_unet import R2AttU_Net
# from models.BCDUNet import BCDUNet

# from UNET import UNet
from newmodel import UNet
from torch.utils.data import DataLoader
import argparse

import pytorch_ssim.ssim as pytorch_ssim

print("--------------PyTorch VERSION:", torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("..............device", device)

parser = argparse.ArgumentParser(description="MemoryNormality")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
parser.add_argument('--epochs', type=int, default=1050, help='number of epochs for training')
parser.add_argument('--val_epoch', type=int, default=2, help='evaluate the model every %d epoch')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=1, help='channel of input images')
parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
parser.add_argument('--t_length', type=int, default=4, help='length of the frame sequences')
parser.add_argument('--ModelName', help='AE/MemAE', type=str, default='MemAE')
parser.add_argument('--ModelSetting', help='Conv3D/Conv3DSpar', type=str,
                    default='Conv3DSpar')  # give the layer details later
parser.add_argument('--MemDim', help='Memory Dimention', type=int, default=2000)
parser.add_argument('--EntropyLossWeight', help='EntropyLossWeight', type=float, default=0.0002)
parser.add_argument('--ShrinkThres', help='ShrinkThres', type=float, default=0.0025)
parser.add_argument('--Suffix', help='Suffix', type=str, default='Non')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=0, help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='UCSDped2', help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='./dataset/', help='directory of data')
parser.add_argument('--exp_dir', type=str, default='log', help='directory of log')
parser.add_argument('--version', type=int, default=0, help='experiment version')

args = parser.parse_args()

random_seed = 1
np.random.seed(random_seed)
torch.manual_seed(random_seed)

torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance


def arrange_image(im_input):
    im_input = np.transpose(im_input, (0, 2, 1, 3, 4))
    b, t, ch, h, w = im_input.shape
    im_input = np.reshape(im_input, [b * t, ch, h, w])
    return im_input


train_folder, test_folder = data_utils.give_data_folder(args.dataset_type,
                                                        args.dataset_path)

print("The training path", train_folder)
print("The testing path", test_folder)

frame_trans = data_utils.give_frame_trans(args.dataset_type, [args.h, args.w])

train_dataset = data_utils.DataLoader(train_folder, frame_trans, time_step=args.t_length - 1, num_pred=1)
test_dataset = data_utils.DataLoader(test_folder, frame_trans, time_step=args.t_length - 1, num_pred=1)

train_batch = data.DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, drop_last=True)
test_batch = data.DataLoader(test_dataset, batch_size = args.batch_size,
                             shuffle=False, num_workers=args.num_workers, drop_last=True)




print("Training data shape", len(train_batch))
# print("Validation data shape", len(test_batch))

# Model setting

if (args.ModelName == 'AE'):
    model = AutoEncoderCov3D(args.c)
elif (args.ModelName == 'MemAE'):
    model_g = AutoEncoderCov3DG(1)
    model_d = AutoEncoderCov3DD(1)
else:
    model = []
    print('Wrong Name.')

# ####VAL############################################################
#
height, width = 256, 256
ch = 1
num_frame = 16
batch_size = 1

gt_file = "ckpt/%s_gt.npy" % (args.dataset_type)

if args.dataset_type == "Avenue":
    data_dir = args.dataset_path + "Avenue/frames/testing/"
elif "UCSD" in args.dataset_type:
    data_dir = args.dataset_path + "%s/Test_jpg/" % args.dataset_type
else:
    print("The dataset is not available..........")
    pass

frame_trans = transforms.Compose([
    transforms.Resize([height, width]),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
unorm_trans = utils.UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

data_loader = data_utils.DataLoader(data_dir, frame_trans, time_step=num_frame - 1, num_pred=1)
video_data_loader = DataLoader(data_loader, batch_size=batch_size, num_workers=0, shuffle=False)



img_crop_size = 0
recon_error_list = [None] * len(video_data_loader)

####val###########################################################


model_g = model_g.to(device)
model_d = model_d.to(device)
parameter_listg = [p for p in model_g.parameters() if p.requires_grad]

parameter_listd = [p for p in model_d.parameters() if p.requires_grad]

for name, p in model_g.named_parameters():
    if not p.requires_grad:
        print("---------NO GRADIENT-----", name)

optimizer_g = torch.optim.Adam(parameter_listg, lr=args.lr,betas=(0.5,0.999))
optimizer_d = torch.optim.Adam(parameter_listd, lr=args.lr,betas=(0.5,0.999))


####定义ssim损失
# loss_ssim = pytorch_ssim.SSIM()



# Report the training process
log_dir = os.path.join(args.exp_dir, args.dataset_type, 'lr_%.5f_entropyloss_%.5f_version_%d' % (
    args.lr, args.EntropyLossWeight, args.version))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
orig_stdout = sys.stdout
f = open(os.path.join(log_dir, 'log.txt'),'w')
sys.stdout= f

for arg in vars(args):
    print(arg, getattr(args, arg))




# Training

train_loss = []

train_loss_mid = []
best_auc = 0.0
# tr_re_loss=[]
test_loss = []

test_loss_mid = []
max_auc = []


loss_adv=loss.l2_loss
loss_recon=nn.L1Loss()
loss_enc=loss.l2_loss
loss_bce=nn.BCELoss()


real_label = torch.ones(size=(1,4), dtype=torch.float32).to(device)
fake_label = torch.zeros(size=(1, 4), dtype=torch.float32).to(device)

for epoch in range(args.epochs):
    print("第:", epoch + 1)
    # model.train()
    tr_re_loss, tr_mem_loss, tr_tot = 0.0, 0.0, 0.0
    progress_bar = tqdm(train_batch)
    recon_error_list_tr = []
    tr_re_loss = []
    for batch_idx, frame in enumerate(progress_bar):
        # print(frame.shape)
        progress_bar.update()
        frame = frame.reshape(args.batch_size, args.t_length, args.c, args.h, args.w)
        frame = frame.permute(0, 2, 1, 3, 4)
        # print(frame.shape)
        # frame = frame.view(args.batch_size, args.t_length,args.c, args.h, args.w)
        frame = frame.to(device)
        # print("让我看看你的形状：",frame.shape)
        #训练生成器
        model_g.train()
        g_output=model_g(frame)
        real_img=g_output['real_img']
        fake_img=g_output['fake_img']
        latent_i=g_output['latent_i']
        latent_o=g_output['latent_o']
        # print("real_img:",real_img.shape)
        # print("fake_img:",fake_img.shape)
        real_pg,real_fg=model_d(real_img)['pre_p'],model_d(real_img)['latent']
        fake_pg,fake_fg=model_d(fake_img)['pre_p'],model_d(fake_img)['latent']
        ####
        rencon_loss=loss_recon(real_img,fake_img)
        encon_loss=loss_enc(latent_o,latent_i)
        adv_loss=loss_adv(real_fg,fake_fg)

        g_loss=50*rencon_loss+1*encon_loss+adv_loss*1
        g_loss.backward(retain_graph=True)


        ####训练判别器########
        model_d.train()
        real_pd, real_fd = model_d(real_img)['pre_p'], model_d(real_img)['latent']
        fake_pd, fake_fd = model_d(fake_img.detach())['pre_p'], model_d(fake_img.detach())['latent']

        real_bce_loss=loss_bce(real_pd,real_label)
        fake_bce_loss=loss_bce(fake_pd,fake_label)

        d_loss=(real_bce_loss+fake_bce_loss)*0.5

        d_loss.backward()




        optimizer_g.step()
        optimizer_g.zero_grad()



        optimizer_d.step()
        optimizer_d.zero_grad()




        # tot_loss = re_loss1
        # tr_re_loss.append(re_loss1.data.item())
        # tr_mem_loss += mem_loss.data.item()
        # tr_tot += tot_loss.data.item()
        # print("重构误差：",re_loss.data.item())
        # train_loss_mid.append((re_loss1).data.item())
        # print("第"+str(epoch+1)+"轮的平均训练误差：",np.mean(train_loss_mid))


    # train_writer.add_scalar('learning_rate', current_lr, epoch)

    with torch.no_grad():

        # model.eval().to(device)

        ts_re_loss = []
        re_loss_val, mem_loss_val = 0.0, 0.0
        for batch_idx, frames in enumerate(video_data_loader):
            frames = frames.reshape([batch_size, num_frame, ch, height, width])
            frames = frames.permute(0, 2, 1, 3, 4)
            # frames = frames.view(batch_size, num_frame,ch, height, width)
            frame = frames.to(device)
            model_output = model_g(frame)

            recon_frames = model_output['latent_i']
            frames=model_output['latent_o']
            recon_np1 = utils.vframes2imgs(unorm_trans(recon_frames.data), step=1, batch_idx=0)
            input_np1 = utils.vframes2imgs(unorm_trans(frames.data), step=1, batch_idx=0)

            r1 = utils.crop_image(recon_np1, img_crop_size) - utils.crop_image(input_np1, img_crop_size)

            sp_error_map1 = sum(r1 ** 2) ** 0.5

            recon_error1 = np.mean(sp_error_map1.flatten())

            recon_error_list[batch_idx] = recon_error1

            ts_re_loss.append(recon_error1)

        auc = eval_utils.eval_video2(gt_file, recon_error_list,epoch, args.dataset_type)
        max_auc.append(auc)
        print("模型精度：", auc)
        print("---------------------------------------------------")
        print("当前轮次最大精度:", np.max(max_auc))
        print("====================================================")
    if best_auc <= auc:
        best_auc = auc
        # torch.save(model.state_dict(), log_dir + "/model-{:04d}.pt".format(epoch))
    else:
        print("best_acurrcy is:", best_auc)

#     if epoch ==50 or epoch ==100 or epoch ==200 or epoch ==500  or epoch ==1000:

#         plt.plot(train_loss,label="train_loss")
#         plt.plot(test_loss,label="val_loss")
#         plt.legend()
#         plt.savefig("./plot/re_loss{:04d}.png".format(epoch))
#         plt.show()

sys.stdout = orig_stdout
f.close()

#
#
#
