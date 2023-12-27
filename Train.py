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
from typing import Dict, List, Tuple

from torch import Tensor
from collections import OrderedDict
import copy
import time
import data.utils as data_utils
import utils.eval as eval_utils
# import data.utils as data_utils
import models.loss as loss
import utils
from models import AutoEncoderCov3D, AutoEncoderCov3DMem
# from models.att_unet import R2AttU_Net
# from models.BCDUNet import BCDUNet
import random
# from UNET import UNet
from vitmodel import UNet
from torch.utils.data import DataLoader
import argparse
from models.loss import get_memory_loss

# # import pytorch_ssim.ssim as pytorch_ssim
# from  models.ssim import SSIMLoss
# from  models.msgms import MSGMSLoss
print("--------------PyTorch VERSION:", torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("..............device", device)

parser = argparse.ArgumentParser(description="MemoryNormality")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=2, help='batch size for training')
parser.add_argument('--epochs', type=int, default=10050, help='number of epochs for training')
parser.add_argument('--val_epoch', type=int, default=2, help='evaluate the model every %d epoch')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=1, help='channel of input images')
parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
parser.add_argument('--t_length', type=int, default=16, help='length of the frame sequences')
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
parser.add_argument('--pseudo_anomaly_jump', type=float, default=0.1,
                    help='pseudo anomaly jump frame (skip frame) probability. 0 no pseudo anomaly')
args = parser.parse_args()

random_seed = 2020
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
# test_dataset = data_utils.DataLoader(test_folder, frame_trans, time_step=args.t_length - 1, num_pred=1)

train_batch = data.DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, drop_last=True)
# test_batch = data.DataLoader(test_dataset, batch_size = args.batch_size,
#                              shuffle=False, num_workers=args.num_workers, drop_last=True)

print("Training data shape", len(train_batch))
# print("Validation data shape", len(test_batch))


# def _create_disjoint_masks(
#         img_size: Tuple[int, int],
#         cutout_size: int = 8,
#         num_disjoint_masks: int = 16,
# ) -> List[Tensor]:
#     img_h, img_w = img_size[0], img_size[1]
#     # print(img_h, img_w)
#     grid_h = math.ceil(img_h / cutout_size)
#     grid_w = math.ceil(img_w / cutout_size)
#     num_grids = grid_h * grid_w
#     disjoint_masks = []
#     for grid_ids in np.array_split(np.random.permutation(num_grids), num_disjoint_masks):
#         flatten_mask = np.ones(num_grids)
#         flatten_mask[grid_ids] = 0
#         mask = flatten_mask.reshape((grid_h, grid_w))
#         mask = mask.repeat(cutout_size, axis=0).repeat(cutout_size, axis=1)
#         mask = torch.tensor(mask, requires_grad=False, dtype=torch.float)
#
#         disjoint_masks.append(mask)
#
#     return disjoint_masks
#
# import matplotlib.pyplot as plt
# def _reconstruct(mb_img: Tensor, cutout_size: int) -> Tensor:
#     _, _, h, w = mb_img.shape
#     # print("h,w:", h, w)
#
#     disjoint_masks = _create_disjoint_masks((h, w), cutout_size, 4)
#
#     mb_reconst = 0
#     for mask in disjoint_masks:
#         mb_cutout = mb_img * mask.cuda()
#         mb_inpaint = model(mb_cutout).cuda()
#
#
#         # plt.imshow(mb_inpaint.detach().numpy())
#         # plt.show()
#
#
#         mb_reconst += mb_inpaint * (1 - mask).cuda()
#
#
#
#     return mb_reconst


# Model setting

if (args.ModelName == 'AE'):
    model = AutoEncoderCov3D(args.c)
elif (args.ModelName == 'MemAE'):
    model= AutoEncoderCov3DMem(1,256,0.0025)
    model = nn.DataParallel(model)
    model.cuda()
else:
    model = []
    print('Wrong Name.')

####VAL############################################################

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

data_loader = data_utils.DataLoader(data_dir, frame_trans, time_step=16 - 1, num_pred=1)
video_data_loader = DataLoader(data_loader, batch_size=batch_size, num_workers=0, shuffle=False)

img_crop_size = 0
recon_error_list = [None] * len(video_data_loader)

#####val###########################################################


model = model.to(device)
parameter_list = [p for p in model.parameters() if p.requires_grad]

for name, p in model.named_parameters():
    if not p.requires_grad:
        print("---------NO GRADIENT-----", name)

optimizer = torch.optim.Adam(parameter_list, lr=args.lr)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40], gamma=0.2)  # version 2

####定义ssim损失
# loss_ssim=pytorch_ssim.SSIM()

# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max =args.epochs)

# Report the training process
log_dir = os.path.join(args.exp_dir, args.dataset_type, 'lr_%.5f_entropyloss_%.5f_version_%d' % (
    args.lr, args.EntropyLossWeight, args.version))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
orig_stdout = sys.stdout
f = open(os.path.join(log_dir, 'log.txt'), 'w')
sys.stdout = f

for arg in vars(args):
    print(arg, getattr(args, arg))

loss_func_mse = nn.MSELoss(reduction='mean').cuda()

train_loss = []

train_loss_mid = []
best_auc = 0.0
# tr_re_loss=[]
test_loss = []

test_loss_mid = []
max_auc = []
for epoch in range(args.epochs):

    opt = [2, 4, 8, 16]


    cutout_size = 4

    pseudolossepoch = 0
    lossepoch = 0
    pseudolosscounter = 0
    losscounter = 0

    print("第:", epoch + 1)
    model.train()
    tr_re_loss, tr_mem_loss, tr_tot = 0.0, 0.0, 0.0
    progress_bar = tqdm(train_batch)
    recon_error_list_tr = []
    tr_re_loss = []
    for batch_idx, frame in enumerate(progress_bar):
        progress_bar.update()
        # jump_pseudo_stat = []
        # cls_labels = []

        frame = frame.reshape([args.batch_size, args.t_length, args.c, args.h, args.w])
        frame = frame.permute(0, 2, 1, 3, 4)
        # frame = frame.view(args.batch_size, args.t_length * args.c, args.h, args.w)
        # real_img=frame
        frame = frame.to(device)
        # print(frame.shape)

        input_frame = frame.to(device)
        # sample=torch.rand(8,4,256,256)
        # disjoint_masks=_create_disjoint_masks([256,256],cutout_size,4)
        # i=0
        # for data in input_frame:
        #     mb_cutout = data * disjoint_masks
        #     mb_inpaint = model(mb_cutout)
        #     mb_reconst [i]= mb_inpaint * (1 - mask)
        #     i=i+1
        # b,c,h,w=input_frame.shape
        #
        # for i in range(b):
        #     ##c,h,w
        #     data=input_frame[i]
        #     for j in range(c):
        #         data[j]=data[j].cuda()*disjoint_masks[j].cuda()
        #
        #     sample[i]=data
        #
        # recon=model(sample)
        #
        # mb_reconst=torch.rand(8,4,256,256).cuda()

        # for i in range(b):
        #     data=recon[i]
        #     for j in range(c):
        #         data[j] = data[j].cuda() * (1-disjoint_masks[j]).cuda()
        #     mb_reconst[i]=data

        # mb_reconst += mb_inpaint * (1 - mask)

        # mb_inpaint=torch.rand(8,4,256,256)
        # for i in range(b):
        #     ##c,h,w
        #     data=mb_inpaint[i]
        #     for j in range(c):
        #         data[j]=data[j].cuda()*(1-disjoint_masks[j]).cuda()
        #
        #     mb_reconst[i]=data



        # mb_reconst += mb_inpaint * (1 - mask)






        # recon=recon.to(device)
        input_frame = frame.to(device)
        # recons=_reconstruct(input_frame,cutout_size)
        # print("让我看看recon:",recon.shape)
        # print(input_frame-recon)
        # print("==========================")

        ########start training

        model_output = model(frame)
        recons,att = model(input_frame)['output'],model(input_frame)['att']
        re_loss1 = loss.get_reconstruction_loss(input_frame, recons, mean=0.5, std=0.5)
        mem_loss=loss.get_memory_loss(att)
        # mse=nn.MSELoss(reduction="mean").cuda()
        # mse_loss=loss_func_mse(input_frame,mb_reconst)
        # print("让我看看mse_loss:",mse_loss.shape)
        # print("让我看看mse_loss:", mse_loss)
        # ssim_loss=SSIMLoss()._ssim(input_frame, mb_reconst).to(device)
        # msgms=MSGMSLoss()._gms(input_frame, mb_reconst).to(device)

        # print(mse_loss.shape)
        # print("====================================")
        # print(ssim_loss.shape)
        # print("=====================================")
        # print(msgms.shape)

        total=re_loss1


        total.backward()
        optimizer.step()
#torch.ones_like(total)
    #     print("第"+str(epoch+1)+"轮的平均训练误差：",np.mean(train_loss_mid))
    #     train_loss.append(np.mean(train_loss_mid))

    #     scheduler.step()

    with torch.no_grad():

        model.eval().to(device)

        ts_re_loss = []
        re_loss_val, mem_loss_val = 0.0, 0.0
        for batch_idx, frames in enumerate(video_data_loader):
            frames = frames.reshape([batch_size, num_frame, ch, height, width])
            frames = frames.permute(0, 2, 1, 3, 4)
            # frames = frames.view(batch_size, num_frame * ch, height, width)

            # real_img_test = frames
            # real_img_test = real_img_test.to(device)
            frame = frames.to(device)
            imgs=frame.view(1,1,16,256,256)
            model_output = model(frame)

            recon_frames = model_output
            outputs = recon_frames.view(1, 1, 16, 256, 256)
            frame = frame.view(1, 1, 16, 256, 256)
            mseimgs = loss_func_mse(outputs[0, :, 8], imgs[0, :, 8]).item().cpu().detach().numpy()

            recon_np1 = utils.vframes2imgs(unorm_trans(recon_frames.data), step=1, batch_idx=0)
            input_np1 = utils.vframes2imgs(unorm_trans(frame.data), step=1, batch_idx=0)

            r1 = utils.crop_image(recon_np1, img_crop_size) - utils.crop_image(input_np1, img_crop_size)
            # print(r1.shape)
            sp_error_map1 = sum(r1 ** 2) ** 0.5
            # print("sp_error_map1:",sp_error_map1.shape)
            mseimgs = mseimgs[:, :, np.newaxis]
            mseimgs = (mseimgs - np.min(mseimgs)) / (np.max(mseimgs) - np.min(mseimgs))
            mseimgs = mseimgs * 255
            mseimgs = mseimgs.astype(dtype=np.uint8)
            color_mseimgs = cv2.applyColorMap(mseimgs, cv2.COLORMAP_JET)
            if batch_idx == 0:
                os.mkdir('./err_img/' + str(epoch) + '/')
                os.mkdir('./recon/' + str(epoch) + '/')
                os.mkdir('./GT/' + str(epoch) + '/')

            cv2.imwrite(os.path.join('./err_img/' + str(epoch) + '/', 'MSE_{:04d}.jpg').format(batch_idx),
                        color_mseimgs)
            # print("让我看看：",np.shape(outputs[0,:,8].cpu().detach().numpy()))

            output = (outputs[0, :, 8].cpu().detach().numpy() + 1)
            output = (outputs[0, :, 8].cpu().detach().numpy() + 1) * 127.5
            # output = (outputs[0, :, 8].cpu().detach().numpy() + 1)

            output = output.transpose(1, 2, 0).astype(dtype=np.uint8)
            cv2.imwrite(os.path.join('./recon/' + str(epoch) + '/', 'Recon_{:04d}.jpg').format(batch_idx), output)

            saveimgs = (frame[0, :, 8].cpu().detach().numpy() + 1) * 127.5
            saveimgs = saveimgs.transpose(1, 2, 0).astype(dtype=np.uint8)
            cv2.imwrite(os.path.join('./GT/' + str(epoch) + '/', 'GT_{:04d}.png').format(batch_idx), saveimgs)

            recon_error1 = np.mean(sp_error_map1.flatten())

            recon_error_list[batch_idx] = recon_error1

            ts_re_loss.append(recon_error1)

        auc = eval_utils.eval_video2(gt_file, recon_error_list, epoch, args.dataset_type)
        max_auc.append(auc)
        print("模型精度：", auc)
        print("---------------------------------------------------")
        print("当前轮次最大精度:", np.max(max_auc))
        print("====================================================")
    if best_auc <= auc:
        best_auc = auc

        torch.save(model.state_dict(), log_dir + "/model-{:04d}.pt".format(epoch))
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




