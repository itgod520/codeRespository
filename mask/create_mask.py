import torch
from typing import Dict, List, Tuple

from torch import Tensor
import  math
import numpy as np
import cv2
import matplotlib.pyplot as plt
def _create_disjoint_masks(
        img_size: Tuple[int, int],
        cutout_size: int = 8,
        num_disjoint_masks: int = 16,
) -> List[Tensor]:
    img_h, img_w = img_size[0],img_size[1]
    print(img_h,img_w)
    grid_h = math.ceil(img_h / cutout_size)
    grid_w = math.ceil(img_w / cutout_size)
    num_grids = grid_h * grid_w
    disjoint_masks = []
    for grid_ids in np.array_split(np.random.permutation(num_grids), num_disjoint_masks):
        flatten_mask = np.ones(num_grids)
        flatten_mask[grid_ids] = 0
        mask = flatten_mask.reshape((grid_h, grid_w))
        mask = mask.repeat(cutout_size, axis=0).repeat(cutout_size, axis=1)
        mask = torch.tensor(mask, requires_grad=False, dtype=torch.float)

        disjoint_masks.append(mask)

    return disjoint_masks

# def _reconstruct( mb_img: Tensor, cutout_size: int) -> Tensor:
#
#     _, _, h, w = mb_img.shape
#     print("h,w:",h,w)
#
#     disjoint_masks = _create_disjoint_masks((h, w), cutout_size, 3)
#
#     mb_reconst = 0
#     for mask in disjoint_masks:
#         mb_cutout = mb_img * mask
#         mb_inpaint = model(mb_cutout)
#         mb_reconst += mb_inpaint * (1 - mask)
#
#     return mb_reconst



opt=[2, 4, 8, 16]
import random
cutout_size = 4
print("cutout_size:",cutout_size)

# img=cv2.imread("D:/dataset/Crowd/1.jpg")
# img=cv2.resize(img,(256,256))
# img=torch.from_numpy(img)

result=_create_disjoint_masks([256,256],cutout_size,3)
print(len(result))
print(result)
# mb_reconst=_reconstruct(img,cutout_size)

# mb_reconst

img=cv2.imread("D:/dataset/Crowd/1.jpg")
img=cv2.resize(img,(256,256))
print(np.shape(img))
print("result:",)
img=torch.from_numpy(img)

# print("result:",np.shape(result))
for i in  range(3):

    new_img=img[:,:,i]*result[i]
    plt.imshow(new_img)
    plt.show()


# _reconstruct(img,cutout_size)


opt=[2, 4, 8, 16]
import random
cutout_size = random.choice(opt)
print(cutout_size)


# _reconstruct(,cutout_size)