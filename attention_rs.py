##
import numpy as np
from tifffile import imread
import albumentations as A
from dino_utils import to_rgb, change_mode, show_image
import torch
from dino_classification import DINOClassification
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
import torch.nn as nn


##
ckpt_path = "/mnt/cluster/data_hdd/jazibmodels/dino_pt_200k_240306/200k_step.ckpt"
ckpt_key = 'teacher'

ckpt = DINOClassification.load_checkpoint(ckpt_path, ckpt_key)
model = DINOClassification.prepare_arch('vit_base', ckpt, 16)

##
path_img = "/home/jazib/projects/SelfSupervisedLearning/NRW_5k/293591-5637040.tif"
img = imread(path_img)  # HWC
# img = to_rgb(img, mode='HWC')
crop = A.random_crop(img, 224, 224, 0, 0)  # HWC

show_image(crop, "crop")
plt.show()
##
crop_t = ToTensor()(crop)
w, h = crop_t.shape[1] - crop_t.shape[1] % 16, crop_t.shape[2] - crop_t.shape[2] % 16
crop_t = crop_t[:, :w, :h].unsqueeze(0)
##
w_featmap = crop_t.shape[-2] // 16
h_featmap = crop_t.shape[-1] // 16

attentions = model.get_last_selfattention(crop_t.to('cpu'))

nh = attentions.shape[1] # number of head
##
# we keep only the output patch attention
attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
##
attentions = attentions.reshape(nh, w_featmap, h_featmap)
attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=16, mode="nearest")[0].detach().numpy()
##
plt.imshow(attentions[2])
plt.show()
##
plt.figure(figsize=(10, 10))
plt.imshow(attentions[11],  cmap='Greys')
plt.imshow(to_rgb(crop, mode='HWC'), alpha=0.5)
plt.show()