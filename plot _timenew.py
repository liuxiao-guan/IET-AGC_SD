
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import numpy as np
import yaml
import io

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torchvision import transforms
import torchvision.datasets as datasets
import torchvision.models as torchvision_models
from torch.utils.data import Dataset
# import models.moco_vits as moco_vits
import utils_ret
from utils_ret import extract_features
import pickle
from PIL import Image, ImageFile
# import natsort
# from models import mae_vits
import timm

import matplotlib.pyplot as plt
# import LovelyPlots.utils as lp
import itertools
import json, glob 
import seaborn as sns

from skimage import io, color, img_as_ubyte
# from skimage.feature import greycomatrix, greycoprops
from sklearn.metrics.cluster import entropy
import pandas as pd
import cv2
import matplotlib.font_manager
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Times New Roman'
rcParams['font.size'] = 20  # 设置字体大小为20


dp = "_instancelevel_blip_nodup_bs4_gpu4"

ret_savepath = f'/root/autodl-tmp/logs/Projects/DCR/ret_plots/DCR/laion_frozentext/{dp}/'
os.makedirs(ret_savepath,exist_ok=True)

# torch.save(simscores.cpu(), os.path.join(ret_savepath, "similarity.pth"))
# torch.save(bg_simscores.cpu(), os.path.join(ret_savepath, "similarity_wtrain.pth"))

simscores = torch.load(os.path.join(ret_savepath, "similarity.pth"))
bg_simscores = torch.load(os.path.join(ret_savepath, "similarity_wtrain.pth"))

main_v,main_l = simscores.topk(1,axis=1,largest=True) 
bg_v,bg_l = bg_simscores.topk(2,axis=1,largest=True)
bg_v = bg_v[:,-1] #remove the first one since it is to self.

print(main_v.shape, bg_v.shape)
plt.figure(figsize=(6.5,6))


x0 =  main_v.cpu().numpy()
x1 = bg_v.cpu().numpy()
bin_width= 0.005
import math
nbins = math.ceil(1 / bin_width)
bins = np.linspace(0,1, nbins)


# # 总样本数
# total_samples_0 = len(x0)
# total_samples_1 = len(x1)

# # 计算每个 bin 的频数 (density=True)
# hist0, _ = np.histogram(x0, bins=bins, density=True)
# hist1, _ = np.histogram(x1, bins=bins, density=True)

# # 将频数转换为百分比 (乘以样本数并转换为百分比)
# # 将概率密度转换为百分比
# percent0 = hist0 * bin_width * 100  # 转换为百分比
# percent1 = hist1 * bin_width * 100  # 转换为百分比

# # 绘制直方图
# plt.bar(bins[:-1], percent0, width=bin_width, alpha=0.4, label='sim(gen,train)', align='edge')
# plt.bar(bins[:-1], percent1, width=bin_width, alpha=0.6, label='sim(train,train)', align='edge')

# # 设置标签和图例
# # plt.title('Histogram with Percentage Y-axis')
# plt.xlabel('Value')
# plt.ylabel('Percentage (%)')
# plt.legend()
# plt.savefig(f"{ret_savepath}/histogram_IET-AGC+.png")
# # 显示图形
# plt.show()


fig = plt.hist(x0, bins, alpha=0.4, label='sim(gen,train)',density=True)
fig = plt.hist(x1, bins, alpha=0.6, label='sim(train,train)',density=True)

plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*0.005*100}'))
plt.xlabel('Sim Score')
plt.ylabel('Percentage (%)')
plt.legend(loc='upper right')

plt.savefig(f"{ret_savepath}/histogram.png")
# plt.close(fig)
plt.figure()
