import torch
import PIL
import numpy as np
import math
import os
import shutil
import matplotlib.pyplot as plt
from Network import TOFlow

model_name = 'interp'

def evaluate(test_img_dir, eval_img_dir, cuda_flag=False, gpuID=0):
    """
    :param test_img_dir: the directory of test dataset (vimeo-test)
    :param eval_img_dir: the directory where you want to save out.pngs
    :param cuda_flag: whether you want to use a gpu.
    :param gpuID: which gpu you want to use.
    :return:
    """
    if cuda_flag:
        torch.cuda.set_device(gpuID)
    net = TOFlow(256, 448, cuda_flag=cuda_flag)
    net.load_state_dict(torch.load(os.path.join('.', 'toflow_models', model_name + '.pkl')))
    def mkdir_if_not_exist(path):
        if not os.path.exists(path):
            os.mkdir(path)
    if cuda_flag:
        net.eval().cuda()
    else:
        net.eval()
    mkdir_if_not_exist(eval_img_dir)
    for video in os.listdir(test_img_dir):
        mkdir_if_not_exist(os.path.join(eval_img_dir, video))
        print('Processing %s' % video)
        for tri in os.listdir(os.path.join(test_img_dir, video)):
            mkdir_if_not_exist(os.path.join(eval_img_dir, video, tri))
            f1name = os.path.join(test_img_dir, video, tri, 'im1.png')
            f2name = os.path.join(test_img_dir, video, tri, 'im2.png')
            f3name = os.path.join(test_img_dir, video, tri, 'im3.png')
            frameFirst = np.array(plt.imread(f1name)).transpose(2, 0, 1)
            frameSecond = np.array(plt.imread(f3name)).transpose(2, 0, 1)
            frameFirst = torch.from_numpy(frameFirst).view(1, 3, 256, 448)
            frameSecond = torch.from_numpy(frameSecond).view(1, 3, 256, 448)
            predicted_img = net(frameFirst.cuda(), frameSecond.cuda())[0, :, :, :]
            plt.imsave(os.path.join(eval_img_dir, video, tri, 'out.png'), predicted_img.permute(1, 2, 0).cpu().detach().numpy())

evaluate('./Dataset/toflow/test/', './evaluate')