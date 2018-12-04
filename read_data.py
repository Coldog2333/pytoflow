import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data

class MemoryFriendlyLoader(torch.utils.data.Dataset):
    def __init__(self, rootdir):
        self.rootdir = rootdir
        self.pathlist = self.loadpath()
        self.count = len(self.pathlist)

    def loadpath(self):
        pathlist = []
        for video in os.listdir(self.rootdir):
            for triplet in os.listdir(os.path.join(self.rootdir, video)):
                pathlist.append(os.path.join(self.rootdir, video, triplet))
        return pathlist

    def __getitem__(self, index):
        frames = []
        # imgs = os.listdir(self.pathlist[index])
        # frames.append(plt.imread(os.path.join(self.pathlist[index], imgs[0])))
        # frames.append(plt.imread(os.path.join(self.pathlist[index], imgs[2])))
        # frames.append(plt.imread(os.path.join(self.pathlist[index], imgs[1])))
        # print(imgs[0],imgs[2],imgs[1])  # 发现是im2.png im1.png im3.png的顺序，我也不懂为什么会这样。。。
        frames.append(plt.imread(os.path.join(self.pathlist[index], 'im1.png')))
        frames.append(plt.imread(os.path.join(self.pathlist[index], 'im3.png')))
        frames.append(plt.imread(os.path.join(self.pathlist[index], 'im2.png')))

        frames = np.array(frames)  # (sample_num, img_num, height, width, nchannels) sample_num指图片组数
        framex = np.transpose(frames[0:2, :, :, :], (0, 3, 1, 2))
        framey = np.transpose(frames[-1, :, :, :], (2, 0, 1))

        if self.pathlist[index] == os.path.join(self.rootdir, '00010', '0060'):    # 仍以00010/0060为例子，用于可视化作图
            flag = True
        else:
            flag = False

        return torch.from_numpy(framex), torch.from_numpy(framey), flag

    def __len__(self):
        return self.count