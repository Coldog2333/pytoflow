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
        frames.append(plt.imread(os.path.join(self.pathlist[index], 'im1.png')))
        frames.append(plt.imread(os.path.join(self.pathlist[index], 'im3.png')))
        frames.append(plt.imread(os.path.join(self.pathlist[index], 'im2.png')))

        frames = np.array(frames)  # (sample_num, img_num, height, width, nchannels) sample_num指图片组数
        framex = np.transpose(frames[0:2, :, :, :], (0, 3, 1, 2))
        framey = np.transpose(frames[-1, :, :, :], (2, 0, 1))

        return torch.from_numpy(framex), torch.from_numpy(framey)

    def __len__(self):
        return self.count