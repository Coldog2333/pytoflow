import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data

# noise
class MemoryFriendlyLoader(torch.utils.data.Dataset):
    def __init__(self, origin_dir, noise_dir, codelistfile):
        self.origin_dir = origin_dir
        self.noise_dir = noise_dir
        self.pathlist = self.loadpath(codelistfile)
        self.count = len(self.pathlist)

    def loadpath(self, codelistfile):
        fp = open(codelistfile)
        pathlist = fp.read().splitlines()
        fp.close()
        return pathlist

    def __getitem__(self, index):
        frames = []
        for i in range(7):
            frames.append(plt.imread(os.path.join(self.noise_dir, self.pathlist[index], 'im%04d.png' % (i+1))))
        frames.append(plt.imread(os.path.join(self.origin_dir, self.pathlist[index], 'im4.png')))

        frames = np.array(frames)  # (sample_num, img_num, height, width, nchannels) sample_num指图片组数
        framex = np.transpose(frames[0:7, :, :, :], (0, 3, 1, 2))
        framey = np.transpose(frames[-1, :, :, :], (2, 0, 1))

        if self.pathlist[index] == '00004/0357':
            pltflag = True
        else:
            pltflag = False

        return torch.from_numpy(framex), torch.from_numpy(framey), pltflag

    def __len__(self):
        return self.count