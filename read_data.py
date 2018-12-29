import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data

class MemoryFriendlyLoader(torch.utils.data.Dataset):
    def __init__(self, origin_img_dir, pathlistfile, edited_img_dir='', task=''):
        self.origin_img_dir = origin_img_dir
        self.edited_img_dir = edited_img_dir
        self.task = task
        self.pathlist = self.loadpath(pathlistfile)
        self.count = len(self.pathlist)

    def loadpath(self, pathlistfile):
        fp = open(pathlistfile)
        pathlist = fp.read().splitlines()
        fp.close()
        return pathlist

    def __getitem__(self, index):
        frames = []
        path_code = self.pathlist[index]
        if self.task == 'interp':
            N = 2   # 这里的N仅仅是为了下面取framex方便, 并非是论文里的N
            for i in [1, 3]:
                frames.append(plt.imread(os.path.join(self.origin_img_dir, path_code, 'im%d.png' % i)))                  # load the first and third images
            frames.append(plt.imread(os.path.join(self.origin_img_dir, path_code, 'im2.png')))                           # load ground truth (the second one)
        else:
            N = 7
            for i in range(7):
                frames.append(plt.imread(os.path.join(self.edited_img_dir, path_code, 'im%04d.png' % (i + 1))))          # load images with noise.
            frames.append(plt.imread(os.path.join(self.origin_img_dir, path_code, 'im4.png')))                           # load ground truth

        frames = np.array(frames)
        framex = np.transpose(frames[0:N, :, :, :], (0, 3, 1, 2))
        framey = np.transpose(frames[-1, :, :, :], (2, 0, 1))

        return torch.from_numpy(framex), torch.from_numpy(framey), path_code

    def __len__(self):
        return self.count
