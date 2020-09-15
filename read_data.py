import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data
from PIL import Image
import torchvision.transforms.functional as F

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
    
    def concat(self, frames, frame_s):
        if frames is None:
            frames = frame_s.unsqueeze(0)
        else:
            frames = torch.cat([frames, frame_s.unsqueeze(0)], dim = 0)
        return frames

    def __getitem__(self, index):
        frames = None
        path_code = self.pathlist[index]
        if self.task == 'interp':
            N = 2   # 这里的N仅仅是为了下面取framex方便, 并非是论文里的N
            for i in [1, 3]:
                frame_s = F.to_tensor(Image.open(os.path.join(self.origin_img_dir, path_code, 'im{}.png'.format(i))))                # load the first and third images
                frames = self.concat(frames,frame_s)
            frame_y = F.to_tensor(Image.open(os.path.join(self.origin_img_dir, path_code, 'im2.png')))                          # load ground truth (the second one)
        else:
            N = 7
            for i in range(7):
                frame_s = F.to_tensor(Image.open(os.path.join(self.edited_img_dir, path_code, 'im{:04d}.png'.format(i + 1))))          # load images with noise.
                frames = self.concat(frames,frame_s)
            frame_y = F.to_tensor(Image.open(os.path.join(self.origin_img_dir, path_code, 'im4.png')))                           # load ground truth

        return frames, frame_y, path_code

    def __len__(self):
        return self.count
    

    
#### Memory Friendly
# train_list = './tiny/vimeo_septuplet/sep_trainlist.txt'
# root = './tiny/vimeo_septuplet/sequences'
# data_set = MemoryFriendlyLoader(root , train_list, root)

# data_loader = DataLoader(data_set,batch_size=10)#, shuffle=True)

# for i, (lr,hr, frame_name) in enumerate(data_loader):
#     print(i, lr.shape, hr.shape)

