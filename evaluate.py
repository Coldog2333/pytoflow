import torch
import numpy as np
import sys
import getopt
import os
import shutil
import matplotlib.pyplot as plt
import datetime
from Network import TOFlow
import warnings
warnings.filterwarnings("ignore", module="matplotlib.pyplot")
# ------------------------------
# I don't know whether you have a GPU.
plt.switch_backend('agg')
# Static
task = ''
dataset_dir = ''
pathlistfile = ''
model_path = ''
gpuID = None

if sys.argv[1] in ['-h', '--help']:
    print("""pytoflow version 1.0
usage: python3 train.py [[option] [value]]...
options:
--task         training task, like interp, denoising, super-resolution
               valid values:[interp, denoise, denoising, sr, super-resolution]
--dataDir      the directory of the input image dataset(Vimeo-90K, Vimeo-90K with noise, blurred Vimeo-90K)
--pathlist     the text file records which are the images for train.
--model        the path of the model used.
--gpuID        the No. of the GPU you want to use.
--help         get help.""")
    exit(0)

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [strParameter[2:] + '=' for strParameter in sys.argv[1::2]])[0]:
    if strOption == '--task':           # task
        task = strArgument
    elif strOption == '--dataDir':      # dataset_dir
        dataset_dir = strArgument
    elif strOption == '--pathlist':     # pathlist file
        pathlistfile = strArgument
    elif strOption == '--model':        # model path
        model_path = strArgument
    elif strOption == '--gpuID':        # gpu id
        gpuID = int(strArgument)

if task == '':
    raise ValueError('Missing [--task].\nPlease enter the training task.')
elif task not in ['interp', 'denoise', 'denoising', 'sr', 'super-resolution']:
    raise ValueError('Invalid [--task].\nOnly support: [interp, denoise/denoising, sr/super-resolution]')

if dataset_dir == '':
    raise ValueError('Missing [--dataDir].\nPlease provide the directory of the dataset. (Vimeo-90K)')

if pathlistfile == '':
    raise ValueError('Missing [--pathlist].\nPlease provide the pathlist index file for test.')

if model_path == '':
    raise ValueError('Missing [--model model_path].\nPlease provide the path of the toflow model.')

if gpuID == None:
    cuda_flag = False
else:
    cuda_flag = True
    torch.cuda.set_device(gpuID)
# --------------------------------------------------------------

def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)

def vimeo_evaluate(input_dir, out_img_dir, test_codelistfile, task='', cuda_flag=True):
    mkdir_if_not_exist(out_img_dir)

    net = TOFlow(256, 448, cuda_flag=cuda_flag, task=task)
    net.load_state_dict(torch.load(model_path))

    if cuda_flag:
        net.cuda().eval()
    else:
        net.eval()

    fp = open(test_codelistfile)
    test_img_list = fp.read().splitlines()
    fp.close()

    if task == 'interp':
        process_index = [1, 3]
        str_format = 'im%d.png'
    elif task in ['interp', 'denoise', 'denoising', 'sr', 'super-resolution']:
        process_index = [1, 2, 3, 4, 5, 6, 7]
        str_format = 'im%04d.png'
    else:
        raise ValueError('Invalid [--task].\nOnly support: [interp, denoise/denoising, sr/super-resolution]')
    total_count = len(test_img_list)
    count = 0

    pre = datetime.datetime.now()
    for code in test_img_list:
        # print('Processing %s...' % code)
        count += 1
        video = code.split('/')[0]
        sep = code.split('/')[1]
        mkdir_if_not_exist(os.path.join(out_img_dir, video))
        mkdir_if_not_exist(os.path.join(out_img_dir, video, sep))
        input_frames = []
        for i in process_index:
            input_frames.append(plt.imread(os.path.join(input_dir, code, str_format % i)))
        input_frames = np.transpose(np.array(input_frames), (0, 3, 1, 2))

        if cuda_flag:
            input_frames = torch.from_numpy(input_frames).cuda()
        else:
            input_frames = torch.from_numpy(input_frames)
        input_frames = input_frames.view(1, input_frames.size(0), input_frames.size(1), input_frames.size(2), input_frames.size(3))
        predicted_img = net(input_frames)[0, :, :, :]
        plt.imsave(os.path.join(out_img_dir, video, sep, 'out.png'),predicted_img.permute(1, 2, 0).cpu().detach().numpy())

        cur = datetime.datetime.now()
        processing_time = (cur - pre).seconds / count
        print('%.2fs per frame.\t%.2fs left.' % (processing_time, processing_time * (total_count - count)))

vimeo_evaluate(dataset_dir, './evaluate', pathlistfile, task=task, cuda_flag=cuda_flag)