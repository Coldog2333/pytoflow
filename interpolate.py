import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.serialization
import math
import PIL
from Network import TOFlow
import matplotlib.pyplot as plt
import utils
import shutil
import datetime

# ------------------------------
# torch.cuda.set_device(0)
# Static
model_name = 'toflow_1_9'  # select model
workplace = '/home/ftp/Coldog/DeepLearning/TOFlow/branch/v0.3.1'
img_dir = os.path.join(workplace, 'mini')

# ------------------------------
# 数据集中的图片长宽都弄成32的倍数了所以这里可以不用这个函数
# 暂时只用于处理batch_size = 1的triple
def Estimate(net, tensorFirst=None, tensorSecond=None, Firstfilename='', Secondfilename='', cuda=True):
    """
    :param tensorFirst: 弄成FloatTensor格式的frameFirst
    :param tensorSecond: 弄成FloatTensor格式的frameSecond
    :return:
    """
    if Firstfilename and Secondfilename:
        tensorFirst = torch.FloatTensor(
            np.array(PIL.Image.open(Firstfilename).convert("RGB")).transpose(2, 0, 1).astype(np.float32) * (
                    1.0 / 255.0))
        tensorSecond = torch.FloatTensor(
            np.array(PIL.Image.open(Secondfilename).convert("RGB")).transpose(2, 0, 1).astype(np.float32) * (
                    1.0 / 255.0))
        # tensorFirst = torch.FloatTensor(np.array(plt.imread(Firstfilename)).transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0))
        # tensorSecond = torch.FloatTensor(np.array(plt.imread(Secondfilename)).transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0))
        # print(tensorFirst.shape)

    tensorOutput = torch.FloatTensor()

    # check whether the two frames have the same shape
    assert (tensorFirst.size(1) == tensorSecond.size(1))
    assert (tensorFirst.size(2) == tensorSecond.size(2))

    intWidth = tensorFirst.size(2)
    intHeight = tensorFirst.size(1)
    # print(intWidth, intHeight)

    # assert(intWidth == 448) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
    # assert(intHeight == 256) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

    if cuda == True:
        tensorFirst = tensorFirst.cuda()
        tensorSecond = tensorSecond.cuda()
        tensorOutput = tensorOutput.cuda()
    # end

    if True:
        # print(tensorFirst.shape)
        tensorPreprocessedFirst = tensorFirst.view(1, 3, intHeight, intWidth)
        tensorPreprocessedSecond = tensorSecond.view(1, 3, intHeight, intWidth)

        intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))  # 宽度弄成32的倍数，便于上下采样
        intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))  # 长度弄成32的倍数，便于上下采样

        tensorPreprocessedFirst = torch.nn.functional.interpolate(input=tensorPreprocessedFirst, size=(
            intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
        tensorPreprocessedSecond = torch.nn.functional.interpolate(input=tensorPreprocessedSecond, size=(
            intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

        # print(tensorPreprocessedFirst.shape)
        tensorFlow = torch.nn.functional.interpolate(
            input=net(tensorPreprocessedFirst, tensorPreprocessedSecond, 0, 0),
            size=(intHeight, intWidth), mode='bilinear', align_corners=False)

        tensorOutput.resize_(3, intHeight, intWidth).copy_(tensorFlow[0, :, :, :])
        tensorOutput = tensorOutput.permute(1, 2, 0)
    # end

    if True:
        tensorFirst = tensorFirst.cpu()
        tensorSecond = tensorSecond.cpu()
        tensorOutput = tensorOutput.cpu()
    # end
    return tensorOutput.detach().numpy()


def Estimate_Imgs(net, img_dir, plus=2):
    count = 0
    for p in range(plus):
        imgnum = len(os.listdir(img_dir))
        for i in range(imgnum, 1, -1):
            f1name = os.path.join(img_dir, '%06d.png' % (i - 1))
            f2name = os.path.join(img_dir, '%06d.png' % i)
            predicted_img = Estimate(net, Firstfilename=f1name, Secondfilename=f2name)
            shutil.move(f2name, os.path.join(img_dir, '%06d.png' % (i * 2 - 1)))
            plt.imsave(os.path.join(img_dir, '%06d.png' % (i * 2 - 2)), predicted_img)
            count += 1
            if count // 100 == count / 100:
                print('  Processed %f%%' % (count / imgnum / (plus - 1) * 100))


def interpolate(original_video, output_video, plus=2):
    audio_format = 'wav'
    img_save_dir = os.path.join('extracted_frames', os.path.split(original_video)[-1].split('.')[0])
    if not os.path.exists('extracted_frames'):
        os.mkdir('extracted_frames')
        print('mkdir ./extracted_frames')
    if not os.path.exists(img_save_dir):
        os.mkdir(img_save_dir)
        print('mkdir %s' % img_save_dir)

    # extract original video
    print('Extracting video frames...')
    try:
        fps = utils.fast_extract_video(original_video, save_dir=img_save_dir)
    except Exception as e:
        print(e)
        print('  failed to use fast_version.')
        fps = utils.extract_video(original_video, save_dir=img_save_dir, stride=1,
                                  isoutput=False, extract_music=True,
                                  music_name='.'.join(original_video.split('.')[:-1]) + '.' + audio_format)

    # render 渲染
    temp_img = np.array(plt.imread(os.path.join(img_save_dir, '000001.png')))
    height = temp_img.shape[0]
    width = temp_img.shape[1]

    intPreprocessedWidth = int(math.floor(math.ceil(width / 32.0) * 32.0))  # 宽度弄成32的倍数，便于上下采样
    intPreprocessedHeight = int(math.floor(math.ceil(height / 32.0) * 32.0))  # 长度弄成32的倍数，便于上下采样

    print('Loading TOFlow Net... ', end='')
    net = TOFlow(intPreprocessedHeight, intPreprocessedWidth)
    net.load_state_dict(torch.load(os.path.join('.', 'toflow_models', model_name + '_params.pkl')))
    net.cuda().eval()
    print('Done.')

    print('Interpolating...')
    Estimate_Imgs(net, os.path.join('extracted_frames', os.path.split(original_video)[-1].split('.')[0]))
    print('Done.')
    print('Generating video...')

    # generate new video
    try:
        utils.fast_extract_video(img_dir=img_save_dir,
                                 music='.'.join(original_video.split('.')[:-1]) + '.' + audio_format,
                                 fps=fps * plus, output_video=output_video)
    except:
        print('  failed to use fast_version.')
        utils.imgs2video(imgdir=img_save_dir, video_no_audio='',
                         combine_music=True, music_name='.'.join(original_video.split('.')[:-1]) + '.' + audio_format,
                         video_with_audio=output_video, fps=fps * plus)

    print('All done.')
    return 0


# ------------------------------
pre_time = datetime.datetime.now()
# if __name__ == '__main__':
# interpolate(r'F:\TOFlow\test.mp4', './new.mp4')
interpolate(r'/home/ftp/Coldog/DeepLearning/TOFlow/branch/v0.3.1/test.mp4', './new.mp4', plus=3)
cur_time = datetime.datetime.now()
h, remaind = divmod((cur_time - pre_time).seconds, 3600)
m, s = divmod(remaind, 60)
print('Cost %d:%d:%d' % (h, m, s))
