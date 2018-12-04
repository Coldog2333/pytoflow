import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.serialization
import math
from Network import TOFlow
import matplotlib.pyplot as plt

# ------------------------------
torch.cuda.set_device(0)
# Static
model_name = 'toflow_1_8_2'                                                       # select model
workplace = '/home/ftp/Coldog/DeepLearning/TOFlow/branch'
test_pic_dir = '/home/ftp/Coldog/Dataset/toflow/train/00010/0060'    # triple frames dir

# ------------------------------
# 数据集中的图片长宽都弄成32的倍数了所以这里可以不用这个函数
# 暂时只用于处理batch_size = 1的triple
def Estimate(net, tensorFirst, tensorSecond):
    """
    :param tensorFirst: 弄成FloatTensor格式的frameFirst
    :param tensorSecond: 弄成FloatTensor格式的frameSecond
    :return:
    """
    tensorOutput = torch.FloatTensor()

    # check whether the two frames have the same shape
    assert (tensorFirst.size(1) == tensorSecond.size(1))
    assert (tensorFirst.size(2) == tensorSecond.size(2))

    intWidth = tensorFirst.size(2)
    intHeight = tensorFirst.size(1)

    assert(intWidth == 448) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
    assert(intHeight == 256) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

    if True:
        tensorFirst = tensorFirst.cuda()
        tensorSecond = tensorSecond.cuda()
        tensorOutput = tensorOutput.cuda()
    # end

    if True:
        tensorPreprocessedFirst = tensorFirst.view(1, 3, intHeight, intWidth)
        tensorPreprocessedSecond = tensorSecond.view(1, 3, intHeight, intWidth)

        intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))  # 宽度弄成32的倍数，便于上下采样
        intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))  # 长度弄成32的倍数，便于上下采样

        tensorPreprocessedFirst = torch.nn.functional.interpolate(input=tensorPreprocessedFirst, size=(
            intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
        tensorPreprocessedSecond = torch.nn.functional.interpolate(input=tensorPreprocessedSecond, size=(
            intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

        tensorFlow = torch.nn.functional.interpolate(
            input=net(tensorPreprocessedFirst, tensorPreprocessedSecond),
            size=(intHeight, intWidth), mode='bilinear', align_corners=False)

        tensorOutput.resize_(3, intHeight, intWidth).copy_(tensorFlow[0, :, :, :])
    # end

    if True:
        tensorFirst = tensorFirst.cpu()
        tensorSecond = tensorSecond.cpu()
        tensorOutput = tensorOutput.cpu()
    # end
    return tensorOutput


def generate(net, model_name, f1name, f2name, fname='', save=True):
    if fname == '':
        fname = os.path.join(os.path.split(f1name)[0], 'predicted_frame' + f1name.split('.')[-1])
    # generate之前要把网络弄成eval()
    print('Reading frames... ', end='')
    frame1 = plt.imread(f1name)
    frame2 = plt.imread(f2name)
    print('Done.')
    print('Processing...')
    frame1 = torch.FloatTensor(frame1)
    frame2 = torch.FloatTensor(frame2)

    frame1 = frame1.view(1, frame1.shape[0], frame1.shape[1], frame1.shape[2])
    frame2 = frame2.view(1, frame2.shape[0], frame2.shape[1], frame2.shape[2])
    frame1 = frame1.permute(0, 3, 1, 2)
    frame2 = frame2.permute(0, 3, 1, 2)

    # frame2 = Estimate(net, frame1, frame3)
    frame1 = frame1.cuda()
    frame2 = frame2.cuda()
    frame = net(frame1, frame2)

    frame = frame[0, :, :, :].permute(1, 2, 0)
    frame = frame.cpu()
    frame = frame.detach().numpy()

    if save==True:
        print('Done.\nSaving predicted frame in %s' % fname)
        plt.imsave(fname, frame)
        print('All done.')  # save frame
    else:
        print('All done.')  # save frame
        return frame

# ------------------------------
print('Loading TOFlow Net... ', end='')
net = TOFlow(256, 448)
net.load_state_dict(torch.load(os.path.join('.', 'toflow_models', model_name + '_params.pkl')))
net.cuda().eval()
print('Done.')
# ------------------------------
generate(net=net, model_name=model_name, f1name=os.path.join(test_pic_dir, 'im1.png'),
         f2name=plt.imread(os.path.join(test_pic_dir, 'im3.png')))
# print('Reading frames... ', end='')
# frame1 = plt.imread(os.path.join(test_pic_dir, 'im1.png'))
# frame3 = plt.imread(os.path.join(test_pic_dir, 'im3.png'))
# print('Done.')
# print('Processing...')
# frame1 = torch.FloatTensor(frame1)
# frame3 = torch.FloatTensor(frame3)
#
# frame1 = frame1.view(1, frame1.shape[0], frame1.shape[1], frame1.shape[2])
# frame3 = frame3.view(1, frame3.shape[0], frame3.shape[1], frame3.shape[2])
# frame1 = frame1.permute(0, 3, 1, 2)
# frame3 = frame3.permute(0, 3, 1, 2)
#
# # frame2 = Estimate(net, frame1, frame3)
# frame1 = frame1.cuda()
# frame3 = frame3.cuda()
# frame2 = net(frame1, frame3)
#
# frame2 = frame2[0,:,:,:].permute(1,2,0)
# frame2 = frame2.cpu()
# frame2 = frame2.detach().numpy()
# print('Done.\nSaving predicted frame in ./predicted_frame2_00010_0060.jpg')
# plt.imsave('./predicted_frame2_00010_0060.jpg', frame2)
# print('All done.')# save frame