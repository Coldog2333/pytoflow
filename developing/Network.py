import os
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.serialization

arguments_strModel = 'F'
# SpyNet_model_dir = './models'  # SpyNet模型参数目录
SpyNet_model_dir = '/home/ftp/Coldog/DeepLearning/TOFlow/branch/models'  # SpyNet模型参数目录


def normalize(tensorInput):
    tensorRed = (tensorInput[:, 0:1, :, :] - 0.485) / 0.229
    tensorGreen = (tensorInput[:, 1:2, :, :] - 0.456) / 0.224
    tensorBlue = (tensorInput[:, 2:3, :, :] - 0.406) / 0.225
    return torch.cat([tensorRed, tensorGreen, tensorBlue], 1)


def denormalize(tensorInput):
    tensorRed = (tensorInput[:, 0:1, :, :] * 0.229) + 0.485
    tensorGreen = (tensorInput[:, 1:2, :, :] * 0.224) + 0.456
    tensorBlue = (tensorInput[:, 2:3, :, :] * 0.225) + 0.406
    return torch.cat([tensorRed, tensorGreen, tensorBlue], 1)


class Basic(torch.nn.Module):
    def __init__(self, intLevel):
        super(Basic, self).__init__()

        # Gk
        self.moduleBasic = torch.nn.Sequential(
            # in_channels=8 是因为RGB*2+flow(3*2+2)=8  flow initialized with zeros.
            torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
            torch.nn.ReLU(inplace=False),
            # inplace – can optionally do the operation in-place. Default: False
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
        )

        # load parameters into the Conv2d Layer
        for intConv in range(5):
            self.moduleBasic[intConv * 2].weight.data.copy_(torch.utils.serialization.load_lua(
                SpyNet_model_dir + '/modelL%d_%s-%d-weight.t7' % (intLevel + 1, arguments_strModel, intConv + 1)))

            self.moduleBasic[intConv * 2].bias.data.copy_(torch.utils.serialization.load_lua(
                SpyNet_model_dir + '/modelL%d_%s-%d-bias.t7' % (intLevel + 1, arguments_strModel, intConv + 1)))
        # end

    # end

    def forward(self, tensorInput):
        return self.moduleBasic(tensorInput)
    # end


class Backward(torch.nn.Module):
    def __init__(self, cuda_flag):
        super(Backward, self).__init__()
        self.cuda_flag = cuda_flag

    def forward(self, tensorInput, tensorFlow):
        # 如果还没定义tensorGrid or tensorGrid的某一维大小与tensorFlow的对应维度大小不一致 的话
        if hasattr(self, 'tensorGrid') == False or \
                self.tensorGrid.size(0) != tensorFlow.size(0) or \
                self.tensorGrid.size(2) != tensorFlow.size(2) or \
                self.tensorGrid.size(3) != tensorFlow.size(3):
            # initialize horizontal flow. 初始化水平flow网格
            tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1,
                                                                                  tensorFlow.size(3)). \
                expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
            # initialize vertical flow. 初始化垂直flow网格
            tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1,
                                                                                tensorFlow.size(2), 1). \
                expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))
            # mix them into a original flow. 组合成初始flow网格
            if self.cuda_flag:
                self.tensorGrid = torch.cat([tensorHorizontal, tensorVertical], 1).cuda()
            else:
                self.tensorGrid = torch.cat([tensorHorizontal, tensorVertical], 1)
        # end

        tensorFlow = torch.cat([tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0),
                                tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0)], 1)

        return torch.nn.functional.grid_sample(input=tensorInput,
                                               grid=(self.tensorGrid + tensorFlow).permute(0, 2, 3, 1),
                                               mode='bilinear', padding_mode='border')
    # end


class SpyNet(torch.nn.Module):
    def __init__(self, cuda_flag):
        super(SpyNet, self).__init__()
        self.cuda_flag = cuda_flag

        # initialize the weight of Gk in 6-layers pyramid. 初始化4层金字塔的Gk的权重
        self.moduleBasic = torch.nn.ModuleList([Basic(intLevel) for intLevel in range(4)])

        self.moduleBackward = Backward(cuda_flag=self.cuda_flag)

    # end

    def forward(self, tensorFirst, tensorSecond):
        tensorFlow = []
        tensorFirst = [tensorFirst]  # apply rgb normalization
        tensorSecond = [tensorSecond]  # apply rgb normalization

        for intLevel in range(3):
            # 最多下采样五次，意味着SpyNet最多6层(只要图片足够大，不小于32×~ or ~×32，不然下采样5次之后就变成一个像素点了)
            # downsample 5 times at most, meaning that SpyNet can be 6 layers at most.
            if tensorFirst[0].size(2) > 32 or tensorFirst[0].size(
                    3) > 32:  # if width and height are smaller than 32, then we won't apply downsampling on it.
                tensorFirst.insert(0, torch.nn.functional.avg_pool2d(input=tensorFirst[0], kernel_size=2,
                                                                     stride=2))  # d:average downsampling
                tensorSecond.insert(0, torch.nn.functional.avg_pool2d(input=tensorSecond[0], kernel_size=2,
                                                                      stride=2))  # d:average downsampling
            # end
        # end
        # 到这里tensorFirst/tensorSecond里就装着[d_5, d_4, d_3, d_2, d_1, 原图]了

        # initialize optical flow, all zero
        tensorFlow = tensorFirst[0].new_zeros(tensorFirst[0].size(0), 2,
                                              int(math.floor(tensorFirst[0].size(2) / 2.0)),
                                              int(math.floor(tensorFirst[0].size(3) / 2.0)))

        for intLevel in range(len(tensorFirst)):  # 循环金字塔level次
            # upsampling, enlarge 2 times, but I don't know why he multiplicative it with 2.0
            tensorUpsampled = torch.nn.functional.interpolate(input=tensorFlow, scale_factor=2, mode='bilinear',
                                                              align_corners=True) * 2.0

            # if the sizes of upsampling and downsampling are not the same, apply zero-padding.
            if tensorUpsampled.size(2) != tensorFirst[intLevel].size(2):
                tensorUpsampled = torch.nn.functional.pad(input=tensorUpsampled, pad=[0, 0, 0, 1],
                                                          mode='replicate')  # mode='replicate' 表示不改变原来的
            if tensorUpsampled.size(3) != tensorFirst[intLevel].size(3):
                tensorUpsampled = torch.nn.functional.pad(input=tensorUpsampled, pad=[0, 1, 0, 0],
                                                          mode='replicate')

            # input 输入网络：[first picture of corresponding level,
            # 			      the output of w with input second picture of corresponding level and upsampling flow,
            # 			      upsampling flow]
            # then we obtain the final flow. 最终再加起来得到intLevel的flow
            tensorFlow = self.moduleBasic[intLevel](torch.cat([tensorFirst[intLevel],
                                                               self.moduleBackward(tensorSecond[intLevel],
                                                                                   tensorUpsampled),
                                                               tensorUpsampled], 1)) + tensorUpsampled
        # end
        return tensorFlow


class warp(torch.nn.Module):
    def __init__(self, h, w, cuda_flag):
        super(warp, self).__init__()
        self.height = h
        self.width = w
        if cuda_flag:
            self.addterm = self.init_addterm().cuda()
        else:
            self.addterm = self.init_addterm()

    def init_addterm(self):
        n = torch.FloatTensor(list(range(self.width)))
        horizontal_term = n.expand((1, 1, self.height, self.width))  # 第一个1是batch size
        n = torch.FloatTensor(list(range(self.height)))
        vertical_term = n.expand((1, 1, self.width, self.height)).permute(0, 1, 3, 2)
        addterm = torch.cat((horizontal_term, vertical_term), dim=1)
        return addterm

    def forward(self, frame, flow):
        """
        :param frame: frame.shape (batch_size=1, n_channels=3, width=256, height=448)
        :param flow: flow.shape (batch_size=1, n_channels=2, width=256, height=448)
        :return: reference_frame: predicted frame
        """
        if True:
            flow = flow + self.addterm
        else:
            self.addterm = self.init_addterm()
            flow = flow + self.addterm

        horizontal_flow = flow[0, 0, :, :].expand(1, 1, self.height, self.width)  # 第一个0是batch size
        vertical_flow = flow[0, 1, :, :].expand(1, 1, self.height, self.width)

        horizontal_flow = horizontal_flow * 2 / (self.width - 1) - 1
        vertical_flow = vertical_flow * 2 / (self.height - 1) - 1
        flow = torch.cat((horizontal_flow, vertical_flow), dim=1)
        flow = flow.permute(0, 2, 3, 1)
        reference_frame = torch.nn.functional.grid_sample(frame, flow)
        return reference_frame


class ResNet(torch.nn.Module):
    """
    Three-layers ResNet/ResBlock
    reference: https://blog.csdn.net/chenyuping333/article/details/82344334
    """

    def __init__(self, task):
        super(ResNet, self).__init__()
        self.task = task
        self.conv_3x2_64_9x9 = torch.nn.Conv2d(in_channels=3 * 2, out_channels=64, kernel_size=9, padding=8 // 2)
        self.conv_3x7_64_9x9 = torch.nn.Conv2d(in_channels=3 * 7, out_channels=64, kernel_size=9, padding=8 // 2)
        self.conv_64_64_9x9 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=9, padding=8 // 2)
        self.conv_64_64_1x1 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        self.conv_64_3_1x1 = torch.nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1)

    def ResBlock(self, x, aver):
        if self.task == 'interp':
            x = torch.nn.functional.relu(self.conv_3x2_64_9x9(x))
            x = torch.nn.functional.relu(self.conv_64_64_1x1(x))
        elif self.task in ['denoise', 'denoising']:
            x = torch.nn.functional.relu(self.conv_3x7_64_9x9(x))
            x = torch.nn.functional.relu(self.conv_64_64_1x1(x))
        elif self.task in ['sr', 'super-resolution']:
            x = torch.nn.functional.relu(self.conv_3x7_64_9x9(x))
            x = torch.nn.functional.relu(self.conv_64_64_9x9(x))
            x = torch.nn.functional.relu(self.conv_64_64_1x1(x))
        else:
            raise NameError('Only support: [interp, denoise/denoising, sr/super-resolution]')
        x = self.conv_64_3_1x1(x) + aver
        return x

    def forward(self, frames):
        aver = frames.mean(dim=1)
        x = frames[:, 0, :, :, :]
        for i in range(1, frames.size(1)):
            x = torch.cat((x, frames[:, i, :, :, :]), dim=1)
        result = self.ResBlock(x, aver)
        return result


class TOFlow(torch.nn.Module):
    def __init__(self, h, w, task, cuda_flag):
        super(TOFlow, self).__init__()
        self.height = h
        self.width = w
        self.task = task
        self.cuda_flag = cuda_flag

        self.SpyNet = SpyNet(cuda_flag=self.cuda_flag)  # SpyNet层
        # for param in self.SpyNet.parameters():  # fix
        #     param.requires_grad = False

        self.warp = warp(self.height, self.width, cuda_flag=self.cuda_flag)

        self.ResNet = ResNet(task=self.task)

    # frameFirst, frameSecond should be TensorFloat
    def forward(self, frames):
        """
        :param frames: the first frame: [batch_size=1, n_channels=3, h, w]
        :return:
        """
        for i in range(frames.size(1)):
            frames[:, i, :, :, :] = normalize(frames[:, i, :, :, :])

        if self.cuda_flag:
            opticalflows = torch.zeros(frames.size(0), frames.size(1), 2, frames.size(3), frames.size(4)).cuda()
            warpframes = torch.empty(frames.size(0), frames.size(1), 3, frames.size(3), frames.size(4)).cuda()
        else:
            opticalflows = torch.zeros(frames.size(0), frames.size(1), 2, frames.size(3), frames.size(4))
            warpframes = torch.empty(frames.size(0), frames.size(1), 3, frames.size(3), frames.size(4))

        if self.task == 'interp':
            process_index = [0, 1]
            opticalflows[:, 1, :, :, :] = self.SpyNet(frames[:, 0, :, :, :], frames[:, 1, :, :, :]) / 2
            opticalflows[:, 0, :, :, :] = self.SpyNet(frames[:, 1, :, :, :], frames[:, 0, :, :, :]) / 2
        elif self.task in ['denoise', 'denoising', 'sr', 'super-resolution']:
            process_index = [0, 1, 2, 4, 5, 6]
            for i in process_index:
                opticalflows[:, i, :, :, :] = self.SpyNet(frames[:, 3, :, :, :], frames[:, i, :, :, :])
            warpframes[:, 3, :, :, :] = frames[:, 3, :, :, :]
        # opticalflow: [batch_size=1, n_channels=2, h, w]
        else:
            raise NameError('Only support: [interp, denoise/denoising, sr/super-resolution]')

        for i in process_index:
            warpframes[:, i, :, :, :] = self.warp(frames[:, i, :, :, :], opticalflows[:, i, :, :, :])
        # warpframes: [batch_size=1, img_num=7, n_channels=3, height=256, width=448]

        Img = self.ResNet(warpframes)
        # Img: [batch_size=1, n_channels=3, h, w]

        Img = denormalize(Img)

        return Img
