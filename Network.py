import math
import torch
import torch.nn as nn
import torch.utils.serialization

arguments_strModel = 'F'        # SpyNet - Sintel final
SpyNet_model_dir = './models/'  # SpyNet模型参数目录

class TOFlow(torch.nn.Module):
    def __init__(self, h, w, cuda):
        super(TOFlow, self).__init__()
        self.height = h
        self.width = w
        self.cuda = cuda

        class SpyNet(torch.nn.Module):
            def __init__(self, cuda):
                super(SpyNet, self).__init__()
                self.cuda = cuda

                class Preprocess(torch.nn.Module):
                    def __init__(self):
                        super(Preprocess, self).__init__()

                    # end

                    # RGB normalization, but I forgot why we should do so.
                    # tensorInput (batch_size, n_channel, width, height)
                    def forward(self, tensorInput):
                        tensorBlue = (tensorInput[:, 0:1, :, :] - 0.406) / 0.225
                        tensorGreen = (tensorInput[:, 1:2, :, :] - 0.456) / 0.224
                        tensorRed = (tensorInput[:, 2:3, :, :] - 0.485) / 0.229

                        return torch.cat([tensorRed, tensorGreen, tensorBlue], 1)
                        # restore them as beginning after normalization. 标准化后拼回来原来的样子
                    # end

                # end

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
                                SpyNet_model_dir + '/modelL' + str(intLevel + 1) + '_' + 'F' + '-' + str(
                                    intConv + 1) + '-weight.t7'))
                            self.moduleBasic[intConv * 2].bias.data.copy_(torch.utils.serialization.load_lua(
                                SpyNet_model_dir + '/modelL' + str(intLevel + 1) + '_' + 'F' + '-' + str(
                                    intConv + 1) + '-bias.t7'))
                        # end

                    # end

                    def forward(self, tensorInput):
                        return self.moduleBasic(tensorInput)
                    # end

                # end

                class Backward(torch.nn.Module):
                    def __init__(self, cuda):
                        super(Backward, self).__init__()
                        self.cuda = cuda
                    # end

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
                            if self.cuda:
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

                # end

                self.modulePreprocess = Preprocess()

                # initialize the weight of Gk in 6-layers pyramid. 初始化6层金字塔的Gk的权重
                self.moduleBasic = torch.nn.ModuleList([Basic(intLevel) for intLevel in range(6)])

                self.moduleBackward = Backward(cuda=self.cuda)

            # end

            def forward(self, tensorFirst, tensorSecond):
                tensorFlow = []
                tensorFirst = [self.modulePreprocess(tensorFirst)]  # apply rgb normalization
                tensorSecond = [self.modulePreprocess(tensorSecond)]  # apply rgb normalization

                for intLevel in range(5):
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

        class STN(torch.nn.Module):
            def __init__(self):
                super(STN, self).__init__()
                self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5)
                self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
                self.conv2_drop = nn.Dropout2d()

                # Spatial transformer localization-network
                self.localization = nn.Sequential(
                    nn.Conv2d(in_channels=3, out_channels=8, kernel_size=7),
                    nn.MaxPool2d(2, stride=2),
                    nn.ReLU(True),
                    nn.Conv2d(in_channels=8, out_channels=10, kernel_size=5),
                    nn.MaxPool2d(2, stride=2),
                    nn.ReLU(True)
                )

                # Regressor for the 3 * 2 affine matrix
                self.fc_loc = nn.Sequential(
                    nn.Linear(in_features=10 * 60 * 108, out_features=32),
                    nn.ReLU(True),
                    nn.Linear(in_features=32, out_features=3 * 2)
                )

                # Initialize the weights/bias with identity transformation
                self.fc_loc[2].weight.data.fill_(0)
                self.fc_loc[2].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

            # Spatial transformer network forward function
            def stn(self, x):
                xs = self.localization(x)
                xs = torch.nn.functional.interpolate(input=xs, size=(60, 108), mode='bilinear', align_corners=False)
                xs = xs.view(-1, 10 * 60 * 108)
                theta = self.fc_loc(xs)
                theta = theta.view(-1, 2, 3)

                grid = torch.nn.functional.affine_grid(theta, x.size())
                x = torch.nn.functional.grid_sample(x, grid)

                return x

            def forward(self, x):
                # transform the input
                x = self.stn(x)
                return x

        class warp(torch.nn.Module):
            def __init__(self, h, w, cuda=True):
                super(warp, self).__init__()
                self.height = h
                self.width = w
                if cuda:
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
                # if self.addterm:
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
            def __init__(self):
                super(ResNet, self).__init__()
                self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=8 // 2)
                self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
                self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1)

            def initialize(self, layer):
                if isinstance(layer, torch.nn.Conv2d):
                    torch.nn.init.normal_(layer.weight.data, 0, 0.1)
                    torch.nn.init.constant_(layer.bias.data, 0)

            def ResBlock(self, x):
                Fx = torch.nn.functional.relu(self.conv1(x))
                Fx = torch.nn.functional.relu(self.conv2(Fx))
                Fx = torch.nn.functional.relu(self.conv3(Fx) + x)
                return Fx

            def forward(self, tframe1, tframe2):
                aver = (tframe1 + tframe2) / 2
                result = self.ResBlock(aver)
                return result


        self.SpyNet = SpyNet(cuda=self.cuda)  # SpyNet层
        # for param in self.SpyNet.parameters():  # fix
        #     param.requires_grad = False

        self.STN = STN()  # STN层

        self.warp = warp(self.height, self.width, cuda=self.cuda)

        self.ResNet = ResNet()
        # self.ResNet.apply(self.ResNet.initialize)

    # frameFirst, frameSecond should be TensorFloat
    def forward(self, frameFirst, frameSecond):
        """
        :param frameFirst: the first frame
        :param frameSecond: the second frame
        :return:
        """
        opticalflow1 = self.SpyNet(frameFirst, frameSecond)
        opticalflow2 = self.SpyNet(frameSecond, frameFirst)
        # opticalflow: [batch_size=1, n_channels=2, h, w]
        opticalflow1 /= 2
        opticalflow2 /= 2
        warpframeFirst = self.STN(frameFirst)
        warpframeSecond = self.STN(frameSecond)
        # warpframe: [batch_size=1, n_channels=3, h, w]
        warp1 = self.warp(warpframeFirst, opticalflow2)
        warp2 = self.warp(warpframeSecond, opticalflow1)
        # warp: [batch_size=1, n_channels=3, h, w]
        Img = self.ResNet(warp1, warp2)
        # Img: [batch_size=1, n_channels=3, h, w]

        return Img
