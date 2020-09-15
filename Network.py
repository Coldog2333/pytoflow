import math
import torch
# import torch.utils.serialization   # it was removed in torch v1.0.0 or higher version.

arguments_strModel = 'sintel-final'
SpyNet_model_dir = './models'  # The directory of SpyNet's weights

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


Backward_tensorGrid = {}

def Backward(tensorInput, tensorFlow, cuda_flag):
    if str(tensorFlow.size()) not in Backward_tensorGrid:
        tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
        tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))
        if cuda_flag:
            Backward_tensorGrid[str(tensorFlow.size())] = torch.cat([ tensorHorizontal, tensorVertical ], 1).cuda()
        else:
            Backward_tensorGrid[str(tensorFlow.size())] = torch.cat([tensorHorizontal, tensorVertical], 1)
    # end

    tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0) ], 1)

    return torch.nn.functional.grid_sample(input=tensorInput, grid=(Backward_tensorGrid[str(tensorFlow.size())] + tensorFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border')
# end

class SpyNet(torch.nn.Module):
    def __init__(self, cuda_flag):
        super(SpyNet, self).__init__()
        self.cuda_flag = cuda_flag

        class Basic(torch.nn.Module):
            def __init__(self, intLevel):
                super(Basic, self).__init__()

                self.moduleBasic = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
                )

            # end

            def forward(self, tensorInput):
                return self.moduleBasic(tensorInput)

        self.moduleBasic = torch.nn.ModuleList([Basic(intLevel) for intLevel in range(4)])

        self.load_state_dict(torch.load(SpyNet_model_dir + '/network-' + arguments_strModel + '.pytorch'), strict=False)


    def forward(self, tensorFirst, tensorSecond):
        tensorFirst = [tensorFirst]
        tensorSecond = [tensorSecond]

        for intLevel in range(3):
            if tensorFirst[0].size(2) > 32 or tensorFirst[0].size(3) > 32:
                tensorFirst.insert(0, torch.nn.functional.avg_pool2d(input=tensorFirst[0], kernel_size=2, stride=2))
                tensorSecond.insert(0, torch.nn.functional.avg_pool2d(input=tensorSecond[0], kernel_size=2, stride=2))

        tensorFlow = tensorFirst[0].new_zeros(tensorFirst[0].size(0), 2,
                                              int(math.floor(tensorFirst[0].size(2) / 2.0)),
                                              int(math.floor(tensorFirst[0].size(3) / 2.0)))

        for intLevel in range(len(tensorFirst)):
            tensorUpsampled = torch.nn.functional.interpolate(input=tensorFlow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0

            # if the sizes of upsampling and downsampling are not the same, apply zero-padding.
            if tensorUpsampled.size(2) != tensorFirst[intLevel].size(2):
                tensorUpsampled = torch.nn.functional.pad(input=tensorUpsampled, pad=[0, 0, 0, 1], mode='replicate')
            if tensorUpsampled.size(3) != tensorFirst[intLevel].size(3):
                tensorUpsampled = torch.nn.functional.pad(input=tensorUpsampled, pad=[0, 1, 0, 0], mode='replicate')

            # input ：[first picture of corresponding level,
            # 		   the output of w with input second picture of corresponding level and upsampling flow,
            # 		   upsampling flow]
            # then we obtain the final flow. 最终再加起来得到intLevel的flow
            tensorFlow = self.moduleBasic[intLevel](torch.cat([tensorFirst[intLevel],
                                                               Backward(tensorInput=tensorSecond[intLevel],
                                                                        tensorFlow=tensorUpsampled,
                                                                        cuda_flag=self.cuda_flag),
                                                               tensorUpsampled], 1)) + tensorUpsampled
        return tensorFlow


class warp(torch.nn.Module):
    def __init__(self, cuda_flag):
        super(warp, self).__init__()
        self.cuda_flag = cuda_flag

#used code from https://github.com/NVIDIA/vid2vid/blob/master/models/networks.py

    def get_grid(self, batchsize, rows, cols, cuda_flag, dtype=torch.float32):
        hor = torch.linspace(-1.0, 1.0, cols)
        hor.requires_grad = False
        hor = hor.view(1, 1, 1, cols)
        hor = hor.expand(batchsize, 1, rows, cols)
        ver = torch.linspace(-1.0, 1.0, rows)
        ver.requires_grad = False
        ver = ver.view(1, 1, rows, 1)
        ver = ver.expand(batchsize, 1, rows, cols)
    
        t_grid = torch.cat([hor, ver], 1)
        t_grid.requires_grad = False
    
        if cuda_flag:
            return t_grid.cuda()
        else:
            return t_grid

    def forward(self, frame, flow):
        """
        :param frame: frame.shape (batch_size=4, n_channels=3, width=256, height=448)
        :param flow: flow.shape (batch_size=4, n_channels=2, width=256, height=448)
        :return: reference_frame: warped frame
        """
        b, c, h, w = frame.size()        
        grid = self.get_grid(b, h, w, self.cuda_flag, dtype=flow.dtype)            
        flow = torch.cat([flow[:, 0:1, :, :] / ((w - 1.0) / 2.0), flow[:, 1:2, :, :] / ((h - 1.0) / 2.0)], dim=1)        
        final_grid = (grid + flow).permute(0, 2, 3, 1)#.cuda(frame.get_device())
        if self.cuda_flag:
            final_grid = final_grid.cuda()
        output = torch.nn.functional.grid_sample(frame, final_grid)
        return output


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
    def __init__(self, task, cuda_flag):
        super(TOFlow, self).__init__()
        self.task = task
        self.cuda_flag = cuda_flag

        self.SpyNet = SpyNet(cuda_flag=self.cuda_flag)  # SpyNet层
        # for param in self.SpyNet.parameters():  # fix
        #     param.requires_grad = False

        self.warp = warp(cuda_flag=self.cuda_flag)

        self.ResNet = ResNet(task=self.task)

    # frames should be TensorFloat
    def forward(self, frames):
        """
        :param frames: [batch_size=1, img_num, n_channels=3, h, w]
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

        else:
            raise NameError('Only support: [interp, denoise/denoising, sr/super-resolution]')
        B, F, C, H, W = frames.size()
        n_frames = frames.reshape(B*F, C, H, W)
        n_opticalflows=opticalflows.reshape(B*F,2,H,W) #2 as of optical_flow 
        warpframes = self.warp(n_frames, n_opticalflows)
        # warpframes: [batch_size=4, img_num=7, n_channels=3, height=256, width=448]
        warpframes = warpframes.reshape(B, F, C, H, W)

        Img = self.ResNet(warpframes)
        # Img: [batch_size=4, n_channels=3, h, w]

        Img = denormalize(Img)

        return Img

# testing
# net = TOFlow('sr', True).cuda()
# out = net(torch.rand(10,7,3,128,128).cuda())
