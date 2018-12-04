import os
import datetime
import torch
import matplotlib.pyplot as plt
import multiprocessing
import psutil
from Network import TOFlow
from read_data import MemoryFriendlyLoader

# --------------------------------------------------------------
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
torch.cuda.set_device(0)
# --------------------------------------------------------------
# Note:
# 确认Static变量是否试自己想要的
# Hyper Parameters
EPOCH = 15
LR = 3 * 1e-4  # for interpolation, this is the original learning rate.
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 1
LR_strategy = []  # [10, 50]
h = 256
w = 448
# Static
work_place = '.'
model_name = 'toflow_1_8_visualize'  # 需要修改
Training_pic_path = 'Training_result.jpg'
model_information_txt = model_name + '_information.txt'
# train_dir = '/home/ftp/Coldog/Dataset/toflow/train/'   # Linux training dataset path
train_dir = '/home/ftp/Coldog/DeepLearning/TOFlow/tiny/train'  # tiny training dataset path
# train_dir = '/home/ftp/Coldog/DeepLearning/TOFlow/single/train'  # single training dataset path
# --------------------------------------------------------------
# prepare DataLoader
Dataset = MemoryFriendlyLoader(rootdir=train_dir)
train_loader = torch.utils.data.DataLoader(dataset=Dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True,
                                           num_workers=0)
sample_size = Dataset.count

# --------------------------------------------------------------
# some functions
def show_time(now):
    s = str(now.year) + '/' + str(now.month) + '/' + str(now.day) + ' ' \
        + '%02d' % now.hour + ':' + '%02d' % now.minute + ':' + '%02d' % now.second
    return s


def delta_time(datetime1, datetime2):
    if datetime1 > datetime2:
        datetime1, datetime2 = datetime2, datetime1
    second = 0
    # second += (datetime2.year - datetime1.year) * 365 * 24 * 3600
    # second += (datetime2.month - datetime1.month) * 30 * 24 * 3600
    second += (datetime2.day - datetime1.day) * 24 * 3600
    second += (datetime2.hour - datetime1.hour) * 3600
    second += (datetime2.minute - datetime1.minute) * 60
    second += (datetime2.second - datetime1.second)
    return second


# def generate(net, model_name, f1=None, f2=None, f1name='', f2name='', fname='', save=True):
#     if fname == '':
#         fname = os.path.join(os.path.split(f1name)[0], 'predicted_frame' + f1name.split('.')[-1])
#     # generate之前要把网络弄成eval()
#     net.eval()
#     if f1name and f2name:
#         print('Reading frames... ', end='')
#         f1 = plt.imread(f1name)
#         f2 = plt.imread(f2name)
#         print('Done.')
#     print('Processing...')
#     frame1 = torch.FloatTensor(f1)
#     frame2 = torch.FloatTensor(f2)
#
#     frame1 = frame1.view(1, frame1.shape[0], frame1.shape[1], frame1.shape[2])
#     frame2 = frame2.view(1, frame2.shape[0], frame2.shape[1], frame2.shape[2])
#     frame1 = frame1.permute(0, 3, 1, 2)
#     frame2 = frame2.permute(0, 3, 1, 2)
#
#     # frame2 = Estimate(net, frame1, frame3)
#     frame1 = frame1.cuda()
#     frame2 = frame2.cuda()
#     frame = net(frame1, frame2)
#
#     frame = frame[0, :, :, :].permute(1, 2, 0)
#     frame = frame.cpu()
#     frame = frame.detach().numpy()
#
#     if save==True:
#         print('Done.\nSaving predicted frame in %s' % fname)
#         plt.imsave(fname, frame)
#         print('All done.')  # save frame
#         net.train()
#     else:
#         print('All done.')  # save frame
#         net.train()
#         return frame
#
#
# def visualization(net, model_name, f1, f2):
#     frame = generate(net, model_name=model_name, f1=f1, f2=f2, save=False)

# --------------------------------------------------------------

toflow = TOFlow(h, w).cuda()

spynet_params = list(map(id, toflow.SpyNet.parameters()))
other_params = filter(lambda p: id(p) not in spynet_params, toflow.parameters())
optimizer = torch.optim.Adam([
    {'params': other_params},
    {'params': toflow.SpyNet.parameters(), 'lr': LR/3}
    ], lr=LR, weight_decay=WEIGHT_DECAY)
# optimizer = torch.optim.Adam(toflow.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
# loss_func = torch.nn.MSELoss()
loss_func = torch.nn.L1Loss()

# Training
prev_time = datetime.datetime.now()  # current time
print('%s  Start training...' % show_time(prev_time))
plotx = []
ploty = []

min_loss = 1

for epoch in range(EPOCH):
    losses = 0
    count = 0
    for step, (x, y, pltflag) in enumerate(train_loader):
        # x (batch_size, img_num=2, height, width, nchannels)
        # y (batch_size, height, width, nchannels)
        frameFirst = x[:, 0, :, :, :].cuda()  # 弄成循环读取的两帧
        frameSecond = x[:, 1, :, :, :].cuda()
        y = y.expand((1, 3, 256, 448))
        reference = y.cuda()

        prediction = toflow(frameFirst, frameSecond, pltflag, epoch)
        prediction = prediction.cuda()
        loss = loss_func(prediction, reference)
        if pltflag==True:
            plt.figure(1)
            plt.imshow(reference[0, :, :, :].permute(1, 2, 0).cpu().detach().numpy())
            plt.axis('off')
            plt.savefig('reference.jpg', dpi=200)
            plt.close(1)

        # losses += loss                # the reason why oom happened
        losses += loss.item()
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        count += len(x)
        if count / 1000 == count // 1000:
            print('%s  Processed %0.2f%% triples.\tMemory used %0.2f%%.\tCpu used %0.2f%%.' %
                  (show_time(datetime.datetime.now()), count / sample_size * 100, psutil.virtual_memory().percent,
                   psutil.cpu_percent(1)))

    print('\n%s  epoch %d: Average_loss=%f\n' % (show_time(datetime.datetime.now()), epoch + 1, losses / (step + 1)))
    plotx.append(epoch + 1)
    ploty.append(losses / (step + 1))
    if epoch // 1 == epoch / 1:  # print error figure per epoch. 每10个epoch打印一次误差折线图
        plt.plot(plotx, ploty)
        plt.savefig(Training_pic_path)  # save the figure

    # learning rate strategy
    if epoch in LR_strategy:  # learning rate strategy
        optimizer.param_groups[0]['lr'] /= 10
    if epoch + 1 >= 5:
        print('\n%s Saving the model temporarily...' % show_time(datetime.datetime.now()))
        if not os.path.exists(os.path.join(work_place, 'toflow_models')):
            os.mkdir(os.path.join(work_place, 'toflow_models'))
        torch.save(toflow.state_dict(), os.path.join(work_place, 'toflow_models', model_name + '_params.pkl'))
        print('Saved.\n')

plt.plot(plotx, ploty)
plt.savefig(Training_pic_path)  # save the last figure

cur_time = datetime.datetime.now()  # current time
h, remainder = divmod(delta_time(prev_time, cur_time), 3600)
m, s = divmod(remainder, 60)
print('%s  Training costs %02d:%02d:%02d' % (show_time(datetime.datetime.now()), h, m, s))

print('\n%s Saving model...' % show_time(datetime.datetime.now()))
if not os.path.exists(os.path.join(work_place, 'toflow_models')):
    os.mkdir(os.path.join(work_place, 'toflow_models'))

# save the complete network, but it has something wrong (toflow can't be serialized)
# torch.save(toflow, os.path.join(work_place, 'toflow_models', model_name + '.pkl'))

# just save the parameters.
torch.save(toflow.state_dict(), os.path.join(work_place, 'toflow_models', model_name + '_params.pkl'))

print('\n%s  Collecting some information...' % show_time(datetime.datetime.now()))
fp = open(os.path.join(work_place, 'toflow_models', model_information_txt), 'w')
fp.write('Model Path:%s\n' % os.path.join(work_place, 'toflow_models', model_name + '_params.pkl'))
fp.write('\nModel Structure:\n')
print(toflow, file=fp)
fp.write('\nModel Hyper Parameters:\n')
fp.write('\tEpoch = %d\n' % EPOCH)
fp.write('\tBatch size = %d\n' % BATCH_SIZE)
fp.write('\tLearning rate = %f\n' % LR)
fp.write('\tWeight decay = %f\n' % WEIGHT_DECAY)
print('\tLR strategy = %s' % str(LR_strategy), file=fp)
fp.write('Train on %dK_%s\n' % (int(sample_size / 1000), 'Vimeo'))
print("Training costs %02d:%02d:%02d" % (h, m, s), file=fp)
fp.close()

cur_time = datetime.datetime.now()
h, remainder = divmod(delta_time(prev_time, cur_time), 3600)
m, s = divmod(remainder, 60)
print('%s  Totally costs %02d:%02d:%02d' % (show_time(datetime.datetime.now()), h, m, s))
print('%s  All done.' % show_time(datetime.datetime.now()))
