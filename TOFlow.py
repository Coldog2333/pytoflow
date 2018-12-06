import os
import datetime
import torch
import matplotlib.pyplot as plt
import multiprocessing
import psutil
from Network import TOFlow
from read_data import MemoryFriendlyLoader

# --------------------------------------------------------------
# I don't know whether you have a GPU.
# torch.cuda.set_device(0)
# --------------------------------------------------------------
# Note:
# please check the Static variable used.
# Hyper Parameters
EPOCH = 15
LR = 3 * 1e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 1
LR_strategy = []
h = 256
w = 448
# Static
work_place = '.'
model_name = 'interp'
Training_pic_path = 'Training_result.jpg'
model_information_txt = model_name + '_information.txt'
train_dir = './Dataset/train/'   # training dataset path
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

# --------------------------------------------------------------

toflow = TOFlow(h, w).cuda()

spynet_params = list(map(id, toflow.SpyNet.parameters()))
other_params = filter(lambda p: id(p) not in spynet_params, toflow.parameters())
optimizer = torch.optim.Adam([
    {'params': other_params},
    {'params': toflow.SpyNet.parameters(), 'lr': LR/10}     # finetune SpyNet
    ], lr=LR, weight_decay=WEIGHT_DECAY)
# optimizer = torch.optim.Adam(toflow.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
# loss_func = torch.nn.MSELoss()
loss_func = torch.nn.L1Loss()

# Training
prev_time = datetime.datetime.now()  # current time
print('%s  Start training...' % show_time(prev_time))
plotx = []
ploty = []

for epoch in range(EPOCH):
    losses = 0
    count = 0
    for step, (x, y) in enumerate(train_loader):
        frameFirst = x[:, 0, :, :, :].cuda()
        frameSecond = x[:, 1, :, :, :].cuda()
        y = y.expand((1, 3, h, w))
        reference = y.cuda()

        prediction = toflow(frameFirst, frameSecond)
        prediction = prediction.cuda()
        loss = loss_func(prediction, reference)

        # losses += loss                # the reason why oom happened
        losses += loss.item()
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        count += len(x)
        # monitor the system resources.
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
