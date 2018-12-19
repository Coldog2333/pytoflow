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
plt.switch_backend('agg')
# --------------------------------------------------------------
# Note:
# 确认Static变量是否试自己想要的
# Hyper Parameters
EPOCH = 15
LR = 1 * 1e-4  # for interpolation, this is the original learning rate.
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 1
LR_strategy = []  # [10, 50]
h = 256
w = 448
# Static
task = 'sr'
use_checkpoint = False
checkpoint_path = './checkpoints/checkpoint_9epoch.ckpt'
work_place = '.'
model_name = 'debug'  # 需要修改
Training_pic_path = 'Training_result.jpg'
model_information_txt = model_name + '_information.txt'
dataset_dir = '/home/ftp/Coldog/Dataset/toflow/vimeo_septuplet/sequences'
# dataset_dir = '/home/ftp/Coldog/Dataset/toflow/vimeo_triplet/sequences'       # interp

if task == 'interp':
    edited_img_dir = ''
elif task in ['denoise', 'denoising']:
    edited_img_dir = '/home/ftp/Coldog/Dataset/toflow/vimeo_septuplet/sequences_with_noise'
elif task in ['sr', 'super-resolution']:
    edited_img_dir = '/home/ftp/Coldog/Dataset/toflow/vimeo_septuplet/sequences_blur'
else:
    raise NameError('Only support: [interp, denoise/denoising, sr/super-resolution]')

pathlistfile = '/home/ftp/Coldog/Dataset/toflow/vimeo_septuplet/sep_trainlist.txt'
# pathlistfile = '/home/ftp/Coldog/Dataset/toflow/vimeo_septuplet/tri_trainlist_tiny.txt'
# pathlistfile = '/home/ftp/Coldog/Dataset/toflow/vimeo_septuplet/sep_testlist_tiny.txt'
# pathlistfile = '/home/ftp/Coldog/Dataset/toflow/vimeo_septuplet/single.txt'
# --------------------------------------------------------------
# prepare DataLoader
Dataset = MemoryFriendlyLoader(origin_img_dir=dataset_dir, edited_img_dir=edited_img_dir, pathlistfile=pathlistfile, task=task)
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


def save_checkpoint(net, optimizer, epoch, losses, savepath):
    save_json = {
        'cuda_flag': net.cuda_flag,
        'h': net.height,
        'w': net.width,
        'net_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'losses': losses
    }
    torch.save(save_json, savepath)


def load_checkpoint(net, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)

    net.cuda_flag = checkpoint['cuda_flag']
    net.height = checkpoint['h']
    net.width = checkpoint['w']
    net.load_state_dict(checkpoint['net_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    losses = checkpoint['losses']

    return net, optimizer, start_epoch, losses


# --------------------------------------------------------------
toflow = TOFlow(h, w, task=task, cuda_flag=True).cuda()

optimizer = torch.optim.Adam(toflow.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
# loss_func = torch.nn.MSELoss()
loss_func = torch.nn.L1Loss()

# Training
prev_time = datetime.datetime.now()  # current time
print('%s  Start training...' % show_time(prev_time))
plotx = []
ploty = []
start_epoch = 0
check_loss = 1

if use_checkpoint:
    toflow, optimizer, start_epoch, ploty = load_checkpoint(toflow, optimizer, checkpoint_path)
    plotx = list(range(len(ploty)))
    check_loss = min(ploty)

for epoch in range(start_epoch, EPOCH):
    losses = 0
    count = 0
    for step, (x, y, path_code) in enumerate(train_loader):
        x = x.cuda()
        y = y.expand((1, 3, 256, 448))
        reference = y.cuda()

        prediction = toflow(x)
        prediction = prediction.cuda()
        loss = loss_func(prediction, reference)

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

        if not os.path.exists('./visualization/'):
            os.mkdir('./visualization/')

        if path_code[0] in ['00004/0357']:
            plt.imsave('./visualization/%d-%s.png' % ((epoch + 1), path_code[0].replace('/','-')),
                       prediction[0, :, :, :].permute(1, 2, 0).cpu().detach().numpy())


    print('\n%s  epoch %d: Average_loss=%f\n' % (show_time(datetime.datetime.now()), epoch + 1, losses / (step + 1)))

    # learning rate strategy
    if epoch in LR_strategy:  # learning rate strategy
        optimizer.param_groups[0]['lr'] /= 10

    plotx.append(epoch + 1)
    ploty.append(losses / (step + 1))
    if epoch // 1 == epoch / 1:  # print error figure per epoch. 每10个epoch打印一次误差折线图
        plt.plot(plotx, ploty)
        plt.savefig(Training_pic_path)  # save the figure

    # checkpoint and then prepare for the next epoch
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')
    save_checkpoint(toflow, optimizer, epoch + 1, ploty, './checkpoints/checkpoint_%depoch.ckpt' % (epoch + 1))

    if check_loss > losses / (step + 1):
        print('\n%s Saving the best model temporarily...' % show_time(datetime.datetime.now()))
        if not os.path.exists(os.path.join(work_place, 'toflow_models')):
            os.mkdir(os.path.join(work_place, 'toflow_models'))
        torch.save(toflow.state_dict(), os.path.join(work_place, 'toflow_models', model_name + '_params_best.pkl'))
        print('Saved.\n')
        check_point = losses / (step + 1)

plt.plot(plotx, ploty)
plt.savefig(Training_pic_path)  # save the last figure

cur_time = datetime.datetime.now()  # current time
h, remainder = divmod(delta_time(prev_time, cur_time), 3600)
m, s = divmod(remainder, 60)
print('%s  Training costs %02d:%02d:%02d' % (show_time(datetime.datetime.now()), h, m, s))

print('\n%s Saving model...' % show_time(datetime.datetime.now()))
if not os.path.exists(os.path.join(work_place, 'toflow_models')):
    os.mkdir(os.path.join(work_place, 'toflow_models'))

# save the complete network
# torch.save(toflow, os.path.join(work_place, 'toflow_models', model_name + '.pkl'))

# just save the parameters.
torch.save(toflow.state_dict(), os.path.join(work_place, 'toflow_models', model_name + '_params_final.pkl'))

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
