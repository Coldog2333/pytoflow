# --task interp --dataDir ./tiny/vimeo_triplet/sequences --pathlist ./tiny/vimeo_triplet/tri_trainlist.txt --gpuID 1
# --task sr --dataDir ./tiny/vimeo_septuplet/sequences --pathlist ./tiny/vimeo_septuplet/sep_trainlist.txt --gpuID 1 --ex_dataDir ./tiny/vimeo_septuplet/sequences_blur/

import os
import datetime
import torch
import matplotlib.pyplot as plt
import multiprocessing
import psutil
import sys
import getopt
from Network import TOFlow
from read_data import MemoryFriendlyLoader

# ------------------------------
# I don't know whether you have a GPU.
plt.switch_backend('agg')
# Static
# visualize_pathlist = ['00010/0060']
visualize_pathlist = ['00004/0357']
task = ''
dataset_dir = ''
edited_img_dir = ''
pathlistfile = ''
gpuID = None

if sys.argv[1] in ['-h', '--help']:
    print("""pytoflow version 1.0
usage: python3 train.py [[option] [value]]...
options:
--task         training task, like interp, denoising, super-resolution
               valid values:[interp, denoise, denoising, sr, super-resolution]
--dataDir      the directory of the image dataset(Vimeo-90K)
--ex_dataDir   the directory of the preprocessed image dataset, for example, the Vimeo-90K mixed by Gaussian noise.
--pathlist     the text file records which are the images for train.
--gpuID        the No. of the GPU you want to use.
--help         get help.""")
    exit(0)

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [strParameter[2:] + '=' for strParameter in sys.argv[1::2]])[0]:
    if strOption == '--task':           # task
        task = strArgument
    elif strOption == '--dataDir':      # dataset_dir
        dataset_dir = strArgument
    elif strOption == '--ex_dataDir':   # if it isn't for interpolation, you should provide an extra image dir(edited)
        edited_img_dir = strArgument
    elif strOption == '--pathlist':     # path list file
        pathlistfile = strArgument
    elif strOption == '--gpuID':        # gpu id
        gpuID = int(strArgument)


if task == '':
    raise ValueError('Missing [--task].\nPlease enter the training task.')
elif task not in ['interp', 'denoise', 'denoising', 'sr', 'super-resolution']:
    raise ValueError('Invalid [--task].\nOnly support: [interp, denoise/denoising, sr/super-resolution]')

if dataset_dir == '':
    raise ValueError('Missing [--dataDir].\nPlease provide the directory of the dataset. (Vimeo-90K)')
if task in ['denoise', 'denoising', 'sr', 'super-resolution'] and edited_img_dir == '':
    raise ValueError('Missing [--ex_dataDir]. \
                    \nPlease provide the directory of the edited image dataset \
                    \nif you train on denoising or super resolution task. (Vimeo-90K)')

if pathlistfile == '':
    raise ValueError('Missing [--pathlist].\nPlease provide the pathlist index file.')

if gpuID == None:
    cuda_flag = False
else:
    cuda_flag = True
    torch.cuda.device(0)
# --------------------------------------------------------------
# Hyper Parameters
if task == 'interp':
    LR = 3 * 1e-5
elif task in ['denoise', 'denoising', 'sr', 'super-resolution']:
    LR = 1 * 1e-4
EPOCH = 5
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 4
LR_strategy = []
h = 256
w = 448

use_checkpoint = False
checkpoint_path = './checkpoints/checkpoint_0epoch.ckpt'
work_place = '.'
model_name = task
Training_pic_path = 'Training_result.jpg'
model_information_txt = model_name + '_information.txt'
# --------------------------------------------------------------
# prepare DataLoader
Dataset = MemoryFriendlyLoader(origin_img_dir=dataset_dir, edited_img_dir=edited_img_dir, pathlistfile=pathlistfile, task=task)
train_loader = torch.utils.data.DataLoader(dataset=Dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
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
        'net_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'losses': losses
    }
    torch.save(save_json, savepath)


def load_checkpoint(net, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)

    net.cuda_flag = checkpoint['cuda_flag']
    net.load_state_dict(checkpoint['net_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    losses = checkpoint['losses']

    return net, optimizer, start_epoch, losses


# --------------------------------------------------------------
toflow = TOFlow(task=task, cuda_flag=cuda_flag).cuda()

optimizer = torch.optim.Adam(toflow.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
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
        if path_code[0] in visualize_pathlist:
            plt.imsave('./visualization/%d-%s.png' % ((epoch + 1), path_code[0].replace('/','-')),
                       prediction[0, :, :, :].permute(1, 2, 0).cpu().detach().numpy())

    print('\n%s  epoch %d: Average_loss=%f\n' % (show_time(datetime.datetime.now()), epoch + 1, losses / (step + 1)))

    # learning rate strategy
    if epoch in LR_strategy:
        optimizer.param_groups[0]['lr'] /= 10

    plotx.append(epoch + 1)
    ploty.append(losses / (step + 1))
    if epoch // 1 == epoch / 1:
        plt.plot(plotx, ploty)
        plt.savefig(Training_pic_path)

    # checkpoint and then prepare for the next epoch
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')
    save_checkpoint(toflow, optimizer, epoch + 1, ploty, './checkpoints/checkpoint_%depoch.ckpt' % (epoch + 1))

    if check_loss > losses / (step + 1):
        print('\n%s Saving the best model temporarily...' % show_time(datetime.datetime.now()))
        if not os.path.exists(os.path.join(work_place, 'toflow_models')):
            os.mkdir(os.path.join(work_place, 'toflow_models'))
        torch.save(toflow.state_dict(), os.path.join(work_place, 'toflow_models', model_name + '_best_params.pkl'))
        print('Saved.\n')
        check_loss = losses / (step + 1)

plt.plot(plotx, ploty)
plt.savefig(Training_pic_path)

cur_time = datetime.datetime.now()
h, remainder = divmod(delta_time(prev_time, cur_time), 3600)
m, s = divmod(remainder, 60)
print('%s  Training costs %02d:%02d:%02d' % (show_time(datetime.datetime.now()), h, m, s))

print('\n%s Saving model...' % show_time(datetime.datetime.now()))
if not os.path.exists(os.path.join(work_place, 'toflow_models')):
    os.mkdir(os.path.join(work_place, 'toflow_models'))

# save the whole network
# torch.save(toflow, os.path.join(work_place, 'toflow_models', model_name + '.pkl'))

# just save the parameters.
torch.save(toflow.state_dict(), os.path.join(work_place, 'toflow_models', model_name + '_final_params.pkl'))

print('\n%s  Collecting some information...' % show_time(datetime.datetime.now()))
fp = open(os.path.join(work_place, 'toflow_models', model_information_txt), 'w')
fp.write('Model Path:%s\n' % os.path.join(work_place, 'toflow_models', model_name + '_final_params.pkl'))
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
