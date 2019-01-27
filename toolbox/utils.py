import os
import cv2
import numpy as np
import subprocess
import matplotlib.pyplot as plt


def imgs2video(imgdir=None, imgs=None, video_no_audio='',
               combine_music=False, music_name='', video_with_audio='',
               fps=24, size=None):
    FOURCC = {'avi': 'DIVX',
              'mp4': 'mp4v',
              'wmv': 'mp4v'}
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if imgdir != None:
        filenames = os.listdir(imgdir)
        if size == None:
            img = cv2.imread(os.path.join(imgdir, filenames[0]))
            size = (img.shape[1], img.shape[0])
        if video_no_audio == '':
            video_no_audio = imgdir + '_no_audio.mp4'
        video = cv2.VideoWriter(video_no_audio, fourcc, fps, size)

        for filename in filenames:
            if filename.endswith('.png'):  # 暂时确定png图片ok
                img = cv2.imread(os.path.join(imgdir, filename))
                video.write(img)
    if imgs != None:
        if size == None:
            size = (imgs[0].shape[1], imgs[0].shape[0])
        video = cv2.VideoWriter(video_no_audio, fourcc, fps, size)
        for img in imgs:
            video.write(img)

    video.release()
    if combine_music == True:
        if music_name == '':
            raise NameError('Missing [--mn music name] Please enter audio filepath.')
        else:
            if video_with_audio == '':
                video_with_audio = imgdir + '_with_audio.mp4'
            command = "ffmpeg -i " + music_name + " -i " + video_no_audio + " " + video_with_audio
            print(subprocess.call(command))

    # video.release()   # 之前的错误
    print('All done.')
    return 0

def fast_imgs2video(imgdir, fps=24, music_name=None, output_video='', thread=1, qscale=0.01):
    if music_name:
        audio_format = music_name.split('.')[-1]
    else:
        audio_format = 'wav'    # 目前学院的机子只支持wav
    if output_video == '':
        output_video = os.path.join(os.path.split(imgdir)[0], os.path.split(imgdir)[-1] + '.mp4')

    img_temp = np.array(plt.imread(os.path.join(imgdir, os.listdir(imgdir)[0])))
    width = img_temp.shape[0]
    height = img_temp.shape[1]
    command = 'ffmpeg' + \
              ' -threads %d' % thread + \
              ' -r %d' % fps + \
              ' -i %s/%%06d.png' % imgdir # + \
              # ' -s %d*%d' % (width, height)
    if music_name != None:
        command += ' -i %s' % music_name
    command += ' -qscale %f' % qscale
    command += ' %s' % output_video + ' -y'
    print(command)
    subprocess.call(command, shell=True)
    return 0


def extract_video(video_name, save_dir='', stride=1, isoutput=False, extract_music=False, music_name=''):
    if music_name == '':
        audio_format = 'wav'
    else:
        audio_format = music_name.split('.')[-1]
    if save_dir == '':
        save_dir = os.path.join(os.path.split(video_name)[0], 'frame', os.path.split(video_name)[-1].split('.')[0])
        # 默认放在视频所在目录下的frame文件夹的, 以视频名为文件夹名的文件夹里
        # 比如这里会放在'F:\TOFlow\frame\test'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cap = cv2.VideoCapture(video_name)
    frame_count = 0
    Total_frame_count = 0
    frames = []
    success = True

    while (success):
        try:
            success, _ = cap.read()
            Total_frame_count += 1
        except:
            break
    success = True
    cap = cv2.VideoCapture(video_name)

    count = 0
    print('Start extracting frames...')
    while (success):
        try:
            success, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # transform to RGB color space.
            frame = frame[:, :, :3]
            plt.imsave(os.path.join(save_dir, '%06d.png' % (frame_count + 1)), frame)
            frame_count += 1
            count += 1
            # skip some frames
            for i in range(1, stride):
                _, _ = cap.read()
                count += 1
            if isoutput == True:
                frames.append(frame)
            if frame_count // 50 == frame_count / 50:
                print('  Extracted %f%% video.' % (count / Total_frame_count * 100))
        except:
            break
    print('Extracted %d frames.' % frame_count)

    if extract_music == True:
        print('Extracting music...')
        if music_name == '':  # default music name
            music_name = '.'.join(video_name.split('.')[:-1]) + '.' + audio_format
        command = 'ffmpeg -i ' + video_name + ' -f %s -vn ' % audio_format + music_name
        subprocess.call(command)
        print('Music extracted.')

    return cap.get(cv2.CAP_PROP_FPS)


def fast_extract_video(video_name, save_dir='', extract_music=False, music_name=''):
    """
    :param video_name: 视频绝对路径, eg. 'F:\TOFlow\test.mp4'
    :param save_dir: 帧存储的文件夹, eg. 'F:\TOFlow\frame\test.mp4'
    :param extract_music: flag, 是否提取音频
    :param music_name: 提取音频存储的绝对路径, eg. 'F\TOFlow\test.mp3'
    :return:
    """
    if music_name:
        audio_format = music_name.split('.')[-1]
    else:
        audio_format = 'wav'    # 目前学院的机子只支持wav
    cap = cv2.VideoCapture(video_name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if save_dir == '':
        save_dir = os.path.join(os.path.split(video_name)[0], 'frame', os.path.split(video_name)[-1].split('.')[0])
        # 默认放在视频所在目录下的frame文件夹的, 以视频名为文件夹名的文件夹里
        # 比如这里会放在'F:\TOFlow\frame\test'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    print('Extracting frames...')
    command = 'ffmpeg' + \
              ' -i ' + video_name + \
              ' -q:v 2 ' + \
              ' -f image2 %s\%%06d.png' % save_dir + \
              ' -s %dx%d' % (width, height) + \
              ' -r %d' % fps
    print(command)
    flag = subprocess.call(command, shell=True)    # ffmpeg -i F:\TOFlow\test.mp4 -f image2 F:\TOFlow\frame\test\%06d.png -s 1920*1080 -r 24
    if flag:
        raise RuntimeError('Cannot extract frames with ffmpeg.')
    print('Extracted frames.')

    if extract_music == True:
        if music_name == '':  # default music name
            music_name = '.'.join(video_name.split('.')[:-1]) + '.' + audio_format
            # 默认放在视频所在目录下,文件名同视频文件
            # 比如这里会保存在'F:\TOFlow\test.mp3'
        print('Extracting music...')
        command = 'ffmpeg' + \
                  ' -i ' + video_name + \
                  ' -f %s' % audio_format + \
                  ' -vn ' + music_name + \
                  ' -y'
        subprocess.call(command, shell=True)
        print('Music extracted.')
    return 0        # 表示成功退出
