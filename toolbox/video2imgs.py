import sys
import getopt
from utils import fast_extract_video

video_name = ''
save_dir = ''
extract_music = False
music_name = ''

if sys.argv[1] in ['-h', '--help']:
    print("""pytoflow version 1.1
usage: python3 video2imgs.py [[option] [value]]...
options:
--vn            the absolute path of the video
--fdir          the directory for saving the extracted frames
--extract_m     do you want to extract the music meanwhile? 
--mn            the absolute path of the music
--gpuID         the No. of the GPU you want to use. default: no gpu.
-h, --help     get help.""")
    exit(0)

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [strParameter[2:] + '=' for strParameter in sys.argv[1::2]])[0]:
    if strOption == '--vn':           # task
        video_name = strArgument
    elif strOption == '--fdir':      # dataset_dir
        save_dir = strArgument
    elif strOption == '--extract_m':   # if it isn't for interpolation, you should provide an extra image dir(edited)
        if strArgument in ['True', 'true', 'TRUE', '1']:
            extract_music = True
    elif strOption == '--mn':     # path list file
        music_name = strArgument
    elif strOption == '--gpuID':        # gpu id
        gpuID = int(strArgument)

if video_name == '':
    raise ValueError('Missing [--vn].\nPlease enter the video name.')

fast_extract_video(video_name=video_name,
                   save_dir=save_dir,
                   extract_music=extract_music,
                   music_name=music_name)