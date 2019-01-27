import sys
import getopt
from utils import fast_imgs2video

imgdir = ''
fps = 24
music_name = None
output_video = ''
thread = 1
qscale = 0.01

if sys.argv[1] in ['-h', '--help']:
    print("""pytoflow version 1.1
usage: python3 imgs2video.py [[option] [value]]...
options:
--fdir          the directory of the frames.
--fps           fps. default: 24
--mn            the absolute path of the music.
--ov            the absolute path of the output video.
--t             multi threads. default: 1
--qscale        the quality of the output video. default: 0.01
--gpuID         the No. of the GPU you want to use. default: no gpu.
-h, --help     get help.""")
    exit(0)

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [strParameter[2:] + '=' for strParameter in sys.argv[1::2]])[0]:
    if strOption == '--fps':           # task
        fps = int(strArgument)
    elif strOption == '--fdir':      # dataset_dir
        imgdir = strArgument
    elif strOption == '--mn':   # if it isn't for interpolation, you should provide an extra image dir(edited)
        music_name = strArgument
    elif strOption == '--ov':     # path list file
        output_video = strArgument
    elif strOption == '--t':
        thread = int(strArgument)
    elif strOption == '--qscale':
        qscale = eval(strArgument)
    elif strOption == '--gpuID':        # gpu id
        gpuID = int(strArgument)

if imgdir == '':
    raise ValueError('Missing [--fdir].\nPlease enter the directory of the frames.')

fast_imgs2video(imgdir=imgdir,
                fps=fps,
                music_name=music_name,
                output_video=output_video,
                thread=thread,
                qscale=qscale)