# pytoflow

This repository is based on the paper 'TOFlow: Video Enhancement with Task-Oriented Flow'. It contains some pre-trained models and all of the codes we used including training code, Network structure and so on.


What's more, you can describe it as the python version of the TOFlow presented there ->  [toflow](https://github.com/anchen1011/toflow)


## Prerequisites

#### PyTorch

  Our implementation is based on PyTorch 0.4.0 ([https://pytorch.org/](https://pytorch.org/)).

#### PIL and matplotlib

  For loading images.

#### opencv-python(cv2)

  For processing videos.

#### CUDA [optional]

  CUDA is suggested ([https://developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit)) for fast inference. The demo code is still runnable without CUDA, but much slower.

#### FFmpeg [optional]

  We use FFmpeg ([http://ffmpeg.org/](http://ffmpeg.org/)) for processing videos. That's ok if you don't have a FFmpeg, but maybe it will cost you lot of time to processing.


## Installation

Our current release has been tested on Ubuntu 16.04 LTS.

#### Clone the repository

```sh
git clone https://github.com/Coldog2333/pytoflow.git
```
#### Install some required packages


## Usage

```
python3 run.py --f1 example/im1.png --f2 example/im3.png --o example/out.png --gpuID 0
``` 

#### Options

+ **--f1**: filename of the first frame
+ **--f2**: filename of the second frame
+ **--o** [optional]: filename of the predicted frame. default: out.png, saving in the same directory of the input frames.
+ **--gpuID** [optional]: No of the GPU you want to use. default: no gpu.


## References

1. Xue T , Chen B , Wu J , et al. Video Enhancement with Task-Oriented Flow[J]. 2017.([http://arxiv.org/abs/1711.09078](http://arxiv.org/abs/1711.09078))
2. Our SpyNet is based on [sniklaus/pytorch-spynet](https://github.com/sniklaus/pytorch-spynet)
