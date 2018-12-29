# pyTOFlow

This repository is based on the paper 'TOFlow: Video Enhancement with Task-Oriented Flow'. It contains some pre-trained models and all of the codes we used including training code, network structure and so on.


What's more, you can describe it as the python version of the TOFlow presented there ->  [toflow](https://github.com/anchen1011/toflow)

## Evaluation Result

#### Vimeo interp.
| Methods | PSNR | SSIM |
| :-- | -- | -- |
| TOFlow | 33.53 | 0.9668 |
| TOFlow + Mask | **33.73** | **0.9682** |
| pyTOFlow | 33.05 | 0.9633 |

## Prerequisites

#### PyTorch

  Our implementation is based on PyTorch 0.4.1 ([https://pytorch.org/](https://pytorch.org/)).

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

```
sh git clone https://github.com/Coldog2333/pytoflow.git
```
#### Install some required packages



## Train
```
python3 train.py [[option] [value]]...
```
#### Options

+ **--task**: training task, like interp, denoising, super-resolution. valid values:[interp, denoise, denoising, sr, super-resolution]
+ **--dataDir**: the directory of the image dataset(Vimeo-90K)
+ **--ex_dataDir**: the directory of the preprocessed image dataset, for example, the Vimeo-90K mixed by Gaussian noise.
+ **--pathlist**: the text file records which are the images for train.
+ **--gpuID** [optional]: No. of the GPU you want to use. default: no gpu.
+ **-h**, **--help**: get help.


#### Examples
```
python3 train.py --task interp --dataDir ./tiny/vimeo_triplet/sequences --pathlist ./tiny/vimeo_triplet/tri_trainlist.txt --gpuID 1

python3 train.py --task denoising --dataDir ./tiny/vimeo_septuplet/sequences --ex_dataDir ./tiny/vimeo_septuplet/sequences_with_noise --pathlist ./tiny/vimeo_septuplet/sep_trainlist.txt --gpuID 1

python3 train.py --task super-resolution --dataDir ./tiny/vimeo_septuplet/sequences --ex_dataDir ./tiny/vimeo_septuplet/sequences_blur --pathlist ./tiny/vimeo_septuplet/sep_trainlist.txt --gpuID 1
```

## Evaluate

```
python3 evaluate.py [[option] [value]]...
```
#### Options

+ **--task**: training task, like interp, denoising, super-resolution. valid values:[interp, denoise, denoising, sr, super-resolution]
+ **--dataDir**: the directory of the input image dataset(Vimeo-90K, Vimeo-90K with noise, blurred Vimeo-90K)
+ **--pathlist**: the text file records which are the images for train.
+ **--model**: the path of the model used.
+ **--gpuID** [optional]: No. of the GPU you want to use. default: no gpu.
+ **-h**, **--help**: get help.

#### Examples

+ interpolation
```
python3 evaluate.py --task interp --dataDir ./tiny/vimeo_triplet/sequences --pathlist ./tiny/vimeo_triplet/tri_testlist.txt --model ./toflow_models/interp.pkl --gpuID 1
```

+ denoising
```
python3 evaluate.py --task denoising --dataDir ./tiny/vimeo_septuplet/sequences_with_noise --ex_dataDir ./tiny/vimeo_septuplet/sequences_with_noise --pathlist ./tiny/vimeo_septuplet/sep_testlist.txt --model ./toflow_models/denoise.pkl --gpuID 1
```

+ super resolution
```
python3 evaluate.py --task super-resolution --dataDir ./tiny/vimeo_septuplet/sequences_blur --ex_dataDir ./tiny/vimeo_septuplet/sequences_blur --pathlist ./tiny/vimeo_septuplet/sep_testlist.txt --model ./toflow_models/sr.pkl --gpuID 1
```
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


## Acknowledgments
