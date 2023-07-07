# FastFaceSwap
just a little project for fast face swapping using one picture
## join my discord server https://discord.gg/hzrJBGPpgN
## requirements:
-python 3.10

-cuda 11.7 with cudnn

-everything else is installed automatically!

# how to run

### Here is a [colab example](https://colab.research.google.com/github/RichardErkhov/FastFaceSwap/blob/main/colab_example.ipynb)

```python main.py``` by default starts face swapping from camera.

flags implemented:

-f, --face: uses the face from the following image. Default: face.jpg

-t, --target: replaces the face in the following video/image. Use 0 for camera. Default: 0

-o, --output: path to output the video. Default: video.mp4

-cam-fix, --camera-fix: fix for logitech cameras that start for 40 seconds in default mode. 

-res, --resolution: camera resolution, given in format WxH (ex 1920x1080). Is set for camera mode only. Default: 1920x1080

--threads: amount of threads to run the program in. Default: 2

--image: if the target is image, you have to use this flag. 

--cli: run in cli mode, turns off preview and now accepts the switch of face enhancer from the command line

--face-enhancer: face enhancer, choice works only in cli mode. In gui mode, you need to choose from gui. Available options:

--preview, --preview-mode: !!experimental, might break something!! enables preview mode, doesn't work with camera. 

--experimental: experimental mode to try to optimize the perfomance of reading of frames, sometimes is faster, but requires additional modules

1) none (default)
2) gfpgan
3) ffe (fast face enhancer)

--no-face-swapper: disables face swapper

example:
``` python main.py -f test.jpg -t "C:/Users/user/Desktop/video.mp4" -o output/test.mp4 --threads 12 ```


fast enhancer is still in development, color correction is needed! Sorry for inconvenience, still training the model.
