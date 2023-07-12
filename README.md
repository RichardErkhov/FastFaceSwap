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

1) none (default)
2) gfpgan
3) ffe (fast face enhancer)
4) codeformer

--preview, --preview-mode: !!experimental, might break something!! enables preview mode, doesn't work with camera. 

--experimental: experimental mode to try to optimize the perfomance of reading of frames, sometimes is faster, but requires additional modules

--no-face-swapper: disables face swapper

--lowmem, --low-memory: attempt to make code available for people with low VRAM, might result in lower quality

--batch: enables batch mode, after it provide a suffix, for example --batch="_test.mp4" will result in output %target%_test.mp4

--select-face: change the face you want, not all faces. After the argument add the path to the image with face from the video. (Just open video in video player, screenshot the frame and save it to file. Put this filename after --select-face argument)

example:
``` python main.py -f test.jpg -t "C:/Users/user/Desktop/video.mp4" -o output/test.mp4 --threads 12 ```


fast enhancer is still in development, color correction is needed! Sorry for inconvenience, still training the model.
