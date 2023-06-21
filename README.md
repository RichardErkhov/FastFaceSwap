# FastFaceSwap
just a little project for fast face swapping using one picture
## join my discord server https://discord.gg/hzrJBGPpgN
## requirements:
-python 3.10

-cuda 11.7 with cudnn

-everything else is installed automatically!

# how to run

```python main.py``` by default starts face swapping from camera.

flags implemented:

-f, --face: uses the face from the following image. Default: face.jpg

-t, --target: replaces the face in the following video/image. Use 0 for camera. Default: 0

-o, --output: path to output the video. Default: video.mp4

-cam-fix, --camera-fix: fix for logitech cameras that start for 40 seconds in default mode. 

-res, --resolution: camera resolution, given in format WxH (ex 1920x1080). Is set for camera mode only. Default: 1920x1080

--threads: amount of threads to run the program in. Default: 2

--image: if the target is image, you have to use this flag. 

example:
``` python main.py -f test.jpg -t "C:/Users/user/Desktop/video.mp4" -o output/test.mp4 -cam-fix --threads 12 ```


fast enhancer is still in development, color correction is needed! Sorry for inconvenience, still training the model.
