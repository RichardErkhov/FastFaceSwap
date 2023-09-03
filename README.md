# FastFaceSwap
just a little project for fast face swapping using one picture. Now supports multigpu! (Almost, check the ending of readme)
## join my discord server https://discord.gg/hzrJBGPpgN
## requirements:
-python 3.10

-cuda 11.7 with cudnn

-everything else is installed automatically!

# how to install

## for windows: I prefer using cmd, not powershell. Sometimes it bugs out, so please, use cmd

- clone the repo

if you are on windows with nvidia gpu:

- install_windows.cmd

if you are on windows with amd gpu:

- install_directml_windows.cmd

if you are on linux with nvidia gpu:

- install_linux.sh

if you are on mac, Im not sure if it's going to work properly, but:

- install_mac.sh

if you are on a phone (android), and want to run it for some reason, you are not really supported, but I managed it to run in termux with ubuntu 22.04 (installed inside termux)

- install_termux.sh

Note for linux guys with permission denied:
`chmod +x install_linux.sh` (or whatever your installer is)

then to start the environment:

- start_venv_windows.cmd (if you are on windows)

- start_venv_linux.sh (if you are on linux or android)

- start_mac.sh (if you are on mac)

Note for linux guys with permission denied: same as installer, `chmod +x start_venv_linux.sh` (or whatever you need to run)


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

--face-enhancer: argument works with cli, face enhancer. In gui mode, you need to choose from gui. Available options:

1) none (default)
2) gfpgan
3) ffe (fast face enhancer)
4) codeformer
5) gfpgan_onnx
6) real_esrgan

--no-face-swapper: disables face swapper

--experimental: experimental mode to try to optimize the perfomance of reading of frames, sometimes is faster, but requires additional modules

--no-cuda: no cuda should be used (might break sometimes)

--lowmem, --low-memory: attempt to make code available for people with low VRAM, might result in lower quality

--batch: enables batch mode, after it provide a suffix, for example --batch="_test.mp4" will result in output %target%_test.mp4

--extract-output-frames: extract frames from output video. After argument write the path to folder.

--codeformer-fidelity: argument works with cli, sets up codeformer's fidelity

--blend: argument works with cli, blending amount from 0.0 to 1.0

--codeformer-skip_if_no_face: argument works with cli, Skip codeformer if no face found

--codeformer-face-upscale: argument works with cli, Upscale the face using codeformer

--codeformer-background-enhance:argument works with cli, Enhance the background using codeformer

--codeformer-upscale: argument works with cli, the amount of upscale to apply to the frame using codeformer

--select-face: change the face you want, not all faces. After the argument add the path to the image with face from the video. (Just open video in video player, screenshot the frame and save it to file. Put this filename after --select-face argument)

--optimization: choose the mode of the model: fp32 (default), fp16 (smaller, might be faster), int8 (doesnt work properly on old gpus, I dont know about new once, please test. On old gpus it uses cpu)

--fast-load: try to load as fast as possible, might break something sometimes  

--bbox-adjust: adjustments to do for the box around the face: x1,y1 coords of left top corner and x2,y2 are bottom right. Give in the form x1xy1xx2xy2 (default: 50x50x50x50). Just try to play to understand

-vcam, --virtual-camera: allows to use OBS virtual camera as output source. Please install obs to make sure it works

example:
``` python main.py -f test.jpg -t "C:/Users/user/Desktop/video.mp4" -o output/test.mp4 --threads 12 ```


fast enhancer is still in development, color correction is needed! Sorry for inconvenience, still training the model.

# ABOUT MULTIGPU MODE

To choose the gpu you want to run on: in globalsz.py, on the line with `select_gpu = None` you can make it `select_gpu = [0, 1]` or something similar (these numbers are id of gpus, starting from 0).

to use all gpus, `select_gpu = None`

Multigpu mode for now only supports just face swapping, **without the enhancer**!!! So if you want enhancer to work, for now select only one gpu.

# please read at least TLDR

TL;DR. This tool was created just to make fun to remake memes, put yourself in the movies and other fun things. Some people on the other hand are doing some nasty things using this software, which is not intended way to use this software. Please be a good person, and don’t do harm to other people. Do not hold my liable for anything.
This tool is provided for experimental and creative purposes only. It allows users to generate and manipulate multimedia content using deep learning technology. Users are cautioned that the tool's output, particularly deepfake content, can have ethical and legal implications.
TL;DR ended ====


Educational and Ethical Use: Users are encouraged to use this tool in a responsible and ethical manner. It should primarily serve educational and artistic purposes, avoiding any malicious or misleading activities that could harm individuals or deceive the public.

Informed Consent: If the tool is used to create content involving real individuals, ensure that you have obtained explicit and informed consent from those individuals to use their likeness. Using someone's image without permission can infringe upon their privacy and rights.

Transparency: If you decide to share or publish content created with this tool, it is important to clearly indicate that the content is generated using deep learning technology. Transparency helps prevent misunderstandings and misinformation.

Legal Considerations: Users are responsible for complying with all applicable laws and regulations related to content creation and sharing. Unauthorized use of copyrighted materials, defamation, and invasion of privacy could lead to legal consequences.

Social Responsibility: Please consider the potential social impact of the content you create. Misuse of this tool could contribute to the spread of misinformation, deepening distrust, and undermining the credibility of authentic media.

No Warranty: This tool is provided "as is," without any warranties or guarantees of any kind, either expressed or implied. The developers of this tool are not liable for any direct, indirect, incidental, special, or consequential damages arising from the use of the tool.

Feedback and Improvement: We encourage users to provide feedback on their experiences with the tool. Your insights can contribute to refining the technology and addressing potential concerns.

By using this tool, you acknowledge that you have read and understood this disclaimer. You agree to use the tool responsibly and in accordance with all applicable laws and ethical standards. The developers of this tool retain the right to modify, suspend, or terminate access to the tool at their discretion.

