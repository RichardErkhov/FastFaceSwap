#!/bin/bash
python -m pip install virtualenv
python -m virtualenv venv
call venv/bin/activate
curl -LO https://github.com/RichardErkhov/FastFaceSwap/releases/download/model/inswapper_128.onnx
curl -LO https://github.com/RichardErkhov/FastFaceSwap/releases/download/model/GFPGANv1.4.pth
curl -LO https://github.com/RichardErkhov/FastFaceSwap/releases/download/model/complex_256_v7_stage3_12999.h5
curl -LO https://github.com/RichardErkhov/FastFaceSwap/releases/download/model/GFPGANv1.4.onnx
pip install torch torchvision torchaudio
pip install onnxruntime-silicon
pip install -r requirements.txt
pip install tensorflow-gpu==2.12
pip install protobuf==3.20.2
pip uninstall opencv-python opencv-headless-python opencv-contrib-python -q -y
pip install opencv-python
