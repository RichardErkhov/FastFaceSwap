python -m pip install virtualenv
python -m virtualenv venv
source venv/bin/activate
wget https://github.com/RichardErkhov/FastFaceSwap/releases/download/model/inswapper_128.onnx
wget https://github.com/RichardErkhov/FastFaceSwap/releases/download/model/GFPGANv1.4.pth
wget https://github.com/RichardErkhov/FastFaceSwap/releases/download/model/complex_256_v7_stage3_12999.h5
wget https://github.com/RichardErkhov/FastFaceSwap/releases/download/model/GFPGANv1.4.onnx
pip install torch torchvision torchaudio
pip install -r requirements.txt
pip install tensorflow-gpu==2.10.1
pip install protobuf==3.20.2
pip uninstall opencv-python opencv-headless-python opencv-contrib-python -q -y
pip install opencv-python