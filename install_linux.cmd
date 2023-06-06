python -m pip install virtualenv
python -m virtualenv venv
call venv/bin/activate.bat
wget https://github.com/RichardErkhov/FastFaceSwap/releases/download/model/inswapper_128.onnx
wget https://github.com/RichardErkhov/FastFaceSwap/releases/download/model/GFPGANv1.4.pth
pip install torch torchvision torchaudio
pip install -r requirements.txt
echo PLEASE ACCEPT
echo PLEASE ACCEPT
echo PLEASE ACCEPT
echo PLEASE ACCEPT
echo PLEASE ACCEPT
echo PLEASE ACCEPT
echo PLEASE ACCEPT
pip uninstall opencv-python opencv-headless-python opencv-contrib-python
pip install opencv-python