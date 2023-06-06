python -m pip install virtualenv
python -m virtualenv venv
call venv\scripts\activate.bat
curl --location --remote-header-name --remote-name https://github.com/RichardErkhov/FastFaceSwap/releases/download/model/inswapper_128.onnx
curl --location --remote-header-name --remote-name https://github.com/RichardErkhov/FastFaceSwap/releases/download/model/GFPGANv1.4.pth
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
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