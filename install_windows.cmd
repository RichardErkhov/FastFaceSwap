python -m pip install virtualenv
python -m virtualenv venv
call venv\scripts\activate.bat
curl --location --remote-header-name --remote-name https://github.com/RichardErkhov/FastFaceSwap/releases/download/model/inswapper_128.onnx
pip install opencv-python mediapipe insightface onnxruntime-gpu
echo PLEASE ACCEPT
echo PLEASE ACCEPT
echo PLEASE ACCEPT
echo PLEASE ACCEPT
echo PLEASE ACCEPT
echo PLEASE ACCEPT
echo PLEASE ACCEPT
pip uninstall opencv-python opencv-headless-python opencv-contrib-python
pip install opencv-python