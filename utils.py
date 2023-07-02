import subprocess 
from threading import Thread
import onnxruntime as rt
import insightface
import cv2
import numpy as np
def add_audio_from_video(video_path, audio_video_path, output_path):
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', video_path,
        '-i', audio_video_path,
        '-c:v', 'copy',
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-shortest',
        output_path
    ]
    subprocess.run(ffmpeg_cmd, check=True)
class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return
def prepare_models(args):
    providers = rt.get_available_providers()
    sess_options = rt.SessionOptions()
    sess_options.intra_op_num_threads = 8
    if not args['no_faceswap']:
        face_swapper = insightface.model_zoo.get_model("inswapper_128.onnx", session_options=sess_options, providers=providers)
    else:
        face_swapper = None
    face_analyser = insightface.app.FaceAnalysis(name='buffalo_l', providers=providers)
    face_analyser.prepare(ctx_id=0, det_size=(640, 640))
    #face_analyser.models.pop("landmark_3d_68")
    #face_analyser.models.pop("landmark_2d_106")
    #face_analyser.models.pop("genderage")
    return face_swapper, face_analyser
def upscale_image(image, generator ):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (256, 256))

    # Normalize the image to [-1, 1]
    image = (image / 255.0) #- 1

    # Expand the dimensions to match the generator's input shape (1, 128, 128, 3)
    image = np.expand_dims(image, axis=0)

    # Generate the upscaled image
    output = generator.predict(image, verbose=0)

    # Denormalize the output image to [0, 255]
    #output = (output + 1) * 127.5

    # Remove the batch dimension and return the final image
    return cv2.cvtColor((np.squeeze(output, axis=0) * 255.0), cv2.COLOR_BGR2RGB) 