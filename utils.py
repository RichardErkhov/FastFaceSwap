import subprocess 
from threading import Thread
import onnxruntime as rt
import insightface
import cv2
import numpy as np
import threading
import queue
import time
from tkinter import messagebox
import tensorflow as tf
def mish_activation(x):
    return x * tf.keras.activations.tanh(tf.keras.activations.softplus(x))

class Mish(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Mish, self).__init__()

    def call(self, inputs):
        return mish_activation(inputs)
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
tf.keras.utils.get_custom_objects().update({'Mish': Mish})
def get_nth_frame(cap, number):
    cap.set(cv2.CAP_PROP_POS_FRAMES, number)
    ret, frame = cap.read()
    if ret:
        return frame
    return None
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
    
class VideoCaptureThread:
    def __init__(self, video_path, buffer_size):
        self.video_path = video_path
        self.buffer_size = buffer_size
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.condition = threading.Condition()
        self.thread = None
        self.frame_counter = 0
        self.start_time = 0
        self.end_time = 0
        self.width = None
        self.height = None
        self.fps = None
        cap = cv2.VideoCapture(self.video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        self.start()
    def start(self):
        self.thread = threading.Thread(target=self._capture_frames)
        self.thread.start()

    def stop(self):
        with self.condition:
            self.condition.notify_all()  # Unblock all threads
        if self.thread:
            self.thread.join()

    def _capture_frames(self):
        cap = cv2.VideoCapture(self.video_path)
        fourcc = cv2.VideoWriter_fourcc(*'H265')
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        try:
            self.start_time = time.time()
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                with self.condition:
                    # Wait until there is space in the buffer
                    while self.frame_queue.full():
                        self.condition.wait()

                    self.frame_queue.put(frame)
                    self.frame_counter += 1
                    self.condition.notify_all()  # Notify all threads
        finally:
            self.end_time = time.time()
            cap.release()

            with self.condition:
                self.frame_queue.put(None)
                self.condition.notify_all()  # Notify all threads

    def read(self):
        with self.condition:
            # Wait until there is a frame available in the buffer
            while self.frame_queue.empty():
                self.condition.wait()

            frame = self.frame_queue.get()
            self.condition.notify_all()  # Notify all threads
            return frame


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
    output = generator(image)#.predict(image, verbose=0)

    # Denormalize the output image to [0, 255]
    #output = (output + 1) * 127.5

    # Remove the batch dimension and return the final image
    return cv2.cvtColor((np.squeeze(output, axis=0) * 255.0), cv2.COLOR_BGR2RGB) 
def show_error():
    messagebox.showerror("Error", "Preview mode does not work with camera, so please use normal mode")
def show_warning():
    messagebox.showwarning("Warning", "Camera is not properly working with experimental mode, sorry")