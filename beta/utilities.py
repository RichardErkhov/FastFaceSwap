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
from PIL import Image
from numpy import asarray
import os
from scipy.spatial import distance
import psutil
import globalsx
from types import NoneType
def is_video_file(filename):
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.webm']  # Add more extensions as needed
    _, ext = os.path.splitext(filename)
    return ext.lower() in video_extensions
def is_picture_file(filename):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.svg', '.tiff', '.webp']
    _, ext = os.path.splitext(filename)
    return ext.lower() in image_extensions
def get_system_usage():
    # Get RAM usage in GB
    ram_usage = round(psutil.virtual_memory().used / 1024**3, 1)

    # Get total RAM in GB
    total_ram = round(psutil.virtual_memory().total / 1024**3, 1)

    # Get CPU usage in percentage
    cpu_usage = round(psutil.cpu_percent(), 0)
    return ram_usage, total_ram, cpu_usage
def extract_frames_from_video(target_video, output_folder):
    target_video_name =  os.path.basename(target_video)
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', target_video,
        f'{output_folder}/{target_video_name}/frame_%05d.png'
    ]
    subprocess.run(ffmpeg_cmd, check=True)
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
def merge_face(temp_frame, original, alpha):
    temp_frame = Image.blend(Image.fromarray(original), Image.fromarray(temp_frame), alpha)
    return asarray(temp_frame)
class GFPGAN_onnxruntime:
    def __init__(self, model_path, use_gpu = False):
        sess_options = rt.SessionOptions()
        sess_options.intra_op_num_threads = 8
        providers = rt.get_available_providers()
        self.ort_session = rt.InferenceSession(model_path, providers=providers, session_options=sess_options)
        self.net_input_name = self.ort_session.get_inputs()[0].name
        _,self.net_input_channels,self.net_input_height,self.net_input_width = self.ort_session.get_inputs()[0].shape
        self.net_output_count = len(self.ort_session.get_outputs())
        self.face_size = 512
        self.face_template = np.array([[192, 240], [319, 240], [257, 371]]) * (self.face_size / 512.0)
        self.upscale_factor = 2
        self.affine = False
        self.affine_matrix = None
    def pre_process(self, img):
        img = cv2.resize(img, (self.face_size, self.face_size))
        img = img / 255.0
        img = img.astype('float32')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img[:,:,0] = (img[:,:,0]-0.5)/0.5
        img[:,:,1] = (img[:,:,1]-0.5)/0.5
        img[:,:,2] = (img[:,:,2]-0.5)/0.5
        img = np.float32(img[np.newaxis,:,:,:])
        img = img.transpose(0, 3, 1, 2)
        return img
    def post_process(self, output, height, width):
        output = output.clip(-1,1)
        output = (output + 1) / 2
        output = output.transpose(1, 2, 0)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        output = (output * 255.0).round()
        if self.affine:
            inverse_affine = cv2.invertAffineTransform(self.affine_matrix)
            inverse_affine *= self.upscale_factor
            if self.upscale_factor > 1:
                extra_offset = 0.5 * self.upscale_factor
            else:
                extra_offset = 0
            inverse_affine[:, 2] += extra_offset
            inv_restored = cv2.warpAffine(output, inverse_affine, (width, height))
            mask = np.ones((self.face_size, self.face_size), dtype=np.float32)
            inv_mask = cv2.warpAffine(mask, inverse_affine, (width, height))
            inv_mask_erosion = cv2.erode(
                inv_mask, np.ones((int(2 * self.upscale_factor), int(2 * self.upscale_factor)), np.uint8))
            pasted_face = inv_mask_erosion[:, :, None] * inv_restored
            total_face_area = np.sum(inv_mask_erosion)
            # compute the fusion edge based on the area of face
            w_edge = int(total_face_area**0.5) // 20
            erosion_radius = w_edge * 2
            inv_mask_center = cv2.erode(inv_mask_erosion, np.ones((erosion_radius, erosion_radius), np.uint8))
            blur_size = w_edge * 2
            inv_soft_mask = cv2.GaussianBlur(inv_mask_center, (blur_size + 1, blur_size + 1), 0)
            inv_soft_mask = inv_soft_mask[:, :, None]
            output = pasted_face
        else:
            inv_soft_mask = np.ones((height, width, 1), dtype=np.float32)
            output = cv2.resize(output, (width, height))
        return output, inv_soft_mask

    def forward(self, img):
        height, width = img.shape[0], img.shape[1]
        img = self.pre_process(img)
        ort_inputs = {self.ort_session.get_inputs()[0].name: img}
        ort_outs = self.ort_session.run(None, ort_inputs)
        output = ort_outs[0][0]
        output, inv_soft_mask = self.post_process(output, height, width)
        output = output.astype(np.uint8)
        return output, inv_soft_mask
def prepare():
    import tensorflow as tf
    def mish_activation(x):
        return x * tf.keras.activations.tanh(tf.keras.activations.softplus(x))
    class Mish(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super(Mish, self).__init__()
        def call(self, inputs):
            return mish_activation(inputs)
    tf.keras.utils.get_custom_objects().update({'Mish': Mish})
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


def prepare_models(det_size='auto', prepare='all', custom_return=False):
    providers = rt.get_available_providers()
    sess_options = rt.SessionOptions()
    sess_options.intra_op_num_threads = 8
    sess_options2 = rt.SessionOptions()
    sess_options2.graph_optimization_level = rt.GraphOptimizationLevel.ORT_DISABLE_ALL #Varying with all the options
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_DISABLE_ALL #Varying with all the options
    if custom_return:
        face_analyzer = insightface.app.FaceAnalysis(name='buffalo_l', providers=providers, session_options=sess_options2)
        face_analyzer.prepare(ctx_id=0, det_size=det_size)
        return face_analyzer
    if not globalsx.args['no_faceswap'] and (prepare == 'all' or prepare == 'face_swapper'):
        globalsx.face_swapper = insightface.model_zoo.get_model("inswapper_128.onnx", session_options=sess_options, providers=providers)
    else:
        globalsx.face_swapper = None
    if globalsx.args['lowmem']:
        if prepare == 'all' or prepare == 'analyser':
            globalsx.face_analyser = insightface.app.FaceAnalysis(name='buffalo_l', providers=providers, session_options=sess_options2)
        if det_size == 'auto' and (prepare == 'all' or prepare == 'analyser'):
            globalsx.face_analyser.prepare(ctx_id=0, det_size=(256, 256))
    else:
        globalsx.face_analyser = insightface.app.FaceAnalysis(name='buffalo_l', providers=providers)
        if det_size == 'auto' and (prepare == 'all' or prepare == 'analyser'):
            globalsx.face_analyser.prepare(ctx_id=0, det_size=(640, 640))
    if det_size != 'auto' and (prepare == 'all' or prepare == 'analyser'):
        globalsx.face_analyzer.prepare(ctx_id=0, det_size=det_size)

    #face_analyser.models.pop("landmark_3d_68")
    #face_analyser.models.pop("landmark_2d_106")
    #face_analyser.models.pop("genderage")
    #return face_swapper, face_analyser

def upscale_image(image, generator ):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (256, 256))
    image = (image / 255.0) #- 1
    image = np.expand_dims(image, axis=0).astype(np.float32)
    #output = generator.run(None, {'input': image})
    output = generator(image)#.predict(image, verbose=0)
    return cv2.cvtColor((np.squeeze(output, axis=0) * 255.0), cv2.COLOR_BGR2RGB)  #np.squeeze(output, axis=0)*255
def show_error(message='', title='Error'):
    messagebox.showerror(title, message)

def show_errorx():
    messagebox.showerror("Error", "Preview mode does not work with camera, so please use normal mode")
def show_warning():
    messagebox.showwarning("Warning", "Camera is not properly working with experimental mode, sorry")

def compute_cosine_distance(emb1, emb2, allowed_distance):
    d = distance.cosine(emb1, emb2)
    check = False
    if d < allowed_distance:
        check = True
    return d, check

def open_cap(file_path=0):
    cap = cv2.VideoCapture(file_path)
    fourcc = cv2.VideoWriter_fourcc(*'H265')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    return cap

def make_swap(frame, face, source_face, paste_back=True): #TODO make fp 16 support 
    return face_swapper.get(frame, face, source_face, paste_back=paste_back)
def swap_frame(frame):
    all_faces = globalsx.face_analyser.get(frame)
    if not isinstance(globalsx.swapper, NoneType) and globalsx.swapper_enabled:
        if globalsx.swap_all_faces:
            for face in faces:
                frame = make_swap(frame, face, globalsx.source_face)
        else:
            for face in all_faces:
                for check_face_list in globalsx.to_swap:
                    emb, chosen_face = check_face_list
                    a = emb.normed_embedding
                    b = face.normed_embedding
                    _, allow = compute_cosine_distance(a,b , 0.75)
                    if not allow:
                        continue
                    frame = make_swap(frame, face, chosen_face)
    return frame