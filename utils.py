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
import globalsz
from types import NoneType
from gfpgan import GFPGANer
import sys
import torch
from swapperfp16 import get_model
import requests
import tqdm
if not globalsz.lowmem:
    import tensorflow as tf
if globalsz.args['experimental']:
    try:
        from imutils.video import FileVideoStream
    except ImportError:
        print("In the experimental mode, you have to pip install imutils")
        exit()
        
def restart_program():
    """Restarts the current program."""
    python = sys.executable
    os.execl(python, python, *sys.argv)
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
    physical_devices = tf.config.list_physical_devices('GPU')
    for i in physical_devices:
        tf.config.experimental.set_memory_growth(i, True)
    #tf.config.experimental.set_virtual_device_configuration(
    #        physical_devices[0],
    #        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
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


def prepare_models(args):
    providers = rt.get_available_providers()
    sess_options = rt.SessionOptions()
    sess_options.intra_op_num_threads = 8
    sess_options2 = rt.SessionOptions()
    sess_options2.graph_optimization_level = rt.GraphOptimizationLevel.ORT_DISABLE_ALL #Varying with all the options
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_DISABLE_ALL #Varying with all the options
    if not args['no_faceswap']:
        face_swapper = insightface.model_zoo.get_model("inswapper_128.onnx", session_options=sess_options, providers=providers)
    else:
        face_swapper = None
    if args['lowmem']:
        face_analyser = insightface.app.FaceAnalysis(name='buffalo_l', providers=providers, session_options=sess_options2)
        face_analyser.prepare(ctx_id=0, det_size=(256, 256))
    else:
        face_analyser = insightface.app.FaceAnalysis(name='buffalo_l', providers=providers)
        face_analyser.prepare(ctx_id=0, det_size=(640, 640))
    #face_analyser.models.pop("landmark_3d_68")
    #face_analyser.models.pop("landmark_2d_106")
    #face_analyser.models.pop("genderage")
    return face_swapper, face_analyser

def upscale_image(image, generator ):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (512, 512))
    image = (image / 255.0) #- 1
    image = np.expand_dims(image, axis=0).astype(np.float32)
    #output = generator.run(None, {'input': image})
    output = generator(image)#.predict(image, verbose=0)
    return cv2.cvtColor((np.squeeze(output, axis=0) * 255.0), cv2.COLOR_BGR2RGB)  #np.squeeze(output, axis=0)*255
def show_error():
    messagebox.showerror("Error", "Preview mode does not work with camera, so please use normal mode")
def show_warning():
    messagebox.showwarning("Warning", "Camera is not properly working with experimental mode, sorry")

def compute_cosine_distance(emb1, emb2, allowed_distance):
    d = distance.cosine(emb1, emb2)
    check = False
    if d < allowed_distance:
        check = True
    return d, check

def load_generator():
    if isinstance(globalsz.generator, NoneType):
        #model_path = 'generator.onnx'
        #providers = rt.get_available_providers()
        #generator = rt.InferenceSession(model_path, providers=providers)
        globalsz.generator = tf.keras.models.load_model('complex_256_v7_stage3_12999.h5')#, custom_objects={'Mish': Mish})
    return globalsz.generator
arch = 'clean'
channel_multiplier = 2
model_path = 'GFPGANv1.4.pth'
def load_restorer():
    if isinstance(globalsz.restorer, NoneType):
        globalsz.restorer = GFPGANer(
            model_path=model_path,
            upscale=0.8,
            arch=arch,
            channel_multiplier=channel_multiplier,
            bg_upsampler=None
        )
    return globalsz.restorer

def count_frames(video_path):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    return total_frames

def load_gfpganonnx():
    if isinstance(globalsz.gfpgan_onnx_model, NoneType):
        globalsz.gfpgan_onnx_model = GFPGAN_onnxruntime(model_path="GFPGANv1.4.onnx")
    return globalsz.gfpgan_onnx_model

def restorer_enhance(facer):
    with globalsz.THREAD_SEMAPHORE:
        cropped_faces, restored_faces, facex = load_restorer().enhance(
            facer,
            has_aligned=False,
            only_center_face=False,
            paste_back=True
        )
    return facex
def create_cap():
    global width, height
    if not globalsz.args['experimental']:
        if globalsz.args['camera_fix'] == True:
            cap = cv2.VideoCapture(globalsz.args['target_path'], cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(globalsz.args['target_path'])
        if isinstance(globalsz.args['target_path'], int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, globalsz.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, globalsz.height)
        fourcc = cv2.VideoWriter_fourcc(*'H265')
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        # Get the video's properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        '''cap = VideoCaptureThread(args['target_path'], 30)
        if isinstance(args['target_path'], int):
            show_warning()
        fps = cap.fps
        width = int(cap.width)
        height = int(cap.height)'''
        cap = cv2.VideoCapture(globalsz.args['target_path'])
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        del cap
        cap = FileVideoStream(globalsz.args['target_path']).start()
        time.sleep(1.0)
    # Create a VideoWriter object to save the processed video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    name = globalsz.args['output']
    if isinstance(globalsz.args['target_path'], str):
        name = f"{globalsz.args['output']}_temp.mp4"
    out = cv2.VideoWriter(name, fourcc, fps, (width, height))
    out.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)
    frame_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return [cap, fps, width, height, out, name, globalsz.args['target_path'], frame_number]

def create_batch_cap(file):
    if not globalsz.args['experimental']:
        if globalsz.args['camera_fix'] == True:
            print("no need for camera_fix, there's not camera available in batch processing")
        cap = cv2.VideoCapture(os.path.join(globalsz.args['target_path'], file))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'H265')
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        '''cap = VideoCaptureThread(args['target_path'], 30)
        if isinstance(args['target_path'], int):
            show_warning()
        fps = cap.fps
        width = int(cap.width)
        height = int(cap.height)'''
        cap = cv2.VideoCapture(os.path.join(globalsz.args['target_path'], file))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        del cap
        # yes, might overflow is too many files, well, it's experimental lol, what do you expect?
        cap = FileVideoStream(os.path.join(globalsz.args['target_path'], file)).start() 
        time.sleep(1.0)

    # Create a VideoWriter object to save the processed video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    name = os.path.join(globalsz.args['output'], f"{file}{globalsz.args['batch']}_temp.mp4")#f"{args['output']}_temp{args['batch']}.mp4"
    out = cv2.VideoWriter(name, fourcc, fps, (width, height))
    frame_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return [cap, fps, width, height, out, name, file, frame_number]

def get_gpu_amount():
    num_devices = -1
    if torch.cuda.is_available() and globalsz.cuda:
        num_devices = torch.cuda.device_count()
    return num_devices

def create_configs_for_onnx():
    listx = []
    gpu_amount = get_gpu_amount()
    if gpu_amount == -1:
        return ['CPUExecutionProvider']
    gpu_list = list(range(gpu_amount))
    if not globalsz.select_gpu == None:
        gpu_list = globalsz.select_gpu
    for idx in gpu_list:
        providers = [('CUDAExecutionProvider', {
            'device_id': idx,
        'gpu_mem_limit': 12 * 1024 * 1024 * 1024,
        'gpu_external_alloc': 0,
        'gpu_external_free': 0,
        'gpu_external_empty_cache': 1,
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'cudnn_conv1d_pad_to_nc1d': 1,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'do_copy_in_default_stream': 1,
        'enable_cuda_graph': 0,
        'cudnn_conv_use_max_workspace': 1,
        'tunable_op_enable': 1,
        'enable_skip_layer_norm_strict_mode': 1,
        'tunable_op_tuning_enable': 1
        }),'CPUExecutionProvider'
        ]
        listx.append(providers)
    return listx

def get_sess_options():
    sess_options = rt.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL#rt.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
    sess_options.execution_order = rt.ExecutionOrder.PRIORITY_BASED
    return sess_options

def prepare_models(args):
    providers = rt.get_available_providers()
    sess_options = rt.SessionOptions()
    sess_options.intra_op_num_threads = 8
    sess_options2 = rt.SessionOptions()
    sess_options2.graph_optimization_level = rt.GraphOptimizationLevel.ORT_DISABLE_ALL #Varying with all the options
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_DISABLE_ALL #Varying with all the options
    if not args['no_faceswap']:
        face_swapper = insightface.model_zoo.get_model("inswapper_128.onnx", session_options=sess_options, providers=providers)
    else:
        face_swapper = None
    if args['lowmem']:
        face_analyser = insightface.app.FaceAnalysis(name='buffalo_l', providers=providers, session_options=sess_options2)
        face_analyser.prepare(ctx_id=0, det_size=(256, 256))
    else:
        face_analyser = insightface.app.FaceAnalysis(name='buffalo_l', providers=providers)
        face_analyser.prepare(ctx_id=0, det_size=(640, 640))
    #face_analyser.models.pop("landmark_3d_68")
    #face_analyser.models.pop("landmark_2d_106")
    #face_analyser.models.pop("genderage")
    return face_swapper, face_analyser

def prepare_swappers_and_analysers(args):
    provider_list = create_configs_for_onnx()
    sess_options = get_sess_options()
    swappers = []
    analysers = []
    for idx, providers in enumerate(provider_list):
        if not args['no_faceswap']:
            if args['optimization'] == "fp16":
                swappers.append(get_model("inswapper_128.fp16.onnx", session_options=sess_options, providers=providers))
            elif args['optimization'] == "int8":
                if "CUDAExecutionProvider" in provider_list:
                    print("int8 may not work on gpu properly and might load your cpu instead")
                swappers.append(get_model("inswapper_128.quant.onnx", session_options=sess_options, providers=providers))
            else:
                swappers.append(insightface.model_zoo.get_model("inswapper_128.onnx", session_options=sess_options, providers=providers))
        else:
            swappers.append(None)

        analysers.append(insightface.app.FaceAnalysis(name='buffalo_l', providers=providers, session_options=sess_options))
        analysers[idx].prepare(ctx_id=0, det_size=(256, 256)) #640, 640
    return swappers, analysers

def download(link, filename):
    response = requests.get(link, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024*16  # 1 KB
    progress_bar = tqdm.tqdm(total=total_size, unit='B', unit_scale=True)

    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

    progress_bar.close()

def check_or_download(filename):
    exists = os.path.exists(filename)
    if not exists:
        download(f"https://github.com/RichardErkhov/FastFaceSwap/releases/download/model/{filename}", filename)
    