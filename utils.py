
import globalsz
import threading

def prepare():
    def mish_activation(x):
        return x * tf.keras.activations.tanh(tf.keras.activations.softplus(x))
    class Mish(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super(Mish, self).__init__()
        def call(self, inputs):
            return mish_activation(inputs)
    tf.keras.utils.get_custom_objects().update({'Mish': Mish})
def fastloadimporter():
    global torch, cv2, gpu_memory_total, tf, device
    import torch
    import cv2
    
    if not globalsz.args['nocuda'] and not globalsz.args['apple']:
        device = torch.device(0)
        gpu_memory_total = round(torch.cuda.get_device_properties(device).total_memory / 1024**3,2)  # Convert bytes to GB
    elif globalsz.args['apple']:
        device = torch.device('mps')
    if not globalsz.args['lowmem']:
        import tensorflow as tf
        prepare()
if globalsz.args['fastload']:
    wait_thread = threading.Thread(target=fastloadimporter)
    wait_thread.start()
import subprocess 
from threading import Thread
import onnxruntime as rt
import insightface
import cv2
import numpy as np
import time
if not globalsz.args['cli']:
    from tkinter import messagebox
from PIL import Image
import os
import psutil
NoneType = type(None)
import sys
#if not globalsz.args['nocuda']:
#    torch.backends.cudnn.benchmark = True
import tqdm
import magic    #pip install python-magic-bin https://github.com/Yelp/elastalert/issues/1927
mime = magic.Magic(mime=True)
if not globalsz.args['fastload']:
    import mediapipe as mp
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.download_util import load_file_from_url
    from realesrgan import RealESRGANer
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact
    from gfpgan import GFPGANer
    import requests
    from swapperfp16 import get_model
    from scipy.spatial import distance
    import queue
    import torch
    from rembg import remove as remove_bg
    from rembg import new_session
if not globalsz.lowmem:
    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('GPU')
    for i in physical_devices:
        tf.config.experimental.set_memory_growth(i, True)
        
        tf.config.experimental.set_virtual_device_configuration(
                i,[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
if globalsz.args['experimental']:
    try:
        from imutils.video import FileVideoStream
    except ImportError:
        print("In the experimental mode, you have to pip install imutils")
        exit()
import numpy as np

def calculate_rotation_angles(image):
    angles = []  # Initialize an empty list to store angles for each face
    
    with globalsz.advanced_face_detector_lock:
        results = globalsz.face_mesh.process(image)  # cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        h, w, c = image.shape
        for face_landmarks in results.multi_face_landmarks:  # Iterate over each face
            landmarks = face_landmarks.landmark
            
            # Identify specific landmarks for angle calculation
            bottom_middle_landmark = (int(landmarks[175].x * w), int(landmarks[175].y * h))
            landmark_151 = (int(landmarks[151].x * w), int(landmarks[151].y * h))

            # Calculate the rotation angle for this face
            angle = np.arctan2(
                bottom_middle_landmark[0] - landmark_151[0],
                landmark_151[1] - bottom_middle_landmark[1]
            ) * 180 / np.pi

            # Append the angle to the list
            angles.append(angle + 180)
        
        return angles  # Return the list of angles
    
    return [0]  # Return [0] if no faces are detected

# Function to rotate an image
def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
    
    # Calculate new dimensions after rotation
    cos_angle = np.abs(rotation_matrix[0, 0])
    sin_angle = np.abs(rotation_matrix[0, 1])
    new_w = int(w * cos_angle + h * sin_angle)
    new_h = int(w * sin_angle + h * cos_angle)
    
    # Adjust the translation part of the rotation matrix
    rotation_matrix[0, 2] += (new_w - w) / 2
    rotation_matrix[1, 2] += (new_h - h) / 2
    
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h))
    return rotated_image, rotation_matrix

# Function to rotate an image back to its original orientation
def rotate_back(image, rotation_matrix, original_shape):
    h, w = original_shape[:2]
    rotated_back = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.WARP_INVERSE_MAP)
    return rotated_back
def get_face_details(image, adjustment_pixels):
    # Initialize empty lists for bounding boxes and rotation angles
    adjusted_bboxes = []
    rotation_angles = []
    
    with globalsz.advanced_face_detector_lock:
        results = globalsz.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        h, w, c = image.shape

        for face_landmarks in results.multi_face_landmarks:
            # Bounding box calculation
            min_x, min_y, max_x, max_y = w, h, 0, 0
            for landmark in face_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)

            # Adjust the bounding box
            min_x -= adjustment_pixels
            min_y -= adjustment_pixels
            max_x += adjustment_pixels
            max_y += adjustment_pixels

            # Ensure the adjusted bounding box stays within image boundaries
            min_x = max(min_x, 0)
            min_y = max(min_y, 0)
            max_x = min(max_x, w)
            max_y = min(max_y, h)

            adjusted_bboxes.append((min_x, min_y, max_x, max_y))

            # Rotation angle calculation
            landmarks = face_landmarks.landmark
            bottom_middle_landmark = (int(landmarks[175].x * w), int(landmarks[175].y * h))
            landmark_151 = (int(landmarks[151].x * w), int(landmarks[151].y * h))

            angle = np.arctan2(
                bottom_middle_landmark[0] - landmark_151[0],
                landmark_151[1] - bottom_middle_landmark[1]
            ) * 180 / np.pi

            rotation_angles.append(angle + 180)

    return adjusted_bboxes, rotation_angles


def get_face_bboxes(image, adjustment_pixels):
    with globalsz.advanced_face_detector_lock:
        results = globalsz.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    adjusted_bboxes = []

    if results.multi_face_landmarks:
        h, w, c = image.shape

        for face_landmarks in results.multi_face_landmarks:
            min_x, min_y, max_x, max_y = w, h, 0, 0

            for landmark in face_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)

            # Adjust the bounding box by adding N pixels to each side
            min_x -= adjustment_pixels
            min_y -= adjustment_pixels
            max_x += adjustment_pixels
            max_y += adjustment_pixels

            # Ensure the adjusted bounding box stays within image boundaries
            min_x = max(min_x, 0)
            min_y = max(min_y, 0)
            max_x = min(max_x, w)
            max_y = min(max_y, h)

            adjusted_bboxes.append((min_x, min_y, max_x, max_y))

    return adjusted_bboxes
def init_advanced_face_detector():
    global mp
    if globalsz.args['fastload']:
        import mediapipe as mp
    if isinstance(globalsz.mp_face_mesh, NoneType):
        globalsz.mp_face_mesh = mp.solutions.face_mesh
    if isinstance(globalsz.face_mesh, NoneType):
        globalsz.face_mesh = globalsz.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    

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
    print(video_path)
    ffmpeg_cmd = [
        'ffmpeg',
        "-an",
        '-i', f'"{video_path}"',
        '-i', f'"{audio_video_path}"',
        #'-c:v', 'copy',    # Copy video codec settings
        #'-c', 'copy',    # Copy audio codec settings
        '-map', '1:a:0?',
        '-map', '0:v:0',
        #'-acodec', 'copy',
        #'-shortest',
        f'"{output_path}"'
    ]
    subprocess.run(ffmpeg_cmd, check=True)
def merge_face(temp_frame, original, alpha):
    temp_frame = Image.blend(Image.fromarray(original), Image.fromarray(temp_frame), alpha)
    return np.asarray(temp_frame)
class GFPGAN_onnxruntime:
    def __init__(self, model_path, use_gpu = False):
        sess_options = rt.SessionOptions()
        sess_options.intra_op_num_threads = 8
        providers = rt.get_available_providers()
        self.ort_session = rt.InferenceSession(model_path, providers=providers, session_options=sess_options)
        self.net_input_name = self.ort_session.get_inputs()[0].name
        _,self.net_input_channels,self.net_input_height,self.net_input_width = self.ort_session.get_inputs()[0].shape
        self.net_output_count = len(self.ort_session.get_outputs())
        self.face_size = 512 #512
        self.face_template = np.array([[192, 240], [319, 240], [257, 371]]) * (self.face_size / 512.0)
        self.upscale_factor = 1
        self.affine = False
        self.affine_matrix = None
    def pre_process(self, img):
        #img = cv2.resize(img, (self.face_size//2, self.face_size//2))
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
def get_nth_frame(cap, number, type=0):
    #type 0 video
    #type 1 camera
    #type 2 image
    if type == 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, number)
    ret, frame = cap.read()
    if ret:
        return frame
    return None
def reset_cap(cap):
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return cap
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
        import queue
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


'''def prepare_models(args):
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
    return face_swapper, face_analyser'''

def upscale_image(image, generator ):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (256, 256))
    image = (image / 255.0) #- 1
    image = np.expand_dims(image, axis=0).astype(np.float32)
    #output = generator.run(None, {'input': image})
    output = generator(image)#.predict(image, verbose=0)
    return cv2.cvtColor((np.squeeze(output, axis=0) * 255.0), cv2.COLOR_BGR2RGB)  #np.squeeze(output, axis=0)*255
def show_error():
    messagebox.showerror("Error", "Preview mode does not work with camera, so please use normal mode")
def show_warning():
    messagebox.showwarning("Warning", "Camera is not properly working with experimental mode, sorry")

def show_error_custom(text=''):
    messagebox.showerror("Error", text)
def compute_cosine_distance(emb1, emb2, allowed_distance):
    global distance
    if globalsz.args['fastload']:
        from scipy.spatial import distance
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

def load_read_esrgan():
    global RRDBNet, load_file_from_url, RealESRGANer, SRVGGNetCompact
    if globalsz.args['fastload']:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from basicsr.utils.download_util import load_file_from_url
        from realesrgan import RealESRGANer
        from realesrgan.archs.srvgg_arch import SRVGGNetCompact
    model_path = None
    if isinstance(globalsz.realeasrgan_enhancer, NoneType):
        model_name = globalsz.realeasrgan_model
        """Load the appropriate model based on the model name."""
        # determine models according to model names
        if model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
        elif model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
        elif model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            netscale = 4
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
        elif model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            netscale = 2
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
        elif model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
            netscale = 4
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
        elif model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
            netscale = 4
            file_url = [
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
            ]

        # determine model paths
        if not model_path:
            model_path = os.path.join('weights', model_name + '.pth')
            if not os.path.isfile(model_path):
                ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
                for url in file_url:
                    model_path = load_file_from_url(url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True)
        
        TILE = 0
        TILE_PAD = 10
        PRE_PAD = 0
        # Initialize restorer
        if globalsz.args['apple']:
            dev = torch.device('mps')
        elif globalsz.args['nocuda']:
            dev = torch.device('cpu')
        else:
            dev=torch.device(f"cuda:{globalsz.select_realesrgan_gpu}")

        globalsz.realeasrgan_enhancer = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            model=model,
            tile=TILE,
            tile_pad=TILE_PAD,
            pre_pad=PRE_PAD,
            half= globalsz.realesrgan_fp16,
            device = dev
        )

    return globalsz.realeasrgan_enhancer

def realesrgan_enhance(image):
    with globalsz.realesrgan_lock:
        output, _ = load_read_esrgan().enhance(image, outscale=globalsz.realesrgan_outscale)
    return output
arch = 'clean'
channel_multiplier = 2
model_path = 'GFPGANv1.4.pth'
def load_restorer():
    global GFPGANer
    if isinstance(globalsz.restorer, NoneType):
        
        if globalsz.args['fastload']:
            from gfpgan import GFPGANer
            
        if globalsz.args['apple']:
            dev = torch.device('mps')
        elif globalsz.args['nocuda']:
            dev = torch.device('cpu')
        else:
            dev = torch.device(f"cuda:{globalsz.select_gfpgan_gpu}")
        globalsz.restorer = GFPGANer(
            model_path=model_path,
            upscale=0.8,
            arch=arch,
            channel_multiplier=channel_multiplier,
            bg_upsampler=None,
            device = dev
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

def create_new_cap(file, face_, output_,batch_post=""):
    if not isinstance(file, int):
        try:
            video_type = mime.from_file(file)
        except Exception as e:
            print(f"{file} is not image or video, error from video_type: {e}")
            return
    else:
        video_type = 'video'
    if video_type.startswith('video'):
        if batch_post != "":
            if not batch_post.endswith(".mp4"):
                batch_post += ".mp4"
        if globalsz.args['camera_fix'] == True:
            cap = cv2.VideoCapture(file, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(file)
        if isinstance(file, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, globalsz.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, globalsz.height)
        fourcc = cv2.VideoWriter_fourcc(*'H265')
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        # Get the video's properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_filename = os.path.basename(output_)
        name = os.path.join(output_.rstrip(output_filename).rstrip(), f"{output_filename}{batch_post}")
        name_temp = os.path.join(output_.rstrip(output_filename).rstrip(), f"{output_filename}{batch_post}_temp.mp4")#f"{args['output']}_temp{args['batch']}.mp4"
        out = cv2.VideoWriter(name_temp, fourcc, fps, (width, height))
        frame_number = 1
        if not isinstance(file, int):
            frame_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        #face_ = 
        return {"type": 1,
                "cap":cap,
                "original_image":None,
                "swapped_image":None,
                "target_path":file,
                "save_path":name,
                "save_temp_path":name_temp,
                "current_frame_index":isinstance(file, int),
                "old_number":-1,
                "frame_number":frame_number,
                "rendering":globalsz.args['cli'],
                "width":width,
                "height":height,
                "fps":fps,
                "faces_to_swap":None,
                "settings":{
                    "threads":None,
                    "enable_swapper": not globalsz.args['no_faceswap'],
                    "enable_enhancer": False,
                    "enhancer_choice": "none",
                    "bbox_adjust": [50, 50, 50, 50],
                    "codeformer_fidelity":0.1,
                    "blender":1.0,
                    "codeformer_skip_if_no_face": False,
                    "codeformer_upscale_face": True,
                    "codeformer_enhancer_background": False,
                    "codeformer_upscale_amount":1,
                    },
                "out_settings_for_resetting":{
                    "name_temp":name_temp,
                    "fourcc":fourcc,
                    "fps":fps,
                    "width":width,
                    "height":height,
                },
                "out":out,
                "count":-1,
                "first_frame":get_nth_frame(cap, 0),
                "temp": [],
                "face":face_
                }
    if video_type.startswith('image'):
        if batch_post != "":
            if not batch_post.endswith(".png"):
                batch_post += ".png"
        output_filename = os.path.basename(output_)
        name = os.path.join(output_.rstrip(output_filename).rstrip(), f"{output_filename}{batch_post}")
        image = cv2.imread(file)
        width, height = image.shape[:2]
        return {"type": 0,
                "cap": None,
                "original_image":image,
                "swapped_image":None,
                "target_path":file,
                "save_path":name,
                "save_temp_path":None,
                "current_frame_index":0,
                "old_number":-1,
                "frame_number":-1,
                "rendering":globalsz.args['cli'],
                "width":width,
                "height":height,
                "fps":-1,
                "faces_to_swap":None,
                "settings":{
                    "threads":None,
                    "enable_swapper": not globalsz.args['no_faceswap'],
                    "enable_enhancer": False,
                    "enhancer_choice": "none",
                    "bbox_adjust": [50, 50, 50, 50],
                    "codeformer_fidelity":0.1,
                    "blender":1.0,
                    "codeformer_skip_if_no_face": False,
                    "codeformer_upscale_face": True,
                    "codeformer_enhancer_background": False,
                    "codeformer_upscale_amount":1,
                    },
                "out_settings_for_resetting":None,
                "out":None,
                "count":-1,
                "first_frame":image,#get_nth_frame(cap, 0),
                "temp": [],
                "face":face_}
    print(video_type)
def write_frame(video):
    if video["type"] == 0:
        print(video['save_path'])
        cv2.imwrite(video['save_path'], video['swapped_image'])
        return
    video['out'].write(video["swapped_image"][:,:, :3])
    return

def get_frame(video, frame_index=-1, toret=False):
    #if index == -1, just get the frame
    if video['type'] == 0:
        if toret:
            return True, video["original_image"]
        return video["original_image"]
    if frame_index != -1:
        return get_nth_frame(video['cap'], frame_index)
    ret, frame = video['cap'].read()
    if ret:
        if toret:
            return ret, frame
        return frame
    return ret, None

def get_gpu_amount():
    num_devices = -1
    if torch.cuda.is_available() and globalsz.cuda:
        num_devices = torch.cuda.device_count()
    return num_devices

def create_configs_for_onnx():
    listx = []
    gpu_amount = get_gpu_amount()
    if gpu_amount == -1 and not globalsz.args['apple']:
        return [('CPUExecutionProvider',),]
    elif globalsz.args['apple']:
        return [('CoreMLExecutionProvider',),]
    gpu_list = list(range(gpu_amount))
    if not globalsz.select_face_swapper_gpu == None:
        gpu_list = globalsz.select_face_swapper_gpu
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
        listx.append([idx, providers])
    return listx
def create_configs_for_onnx_rembg():
    listx = []
    gpu_amount = get_gpu_amount()
    if gpu_amount == -1 and not globalsz.args['apple']:
        return [('CPUExecutionProvider',),]
    elif globalsz.args['apple']:
        return [('CoreMLExecutionProvider',),]
    gpu_list = list(range(gpu_amount))
    if not globalsz.select_rembg_gpu == None:
        gpu_list = globalsz.select_rembg_gpu
    for idx in gpu_list:
        providers = [('CUDAExecutionProvider', {
            'device_id': idx,
        #'gpu_mem_limit': 12 * 1024 * 1024 * 1024,
        #'gpu_external_alloc': 0,
        #'gpu_external_free': 0,
        #'gpu_external_empty_cache': 1,
        #'cudnn_conv_algo_search': 'EXHAUSTIVE',
        #'cudnn_conv1d_pad_to_nc1d': 1,
        #'arena_extend_strategy': 'kNextPowerOfTwo',
        #'do_copy_in_default_stream': 1,
        #'enable_cuda_graph': 0,
        #'cudnn_conv_use_max_workspace': 1,
        #'tunable_op_enable': 1,
        #'enable_skip_layer_norm_strict_mode': 1,
        #'tunable_op_tuning_enable': 1
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

def prepare_rembg(args):
    global remove_bg, new_session
    if isinstance(globalsz.rembg_models, NoneType):
        if args['fastload']:
            from rembg import remove as remove_bg
            from rembg import new_session
        provider_list = create_configs_for_onnx_rembg()
        #sess_options = get_sess_options()
        globalsz.rembg_models = []
        #for idx, providers in enumerate(provider_list):
        globalsz.rembg_models.append(new_session(globalsz.rembg_model))#, providers=providers))
    #return rembg_models
    
def remove_background(frame,args, ct=0, magic = True):
    global remove_bg
    ct = 0
    with globalsz.rembg_lock:
        prepare_rembg(args)
    # Convert frame to PNG bytes
    _, buffer = cv2.imencode('.png', frame)
    frame_bytes = buffer.tobytes()

    # Remove background
    output_bytes = remove_bg(frame_bytes, session=globalsz.rembg_models[ct], post_process_mask=True, bgcolor=globalsz.rembg_color)  # Make the background fully transparent for now

    # Convert bytes back to a NumPy array
    nparr = np.frombuffer(output_bytes, np.uint8)
    output_frame = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if magic:
        # Get the alpha channel
        alpha_channel = output_frame[:, :, 3]

        # Step 1: Dilate and Erode
        kernel = np.ones((5,5), np.uint8)
        alpha_channel = cv2.dilate(alpha_channel, kernel, iterations=1)
        alpha_channel = cv2.erode(alpha_channel, kernel, iterations=1)

        # Step 2: Blur and Threshold
        alpha_channel = cv2.GaussianBlur(alpha_channel, (5, 5), 0)
        _, alpha_channel = cv2.threshold(alpha_channel, 127, 255, cv2.THRESH_BINARY)

        # Replace the alpha channel in the output frame
        output_frame[:, :, 3] = alpha_channel
    #print(output_frame.shape)
    return output_frame

def prepare_swappers_and_analysers(args):
    global get_model
    provider_list = create_configs_for_onnx()
    sess_options = get_sess_options()
    swappers = []
    analysers = []
    for idx, (device_id, providers) in enumerate(provider_list):
        if not args['no_faceswap']:
            if args['optimization'] == "fp16":
                
                if globalsz.args['fastload']:
                    from swapperfp16 import get_model
                swappers.append(get_model("inswapper_128.fp16.onnx", argsz=args, session_options=sess_options, providers=providers))
            elif args['optimization'] == "int8":
                if "CUDAExecutionProvider" in provider_list:
                    print("int8 may not work on gpu properly and might load your cpu instead")
                    
                if globalsz.args['fastload']:
                    from swapperfp16 import get_model
                swappers.append(get_model("inswapper_128.quant.onnx", argsz=args, session_options=sess_options, providers=providers))
            else:
                
                if globalsz.args['fastload']:
                    from swapperfp16 import get_model
                swappers.append(get_model("inswapper_128.onnx", argsz=args, session_options=sess_options, providers=providers))
        else: #insightface.model_zoo.
            swappers.append(None)

        analysers.append(insightface.app.FaceAnalysis(name='buffalo_l',allowed_modules=["recognition", "detection"], providers=providers, session_options=sess_options))
        analysers[idx].prepare(ctx_id=0, det_size=(640, 640)) #640, 640
    return swappers, analysers

def download(link, filename):
    if globalsz.args['fastload']:
        import requests
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

