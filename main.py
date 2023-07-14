import argparse
import os
import psutil
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--face', help='use this face', dest='face', default="face.jpg")
parser.add_argument('-t', '--target', help='replace this face. If camera, use integer like 0',default="0", dest='target_path')
parser.add_argument('-o', '--output', help='path to output of the video',default="video.mp4", dest='output')
parser.add_argument('-cam-fix', '--camera-fix', help='fix for logitech cameras that start for 40 seconds in default mode.', dest='camera_fix', action='store_true')
parser.add_argument('-res', '--resolution', help='camera resolution, given in format WxH (ex 1920x1080). Is set for camera mode only',default="1920x1080", dest='resolution')
parser.add_argument('--threads', help='amount of gpu threads',default="4", dest='threads')
parser.add_argument('--image', help='Include if the target is image', dest='image', action='store_true')
parser.add_argument('--cli', help='run in cli mode, turns off preview and now accepts switch of face enhancer from the command', dest='cli', action='store_true')
parser.add_argument('--face-enhancer', help='face enhancer, choice works only in cli mode. In gui mode, you need to choose from gui', dest='face_enhancer', default='none', choices=['none','gfpgan', 'ffe', 'codeformer', 'gpfgan_onnx'])
parser.add_argument('--no-face-swapper', '--no-swapper', help='disables face swapper', dest='no_faceswap', action='store_true')
parser.add_argument('--preview-mode', help='experimental: preview mode', dest='preview', action='store_true')
parser.add_argument('--experimental', help='experimental mode, enables features like buffered video reader', dest='experimental', action='store_true')
parser.add_argument('--no-cuda', help='no cuda should be used', dest='nocuda', action='store_true')
parser.add_argument('--low-memory', '--lowmem', help='low memory usage attempt', dest='lowmem', action='store_true')
parser.add_argument('--batch', help='batch processing mode, after the argument write which suffix should the output files have', dest='batch', default='')
#parser.add_argument('--extract-target-frames', help='extract frames from target video. After argument write the path to folder', dest='extract_target', default="")
parser.add_argument('--extract-output-frames', help='extract frames from output video. After argument write the path to folder', dest='extract_output', default="")
parser.add_argument('--codeformer-fidelity', help='sets up codeformer\'s fidelity if used with cli mode',default=0.1, dest='codeformer_fidelity')
parser.add_argument('--blend', help='works with cli, blending amount from 0.0 to 1.0', default=1.0, dest='alpha')
parser.add_argument('--codeformer-skip_if_no_face', help='works only in cli. Skip codeformer if no face found', dest='codeformer_skip_if_no_face', action='store_true')
parser.add_argument('--codeformer-face-upscale', help='works only in cli. Upscale the face using codeformer', dest='codeformer_face_upscale', action='store_true')
parser.add_argument('--codeformer-background-enhance', help='works only in cli. Enhance the background using codeformer', dest='codeformer_background_enhance', action='store_true')
parser.add_argument('--codeformer-upscale', help='works with cli, the amount of upscale to apply to the frame using codeformer', default=1, dest='codeformer_upscale')
parser.add_argument('--select-face', help='change the face you want, not all faces. After the argument add the path to the image with face from the video', dest='selective', default='')
args = {}
for name, value in vars(parser.parse_args()).items():
    args[name] = value
width, height = args['resolution'].split('x')
width, height = int(width), int(height)
if args['batch'] != "" and not args['batch'].endswith(".mp4"):
    args['batch'] += '.mp4'
#if args['extract_target'] != '':
#    os.makedirs(args['extract_target'])
if args['extract_output'] != '':
    os.makedirs(args['extract_output'])

#if args['cli']:
    #testx = input("Are you sure you want to extract frames from videos? It will be done in the background (yes for yes and anything else for no):")
    #if testx == 'yes':
        #if args['batch'] == ''
#just a fix, sometimes speeds up things
os.environ['OMP_NUM_THREADS'] = '1'
from types import NoneType
from threading import Thread
from tqdm import tqdm
import numpy as np
from gfpgan import GFPGANer
import threading, os, torch, time, cv2
from plugins.codeformer_app_cv2 import inference_app as codeformer
from utils import *
if not args['lowmem']:
    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    #tf.config.experimental.set_virtual_device_configuration(
    #        physical_devices[0],
    #        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)])
    prepare()
if args['experimental']:
    try:
        from imutils.video import FileVideoStream
    except ImportError:
        print("In the experimental mode, you have to pip install imutils")
        exit()
def select_face():
    global args, select_face_label
    filex = askopenfilename(title="Select a face")
    if filex:
        args['face'] = filex
    select_face_label.config(text=f'Face filename: {args["face"]}')

def select_target():
    global args, select_target_label
    if args['batch'] == "":
        filex = askopenfilename(title="Select a target")
    else:
        filex = askdirectory(initialdir="target")
    if filex:
        args['target_path'] = filex
    select_target_label.config(text=f'Target filename: {args["target_path"]}')
def select_camera():
    global args, select_target_label
    args["target_path"] = "0"
    select_target_label.config(text=f'Target filename: {args["target_path"]}')
def select_output():
    global args, select_output_label
    if args['batch'] == "":
        filename, ext = 'output.mp4', '.mp4'
        if args['image']:
            filename, ext = 'output.png', '.png'
        filex = asksaveasfilename(initialfile=filename, defaultextension=ext, filetypes=[("All Files","*.*"),("Videos","*.mp4")])
    else:
        filex = askdirectory(initialdir="output")
    if filex:
        args['output'] = filex
    select_output_label.config(text=f'Output filename: {args["output"]}')

    
if not args['cli']:
    import tkinter as tk
    from tkinter import ttk
    from tkinter.filedialog import asksaveasfilename, askdirectory, askopenfilename
    def finish(menu):
        global thread_amount_temp
        thread_amount_temp = thread_amount_input.get()
        menu.destroy()
    menu = tk.Tk()
    menu.geometry("500x500")
    button_start_program = tk.Button(menu, text="Start Program", command=lambda: finish(menu))
    button_start_program.pack()
    select_face_label = tk.Label(text=f'Face filename: {args["face"]}')
    select_face_label.pack()
    button_select_face = tk.Button(menu, text='Select face', command=select_face)
    button_select_face.pack()
    select_target_label = tk.Label(text=f'Target filename: {args["target_path"]}')
    select_target_label.pack()
    button_select_target = tk.Button(menu, text='Select target', command=select_target)
    button_select_target.pack()
    button_select_camera = tk.Button(menu, text='run from camera', command=select_camera)
    button_select_camera.pack()
    select_output_label = tk.Label(text=f'output filename: {args["output"]}')
    select_output_label.pack()
    button_select_output = tk.Button(menu, text='Select output', command=select_output)
    button_select_output.pack()
    thread_amount_label = tk.Label(menu, text='Select the number of threads')
    thread_amount_label.pack()
    thread_amount_input = tk.Entry(menu)
    thread_amount_input.pack()
    menu.mainloop()
    if thread_amount_temp != "":
        args['threads'] = int(thread_amount_temp)

if (args['target_path'].isdigit()):
    args['target_path'] = int(args['target_path'])
if args['preview'] and isinstance(args['target_path'], int):
    show_error()
    exit()
if args['preview'] and args['cli']:
    print("Preview mode does not work with cli, so please use GUI")
    exit()
THREAD_SEMAPHORE = threading.Semaphore()
arch = 'clean'
channel_multiplier = 2
model_path = 'GFPGANv1.4.pth'
restorer = None
generator = None
device = torch.device(0)
if not args['nocuda']:
    gpu_memory_total = round(torch.cuda.get_device_properties(device).total_memory / 1024**3,2)  # Convert bytes to GB
adjust_x1 = 50
adjust_y1 = 50
adjust_x2 = 50
adjust_y2 = 50
def load_restorer():
    global restorer
    if isinstance(restorer, NoneType):
        restorer = GFPGANer(
            model_path=model_path,
            upscale=0.8,
            arch=arch,
            channel_multiplier=channel_multiplier,
            bg_upsampler=None
        )
    return restorer

def load_generator():
    global generator
    if isinstance(generator, NoneType):
        #model_path = 'generator.onnx'
        #providers = rt.get_available_providers()
        #generator = rt.InferenceSession(model_path, providers=providers)
        generator = tf.keras.models.load_model('256_v2_big_generator_pretrain_stage1_38499.h5')#, custom_objects={'Mish': Mish})
    return generator
def set_adjust_value():
    global adjust_x1, adjust_y1, adjust_x2, adjust_y2
    try:
        adjust_x1 = int(entry_x1.get())
        adjust_y1 = int(entry_y1.get())
        adjust_x2 = int(entry_x2.get())
        adjust_y2 = int(entry_y2.get())
    except:
        print("YOU HAVE TO PUT INTEGERS")
    entry_x1.delete(0, tk.END)
    entry_x2.delete(0, tk.END)
    entry_y1.delete(0, tk.END)
    entry_y2.delete(0, tk.END)
    entry_x1.insert(0, adjust_x1)
    entry_y1.insert(0, adjust_y1)
    entry_x2.insert(0, adjust_x2)
    entry_y2.insert(0, adjust_y2)
def count_frames(video_path):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    return total_frames

def get_system_usage():
    # Get RAM usage in GB
    ram_usage = round(psutil.virtual_memory().used / 1024**3, 1)

    # Get total RAM in GB
    total_ram = round(psutil.virtual_memory().total / 1024**3, 1)

    # Get CPU usage in percentage
    cpu_usage = round(psutil.cpu_percent(), 0)
    return ram_usage, total_ram, cpu_usage

if not args['cli']:
    root = tk.Tk()
    if not args['preview']:
        root.geometry("250x600")
    else:
        root.geometry("250x730")
    faceswapper_checkbox_var = tk.IntVar(value=1)
    faceswapper_checkbox = ttk.Checkbutton(root, text="Face swapper", variable=faceswapper_checkbox_var)
    faceswapper_checkbox.pack()
    checkbox_var = tk.IntVar()
    checkbox = ttk.Checkbutton(root, text="Face enhancer", variable=checkbox_var)
    checkbox.pack()
    enhancer_choice = tk.StringVar(value='fastface enhancer')
    choices = ['fastface enhancer', 'gfpgan', 'codeformer', 'gfpgan onnx']

    if not args['lowmem']:
        choices.remove('fastface enhancer')

    dropdown = ttk.OptionMenu(root, enhancer_choice, enhancer_choice.get(), *choices)
    dropdown.pack()


    show_bbox_var = tk.IntVar()
    show_bbox = ttk.Checkbutton(root, text="draw bounding box around faces", variable=show_bbox_var)
    show_bbox.pack()
    label = tk.Label(root, text="bounding box adjustment")
    label.pack()

    label = tk.Label(root, text="up")
    label.pack()

    entry_y1 = tk.Entry(root)
    entry_y1.pack() 
    entry_y1.insert(0, adjust_y1)

    label = tk.Label(root, text="right")
    label.pack()

    entry_x2 = tk.Entry(root)
    entry_x2.pack() 
    entry_x2.insert(0, adjust_x2)
    label = tk.Label(root, text="left")
    label.pack()

    entry_x1 = tk.Entry(root)
    entry_x1.pack() 
    entry_x1.insert(0, adjust_x1)
    label = tk.Label(root, text="down")
    label.pack()

    entry_y2 = tk.Entry(root)
    entry_y2.pack() 
    entry_y2.insert(0, adjust_y2)

    button = tk.Button(root, text="Set Values", command=set_adjust_value)
    button.pack()  # Add the button to the window
    codeformer_fidelity = 0.1
    def on_codeformer_slider_move(value):
        global codeformer_fidelity
        codeformer_fidelity = float(value)
    label = tk.Label(root, text="Codeformer fidelity")
    label.pack()
    codeformer_slider = tk.Scale(root, from_=0.1, to=2.0, resolution=0.1,  orient=tk.HORIZONTAL, command=on_codeformer_slider_move)
    codeformer_slider.pack()
    alpha = 0.0
    def alpha_slider_move(value):
        global alpha
        alpha = float(value)
    label = tk.Label(root, text="blender")
    label.pack()
    alpha_slider = tk.Scale(root, from_=0.0, to=1.0, resolution=0.1,  orient=tk.HORIZONTAL, command=alpha_slider_move)
    alpha_slider.pack()
    alpha_slider.set(1.0)
    
    codeformer_skip_if_no_face_var = tk.IntVar()
    codeformer_skip_if_no_face = ttk.Checkbutton(root, text="Skip codeformer if not face is found", variable=codeformer_skip_if_no_face_var)
    codeformer_skip_if_no_face.pack()
    codeformer_upscale_face_var = tk.IntVar()
    codeformer_upscale_face = ttk.Checkbutton(root, text="Upscale face using codeformer", variable=codeformer_upscale_face_var)
    codeformer_upscale_face.pack()
    codeformer_upscale_face_var.set(1)
    codeformer_enhance_background_var = tk.IntVar()
    codeformer_enhance_background = ttk.Checkbutton(root, text="Enhance background using codeformer", variable=codeformer_enhance_background_var)
    codeformer_enhance_background.pack()
    codeformer_upscale_amount_value = 1
    def codeformer_upscale_amount_move(value):
        global codeformer_upscale_amount_value
        codeformer_upscale_amount_value = int(value)
    codeformer_upscale_amount = tk.Scale(root, from_=1, to=3, resolution=1,  orient=tk.HORIZONTAL, command=codeformer_upscale_amount_move)
    codeformer_upscale_amount.pack()
    codeformer_upscale_amount.set(1)
    
    if not args['preview'] and not isinstance(args['target_path'], int):
        progress_label = tk.Label(root)
        progress_label.pack()
    usage_label1 = tk.Label(root)
    usage_label1.pack()
    if not args['nocuda']:
        usage_label2 = tk.Label(root)
        usage_label2.pack()
    if args['preview']:
        frame_index = 0
        def on_slider_move(value):
            global frame_index
            frame_index = int(value)
        def edit_index(amount):
            global frame_index
            frame_index += amount
            slider.set(frame_index)
        frame_amount = count_frames(args['target_path'])
        label = tk.Label(root, text="frame number")
        label.pack()
        slider = tk.Scale(root, from_=1, to=frame_amount, orient=tk.HORIZONTAL, command=on_slider_move)
        slider.pack()
        frame_count_label = tk.Label(root, text=f"total frames: {frame_amount}")
        frame_count_label.pack(fill=tk.X)
        button_width = root.winfo_width() // 2
        frame_back_button = tk.Button(root, text='<', width=button_width, command=lambda: edit_index(-1))
        frame_back_button.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        frame_forward_button = tk.Button(root, text='>', width=button_width, command=lambda: edit_index(1))
        frame_forward_button.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    def update_progress_bar(length, progress, total, gpu_usage, vram_usage, total_vram):
        if not args['preview'] and not isinstance(args['target_path'], int):
            filled_length = int(length * progress // total)
            bar = '█' * filled_length + '—' * (length - filled_length)
            percent = round(100.0 * progress / total, 1)
            progress_text = f'Progress: |{bar}| {percent}% {progress}/{total}'
            progress_label['text'] = progress_text
        if not args['nocuda']:
            usage_label1['text'] = f"gpu usage: {gpu_usage}%|VRAM usage: {vram_usage}/{total_vram}GB"
            ram_usage, total_ram, cpu_usage = get_system_usage()
            usage_label2['text'] = f"cpu usage: {cpu_usage}%|RAM usage: {ram_usage}/{total_ram}GB"
        else:
            ram_usage, total_ram, cpu_usage = get_system_usage()
            usage_label1['text'] = f"cpu usage: {cpu_usage}%|RAM usage: {ram_usage}/{total_ram}GB"
        #progress_var.set(text=progress_text)
        root.update()
gfpgan_onnx_model = None
def load_gfpganonnx():
    global gfpgan_onnx_model
    if isinstance(gfpgan_onnx_model, NoneType):
        gfpgan_onnx_model = GFPGAN_onnxruntime(model_path="GFPGANv1.4.onnx")
    return gfpgan_onnx_model

def restorer_enhance(facer):
    with THREAD_SEMAPHORE:
        cropped_faces, restored_faces, facex = load_restorer().enhance(
            facer,
            has_aligned=False,
            only_center_face=False,
            paste_back=True
        )
    return facex

def create_cap():
    global width, height
    if not args['experimental']:
        if args['camera_fix'] == True:
            cap = cv2.VideoCapture(args['target_path'], cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(args['target_path'])
        if isinstance(args['target_path'], int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
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
        cap = cv2.VideoCapture(args['target_path'])
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        del cap
        cap = FileVideoStream(args['target_path']).start()
        time.sleep(1.0)
    # Create a VideoWriter object to save the processed video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    name = args['output']
    if isinstance(args['target_path'], str):
        name = f"{args['output']}_temp.mp4"
    out = cv2.VideoWriter(name, fourcc, fps, (width, height))
    frame_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return [cap, fps, width, height, out, name, args['target_path'], frame_number]

def create_batch_cap(file):
    if not args['experimental']:
        if args['camera_fix'] == True:
            print("no need for camera_fix, there's not camera available in batch processing")
        cap = cv2.VideoCapture(os.path.join(args['target_path'], file))
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
        cap = cv2.VideoCapture(os.path.join(args['target_path'], file))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        del cap
        # yes, might overflow is too many files, well, it's experimental lol, what do you expect?
        cap = FileVideoStream(os.path.join(args['target_path'], file)).start() 
        time.sleep(1.0)

    # Create a VideoWriter object to save the processed video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    name = os.path.join(args['output'], f"{file}{args['batch']}_temp.mp4")#f"{args['output']}_temp{args['batch']}.mp4"
    out = cv2.VideoWriter(name, fourcc, fps, (width, height))
    frame_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return [cap, fps, width, height, out, name, file, frame_number]
def face_analyser_thread(frame):
    global alpha
    if not args['cli']:
        test1 = alpha != 0
    else:
        test1 = args['alpha'] != 0
    if test1:        
        original_frame = frame
        faces = face_analyser.get(frame)
        bboxes = []
        for face in faces:
            if args['selective'] != '':
                a = target_embedding.normed_embedding
                b = face.normed_embedding
                _, allow = compute_cosine_distance(a,b , 0.75)
                if not allow:
                    continue
            bboxes.append(face.bbox)
            ttest1 = False
            if not args['cli']:
                if faceswapper_checkbox_var.get() == True:
                    ttest1=True
            if not args['no_faceswap'] and (ttest1 == True or args['cli']):
                frame = face_swapper.get(frame, face, source_face, paste_back=True)
            try:
                test1 = checkbox_var.get() == 1 
                test2 = not enhancer_choice.get() == "codeformer"
            except:
                test1 = False
                test2 = False
            if (test1 and test2) or (args['face_enhancer'] != 'none' and args['cli'] and args['face_enhancer'] != 'codeformer'):
                try:
                    i = face.bbox
                    x1, y1, x2, y2 = int(i[0]),int(i[1]),int(i[2]),int(i[3])
                    x1 = max(x1-adjust_x1, 0)
                    y1 = max(y1-adjust_y1, 0)
                    x2 = min(x2+adjust_x2, width)
                    y2 = min(y2+adjust_y2, height)
                    facer = frame[y1:y2, x1:x2]
                    if not args['cli']:
                        if enhancer_choice.get() == "fastface enhancer":
                            facex = upscale_image(facer, load_generator())
                        elif enhancer_choice.get() == "gfpgan":
                            facex = restorer_enhance(facer)
                        elif enhancer_choice.get() == "gfpgan onnx":
                            facex, _ = load_gfpganonnx().forward(facer)
                    else:
                        if args['face_enhancer'] == 'gfpgan':
                            facex = restorer_enhance(facer)
                        elif args['face_enhancer'] == 'ffe':
                            facex = upscale_image(facer, load_generator())
                        elif args['face_enhancer'] == "gpfgan_onnx":
                            facex, _ = load_gfpganonnx().forward(facer)
                    facex = cv2.resize(facex, ((x2-x1), (y2-y1)))
                    frame[y1:y2, x1:x2] = facex
                except Exception as e:
                    print(e)
        if not args['cli']:
            if enhancer_choice.get() == "codeformer" and checkbox_var.get() == 1 : 
                #frame, background enhance bool true, face upscample bool true, upscale int 2,
                # codeformer fidelity float 0.8, skip_if_no_face bool false 
                frame = codeformer(frame, codeformer_enhance_background_var.get(), codeformer_upscale_face_var.get(), codeformer_upscale_amount_value, codeformer_fidelity, codeformer_skip_if_no_face_var.get())
        else:
            if args['face_enhancer'] == 'codeformer':
                frame = codeformer(frame, args['codeformer_background_enhance'], args['codeformer_face_upscale'], args['codeformer_upscale'], float(args['codeformer_fidelity']), args['codeformer_skip_if_no_face'])
        if not args['cli']:
            test1 = alpha != 1
        else:
            test1 = args['alpha'] != 1
        if test1:
            frame = merge_face(frame, original_frame, alpha)
        return bboxes, frame
    return [], frame


def is_video_file(filename):
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.webm']  # Add more extensions as needed
    _, ext = os.path.splitext(filename)
    return ext.lower() in video_extensions

def get_embedding(face_image):
    try:
        return face_analyser.get(face_image)
    except IndexError:
        return None
    
def main():
    global args, width, height, frame_index, face_analyser, face_swapper, source_face, progress_var, target_embedding, count, frame_number, listik
    face_swapper, face_analyser = prepare_models(args)
    input_face = cv2.imread(args['face'])
    source_face = sorted(face_analyser.get(input_face), key=lambda x: x.bbox[0])[0]
    target_embedding = None
    gpu_usage = 0
    vram_usage = 0
    if args['selective'] != '':
        im = cv2.imread(args['selective'])
        #im = cv2.resize(im, (640, 640))
        target_embedding = get_embedding(im)[0]
    if args['image'] == True :
        image = cv2.imread(args['target_path'])
        bbox, image = face_analyser_thread(image)
        try:
            test1 = checkbox_var.get() == 1
        except:
            test1 = False
        if test1 or (args['face_enhancer'] != 'none' and args['cli']):
            image = restorer_enhance(image)
        cv2.imwrite(args['output'], image)
        print("image processing finished")
        return 
    caps = []
    if args['batch'] == '':
        caps.append(create_cap())
    else:
        for file in os.listdir(args['target_path']):
            if is_video_file(file):
                caps.append(create_batch_cap(file))
    for cap, fps, width, height, out, name, file, frame_number in caps:
        #root.after(1, update_progress_length, frame_number)
        #update_progress_bar( 10, 0, frame_number)
        count = 0
        with tqdm(total=frame_number) as progressbar:
            temp = []
            bbox = []
            start = time.time()
            if not args['preview']:
                for i in range(int(args['threads'])):
                    if args['experimental']:
                        frame = cap.read()
                        if isinstance(frame, NoneType):
                            break
                    else:
                        ret, frame = cap.read()
                        if not ret:
                            break
                    temp.append(ThreadWithReturnValue(target=face_analyser_thread, args=(frame,)))
                    temp[-1].start()
            while True:
                try:
                    if not args['preview']:
                        if args['experimental']:
                            frame = cap.read()
                            if isinstance(frame, NoneType): #== None:
                                break
                        else:
                            ret, frame = cap.read()
                            if not ret:
                                break
                        temp.append(ThreadWithReturnValue(target=face_analyser_thread, args=(frame,)))
                        temp[-1].start()
                        while len(temp) >= int(args['threads']):
                            bbox, frame = temp.pop(0).join()
                    else:
                        bbox, frame = face_analyser_thread(get_nth_frame(cap, frame_index))
                    if not args['cli']:
                        if show_bbox_var.get() == 1:
                            for i in bbox: 
                                x1, y1, x2, y2 = int(i[0]),int(i[1]),int(i[2]),int(i[3])
                                x1 = max(x1-adjust_x1, 0)
                                y1 = max(y1-adjust_y1, 0)
                                x2 = min(x2+adjust_x2, width)
                                y2 = min(y2+adjust_y2, height)
                                color = (0, 255, 0)  # Green color (BGR format)
                                thickness = 2  # Line thickness
                                cv2.rectangle(frame, (x1,y1), (x2,y2), color, thickness)
                    if time.time() - start > 1:
                        start = time.time()
                        if not args['nocuda']:
                            vram_usage, gpu_usage = round(gpu_memory_total - torch.cuda.mem_get_info(device)[0] / 1024**3,2), torch.cuda.utilization(device=device)
                            progressbar.set_description(f"VRAM: {vram_usage}/{gpu_memory_total} GB, usage: {gpu_usage}%")
                    if not args['cli']:
                        count += 1
                        if not args['nocuda']:
                            listik = [count, frame_number,gpu_usage, vram_usage,gpu_memory_total]
                        else:
                            listik = [count, frame_number, 0, 0, 0]
                        cv2.imshow('Face Detection', frame)
                    if not args['preview']:
                        out.write(frame)
                    if args['extract_output'] != '':
                        cv2.imwrite(os.path.join(args['extract_output'], os.path.basename(file), f"frame_{count:05d}.png"), frame)
                    progressbar.update(1)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except KeyboardInterrupt:
                    break
            for i in temp:
                bbox, frame = i.join()
                if not args['cli']:
                    if show_bbox_var.get() == 1:
                        for i in bbox: 
                            x1, y1, x2, y2 = int(i[0]),int(i[1]),int(i[2]),int(i[3])
                            x1 = max(x1-adjust_x1, 0)
                            y1 = max(y1-adjust_y1, 0)
                            x2 = min(x2+adjust_x2, width)
                            y2 = min(y2+adjust_y2, height)
                            color = (0, 255, 0)  # Green color (BGR format)
                            thickness = 2  # Line thickness
                            cv2.rectangle(frame, (x1,y1), (x2,y2), color, thickness)
                if time.time() - start > 1:
                    start = time.time()
                    if not args['nocuda']:
                        vram_usage, gpu_usage = round(gpu_memory_total - torch.cuda.mem_get_info(device)[0] / 1024**3,2), torch.cuda.utilization(device=device)
                        progressbar.set_description(f"VRAM: {vram_usage}/{gpu_memory_total} GB, usage: {gpu_usage}%")
                
                if not args['preview']:
                    out.write(frame)
                if args['extract_output'] != '':
                    cv2.imwrite(os.path.join(args['extract_output'], os.path.basename(file), f"frame_{count:05d}.png"), frame)
                progressbar.update(1)
                if not args['cli']:
                    cv2.imshow('Face Detection', frame)
                    #update_progress_bar(10, count, frame_number)
                if args['preview']:
                    old_number = frame_index
                    while frame_index == old_number:
                        time.sleep(0.01)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        out.release()
        cap.release()
        cv2.destroyAllWindows()
        if args['batch'] != '':
            try:
                add_audio_from_video(os.path.join(args['output'], f"{file}{args['batch']}_temp.mp4"),os.path.join(args['output'], file) ,os.path.join(args['output'], f"{file}{args['batch']}"))
                os.remove(os.path.join(args['output'], f"{file}{args['batch']}_temp.mp4"))
            except Exception as e:
                print(f"SOMETHING WENT WRONG DURING THE ADDING OF THE AUDIO TO THE VIDEO!file: {os.path.join(args['output'], file)}, error:{e}")
        else:
             if not isinstance(args['target_path'], int):
                add_audio_from_video(name, args['target_path'], args['output'])
                os.remove(name)
        
        
    print("Processing finished, you may close the window now")
    exit()
if args['batch'] != '':
    os.makedirs(args['output'], exist_ok=True)
if not args['cli']:
    listik = [0, 1, 0, 0, 0]
    threading.Thread(target=main).start()
    def update_gui():
        update_progress_bar(7, listik[0], listik[1], listik[2], listik[3], listik[4])
        root.after(300, update_gui)
    update_gui()
    root.mainloop()
else:
    main()
