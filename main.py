from types import NoneType
import cv2
from threading import Thread
from tqdm import tqdm
import argparse
import numpy as np
from gfpgan import GFPGANer
import threading
import os
import torch
import time
from utils import add_audio_from_video, ThreadWithReturnValue, prepare_models, upscale_image
import tensorflow as tf
from tkinter import filedialog
from tkinter.filedialog import asksaveasfilename
from tkinter import messagebox
def show_error():
    messagebox.showerror("Error", "Preview mode does not work with camera, so please use normal mode")
def mish_activation(x):
    return x * tf.keras.activations.tanh(tf.keras.activations.softplus(x))

class Mish(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Mish, self).__init__()

    def call(self, inputs):
        return mish_activation(inputs)
tf.keras.utils.get_custom_objects().update({'Mish': Mish})
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--face', help='use this face', dest='face', default="face.jpg")
parser.add_argument('-t', '--target', help='replace this face. If camera, use integer like 0',default="0", dest='target_path')
parser.add_argument('-o', '--output', help='path to output of the video',default="video.mp4", dest='output')
parser.add_argument('-cam-fix', '--camera-fix', help='fix for logitech cameras that start for 40 seconds in default mode.', dest='camera_fix', action='store_true')
parser.add_argument('-res', '--resolution', help='camera resolution, given in format WxH (ex 1920x1080). Is set for camera mode only',default="1920x1080", dest='resolution')
parser.add_argument('--threads', help='amount of gpu threads',default="2", dest='threads')
parser.add_argument('--image', help='Include if the target is image', dest='image', action='store_true')
parser.add_argument('--cli', help='run in cli mode, turns off preview and now accepts switch of face enhancer from the command', dest='cli', action='store_true')
parser.add_argument('--face-enhancer', help='face enhancer, choice works only in cli mode. In gui mode, you need to choose from gui', dest='face_enhancer', default='none', choices=['none','gfpgan', 'ffe'])
parser.add_argument('--no-face-swapper', '--no-swapper', help='disables face swapper', dest='no_faceswap', action='store_true')
parser.add_argument('--preview-mode', help='experimental: preview mode', dest='preview', action='store_true')
args = {}
for name, value in vars(parser.parse_args()).items():
    args[name] = value
width, height = args['resolution'].split('x')
width, height = int(width), int(height)
def select_face():
    global args, select_face_label
    args['face'] = filedialog.askopenfilename(title="Select a face")
    select_face_label.config(text=f'Face filename: {args["face"]}')

def select_target():
    global args, select_target_label
    args['target_path'] = filedialog.askopenfilename(title="Select a target")
    select_target_label.config(text=f'Target filename: {args["target_path"]}')
def select_camera():
    global args, select_target_label
    args["target_path"] = "0"
    select_target_label.config(text=f'Target filename: {args["target_path"]}')
def select_output():
    global args, select_output_label
    filename, ext = 'output.mp4', '.mp4'
    if args['image']:
        filename, ext = 'output.png', '.png'
    args['output'] = asksaveasfilename(initialfile=filename, defaultextension=ext, filetypes=[("All Files","*.*"),("Videos","*.mp4")])
    select_output_label.config(text=f'Output filename: {args["output"]}')
    
if not args['cli']:
    import tkinter as tk
    from tkinter import ttk
    def eee():
        print("run")
        while True:
            time.sleep(1)
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
THREAD_SEMAPHORE = threading.Semaphore()
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
#tf.config.experimental.set_virtual_device_configuration(
#        physical_devices[0],
#        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)])


arch = 'clean'
channel_multiplier = 2
model_path = 'GFPGANv1.4.pth'
restorer = None
generator = None
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
        generator = tf.keras.models.load_model('256_v2_big_generator_pretrain_stage1_38499.h5')#, custom_objects={'Mish': Mish})
    return generator

device = torch.device(0)
gpu_memory_total = round(torch.cuda.get_device_properties(device).total_memory / 1024**3,2)  # Convert bytes to GB
adjust_x1 = 50
adjust_y1 = 50
adjust_x2 = 50
adjust_y2 = 50
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
if not args['cli']:
    root = tk.Tk()
    if not args['preview']:
        root.geometry("200x330")
    else:
        root.geometry("250x420")
    faceswapper_checkbox_var = tk.IntVar(value=1)
    faceswapper_checkbox = ttk.Checkbutton(root, text="Face swapper", variable=faceswapper_checkbox_var)
    faceswapper_checkbox.pack()
    checkbox_var = tk.IntVar()
    checkbox = ttk.Checkbutton(root, text="Face enhancer", variable=checkbox_var)
    checkbox.pack()
    enhancer_choice = tk.IntVar()
    r1 = tk.Radiobutton(root, text='fastface enhancer', variable=enhancer_choice, value = 0)
    r1.pack()
    r2 = tk.Radiobutton(root, text='gfpgan', variable=enhancer_choice, value = 1)
    r2.pack()



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

    if args['preview']:
        frame_index = 0
        def on_slider_move(value):
            global frame_index
            frame_index = int(value)
            #print("Slider value:", value)
        def edit_index(amount):
            global frame_index
            frame_index += amount
            slider.set(frame_index)
        frame_amount = count_frames(args['target_path'])
        slider = tk.Scale(root, from_=1, to=frame_amount, orient=tk.HORIZONTAL, command=on_slider_move)
        slider.pack()
        frame_count_label = tk.Label(root, text=str(frame_amount))
        frame_count_label.pack(fill=tk.X)
        button_width = root.winfo_width() // 2
        frame_back_button = tk.Button(root, text='<', width=button_width, command=lambda: edit_index(-1))
        frame_back_button.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        frame_forward_button = tk.Button(root, text='>', width=button_width, command=lambda: edit_index(1))
        frame_forward_button.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

def get_nth_frame(cap, number):
    cap.set(cv2.CAP_PROP_POS_FRAMES, number)
    ret, frame = cap.read()
    if ret:
        return frame
    return None

def main():
    global args, width, height, frame_index
    face_swapper, face_analyser = prepare_models(args)
    #try:
    input_face = cv2.imread(args['face'])
    source_face = sorted(face_analyser.get(input_face), key=lambda x: x.bbox[0])[0]
    #except:
    #    print("You forgot to add the input face")
    #    exit()
        
    def face_analyser_thread(frame):
        faces = face_analyser.get(frame)
        bboxes = []
        for face in faces:
            bboxes.append(face.bbox)
            ttest1 = False
            if not args['cli']:
                if faceswapper_checkbox_var.get() == True:
                    ttest1=True
            if not args['no_faceswap'] and ttest1 == True:
                frame = face_swapper.get(frame, face, source_face, paste_back=True)   
            try:
                test1 = checkbox_var.get() == 1 
            except:
                test1 = False
            if test1 or (args['face_enhancer'] != 'none' and args['cli']):
                try:

                    i = face.bbox
                    x1, y1, x2, y2 = int(i[0]),int(i[1]),int(i[2]),int(i[3])
                    x1 = max(x1-adjust_x1, 0)
                    y1 = max(y1-adjust_y1, 0)
                    x2 = min(x2+adjust_x2, width)
                    y2 = min(y2+adjust_y2, height)
                    facer = frame[y1:y2, x1:x2]
                    if not args['cli']:
                        if enhancer_choice.get() == 0:
                            facex = upscale_image(facer, load_generator())
                        else:
                            with THREAD_SEMAPHORE:
                                cropped_faces, restored_faces, facex = load_restorer().enhance(
                                    facer,
                                    has_aligned=False,
                                    only_center_face=False,
                                    paste_back=True
                                )
                    else:
                        if args['face_enhancer'] == 'gfpgan':
                            with THREAD_SEMAPHORE:
                                cropped_faces, restored_faces, facex = load_restorer().enhance(
                                    facer,
                                    has_aligned=False,
                                    only_center_face=False,
                                    paste_back=True
                                )
                        elif args['face_enhancer'] == 'ffe':
                            facex = upscale_image(facer, load_generator())
                    facex = cv2.resize(facex, ((x2-x1), (y2-y1)))
                    frame[y1:y2, x1:x2] = facex
                except Exception as e:
                    print(e)

        return bboxes, frame
    if args['image'] == True :
        image = cv2.imread(args['target_path'])
        bbox, image = face_analyser_thread(image)
        try:
            test1 = checkbox_var.get() == 1
        except:
            test1 = False
        if test1 or (args['face_enhancer'] != 'none' and args['cli']):
            cropped_faces, restored_faces, image = load_restorer().enhance(
                        image,
                        has_aligned=False,
                        only_center_face=False,
                        paste_back=True
                    )
        cv2.imwrite(args['output'], image)
        print("image processing finished")
        return 

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
    # Create a VideoWriter object to save the processed video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    name = args['output']
    if isinstance(args['target_path'], str):
        name = f"{args['output']}_temp.mp4"
    out = cv2.VideoWriter(name, fourcc, fps, (width, height))
    with tqdm() as progressbar:
        temp = []
        bbox = []
        start = time.time()
        if not args['preview']:
            for i in range(int(args['threads'])):
                ret, frame = cap.read()
                if not ret:
                    break
                temp.append(ThreadWithReturnValue(target=face_analyser_thread, args=(frame,)))
                temp[-1].start()
        while cap.isOpened():
            try:
                if not args['preview']:
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
                    progressbar.set_description(f"VRAM: {round(gpu_memory_total - torch.cuda.mem_get_info(device)[0] / 1024**3,2)}/{gpu_memory_total} GB, usage: {torch.cuda.utilization(device=device)}%")
                if not args['cli']:
                    cv2.imshow('Face Detection', frame)
                out.write(frame)
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
                progressbar.set_description(f"VRAM: {round(gpu_memory_total - torch.cuda.mem_get_info(device)[0] / 1024**3,2)}/{gpu_memory_total} GB, usage: {torch.cuda.utilization(device=device)}%")
            if not args['cli']:
                cv2.imshow('Face Detection', frame)
            if not args['preview']:
                out.write(frame)
            progressbar.update(1)
            if args['preview']:
                old_number = frame_index
                while frame_index == old_number:
                    time.sleep(0.01)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    out.release()
    cap.release()
    cv2.destroyAllWindows()
    if not isinstance(args['target_path'], int):
        add_audio_from_video(name, args['target_path'], args['output'])
        os.remove(name)
    print("Processing finished, you may close the window now")
    exit()

if not args['cli']:
    threading.Thread(target=main).start()
    root.mainloop()
else:
    main()