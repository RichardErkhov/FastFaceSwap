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
from utils import add_audio_from_video, ThreadWithReturnValue, prepare_models
import tensorflow as tf

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
args = {}
for name, value in vars(parser.parse_args()).items():
    args[name] = value
width, height = args['resolution'].split('x')
width, height = int(width), int(height)
if (args['target_path'].isdigit()):
    args['target_path'] = int(args['target_path'])
if not args['cli']:
    import tkinter as tk
    from tkinter import ttk

THREAD_SEMAPHORE = threading.Semaphore()
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.experimental.set_virtual_device_configuration(
        physical_devices[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])

generator = tf.keras.models.load_model('256_v2_big_generator_pretrain_stage1_38499.h5')
arch = 'clean'
channel_multiplier = 2
model_path = 'GFPGANv1.4.pth'
restorer = GFPGANer(
    model_path=model_path,
    upscale=0.8,
    arch=arch,
    channel_multiplier=channel_multiplier,
    bg_upsampler=None
)

def upscale_image(image):
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
    return cv2.cvtColor((np.squeeze(output, axis=0) * 255.0), cv2.COLOR_BGR2RGB) #  #
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
if not args['cli']:
    root = tk.Tk()
    root.geometry("200x300")
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


def main():
    global args, width, height
    face_swapper, face_analyser = prepare_models()
    try:
        input_face = cv2.imread(args['face'])
        source_face = sorted(face_analyser.get(input_face), key=lambda x: x.bbox[0])[0]
    except:
        print("You forgot to add the input face")
        exit()
        
    def face_analyser_thread(frame):
        faces = face_analyser.get(frame)
        bboxes = []
        for face in faces:
            bboxes.append(face.bbox)
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
                            facex = upscale_image(facer)
                        else:
                            with THREAD_SEMAPHORE:
                                cropped_faces, restored_faces, facex = restorer.enhance(
                                    facer,
                                    has_aligned=False,
                                    only_center_face=False,
                                    paste_back=True
                                )
                    else:
                        if args['face_enhancer'] == 'gfpgan':
                            with THREAD_SEMAPHORE:
                                cropped_faces, restored_faces, facex = restorer.enhance(
                                    facer,
                                    has_aligned=False,
                                    only_center_face=False,
                                    paste_back=True
                                )
                        elif args['face_enhancer'] == 'ffe':
                            facex = upscale_image(facer)
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
            cropped_faces, restored_faces, image = restorer.enhance(
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
        for i in range(int(args['threads'])):
            ret, frame = cap.read()
            if not ret:
                break
            temp.append(ThreadWithReturnValue(target=face_analyser_thread, args=(frame,)))
            temp[-1].start()
        while cap.isOpened():
            try:
                ret, frame = cap.read()
                if not ret:
                    break
                temp.append(ThreadWithReturnValue(target=face_analyser_thread, args=(frame,)))
                temp[-1].start()
                while len(temp) >= int(args['threads']):
                    bbox, frame = temp.pop(0).join()
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
            out.write(frame)
            progressbar.update(1)
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