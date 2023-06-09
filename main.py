import cv2
import insightface
from threading import Thread
from tqdm import tqdm
import onnxruntime as rt
import argparse
import argparse
import cv2
import numpy as np
from gfpgan import GFPGANer
import tkinter as tk
from tkinter import ttk
import threading
import subprocess
import os
import torch
import time
device = torch.device(0)
gpu_memory_total = round(torch.cuda.get_device_properties(device).total_memory / 1024**3,2)  # Convert bytes to GB
root = tk.Tk()
root.geometry("200x200")
checkbox_var = tk.IntVar()
checkbox = ttk.Checkbutton(root, text="Face enhancer", variable=checkbox_var)
checkbox.pack()

def add_audio_from_video(video_path, audio_video_path, output_path):
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', video_path,
        '-i', audio_video_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-shortest',
        output_path
    ]

    subprocess.run(ffmpeg_cmd, check=True)
def main():
    arch = 'clean'
    channel_multiplier = 2
    model_path = 'GFPGANv1.4.pth'
    restorer = GFPGANer(
        model_path=model_path,
        upscale=1,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=None
    )
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--face', help='use this face', dest='face', default="face.jpg")
    parser.add_argument('-t', '--target', help='replace this face. If camera, use integer like 0',default="0", dest='target_path')
    parser.add_argument('-o', '--output', help='path to output of the video',default="video.mp4", dest='output')
    parser.add_argument('-cam-fix', '--camera-fix', help='fix for logitech cameras that start for 40 seconds in default mode.', dest='camera_fix', action='store_true')
    parser.add_argument('-res', '--resolution', help='camera resolution, given in format WxH (ex 1920x1080). Is set for camera mode only',default="1920x1080", dest='resolution')
    parser.add_argument('--threads', help='amount of gpu threads',default="2", dest='threads')
    parser.add_argument('--image', help='Include if the target is image', dest='image', action='store_true')
    args = {}
    providers = rt.get_available_providers()
    for name, value in vars(parser.parse_args()).items():
        args[name] = value
    width, height = args['resolution'].split('x')
    width, height = int(width), int(height)
    if (args['target_path'].isdigit()):
        args['target_path'] = int(args['target_path'])
    sess_options = rt.SessionOptions()
    sess_options.intra_op_num_threads = 8
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
    face_swapper = insightface.model_zoo.get_model("inswapper_128.onnx", session_options=sess_options, providers=providers)
    face_analyser = insightface.app.FaceAnalysis(name='buffalo_l', providers=providers)
    face_analyser.prepare(ctx_id=0, det_size=(640, 640))
    face_analyser.models.pop("landmark_3d_68")
    face_analyser.models.pop("landmark_2d_106")
    face_analyser.models.pop("genderage")
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
        return bboxes, frame
    if args['image'] == True:
        image = cv2.imread(args['target_path'])
        bbox, image = face_analyser_thread(image)

        if checkbox_var.get() == 1:
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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

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
        while cap.isOpened():
            try:
                ret, frame = cap.read()
                if not ret:
                    break
                temp.append(ThreadWithReturnValue(target=face_analyser_thread, args=(frame,)))
                temp[-1].start()
                while len(temp) >= int(args['threads']):
                    bbox, frame = temp.pop(0).join()
                '''cropped_faces, restored_faces, frame = restorer.enhance(
                    frame,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True
                )'''
                #frame = cv2.resize(frame, (1280, 720))
                if checkbox_var.get() == 1:
                    for i in bbox: 
                        x1, y1, x2, y2 = int(i[0]),int(i[1]),int(i[2]),int(i[3])
                        x1 = max(x1-50, 0)
                        y1 = max(y1-50, 0)
                        x2 = min(x2+50, 1280)
                        y2 = min(y2+50, 720)
                        face = frame[y1:y2, x1:x2]

                        try:
                            cropped_faces, restored_faces, facex = restorer.enhance(
                                face,
                                has_aligned=False,
                                only_center_face=False,
                                paste_back=True
                            )
                            facex = cv2.resize(facex, ((x2-x1), (y2-y1)))
                            #frame = blend_images(face, frame, (x1, y1, x2-x1, y2-y1))
                            #frame = blend_images(frame, face, (x1, y1))
                            '''try:
                                
                            except Exception as e:
                                print(e)'''
                            frame[y1:y2, x1:x2] = facex
                        except Exception as e:  
                            print(e)

                if time.time() - start > 1:
                    start = time.time()
                    progressbar.set_description(f"VRAM: {round(gpu_memory_total - torch.cuda.mem_get_info(device)[0] / 1024**3,2)}/{gpu_memory_total} GB, usage: {torch.cuda.utilization(device=device)}%")
                cv2.imshow('Face Detection', frame)
                out.write(frame)
                progressbar.update(1)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except KeyboardInterrupt:
                break
    out.release()
    cap.release()
    cv2.destroyAllWindows()
    add_audio_from_video(name, args['target_path'], args['output'])
    os.remove(name)
    print("Processing finished, you may close the window now")
    exit()

threading.Thread(target=main).start()
root.mainloop()