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
root = tk.Tk()
root.geometry("200x200")
checkbox_var = tk.IntVar()
checkbox = ttk.Checkbutton(root, text="Face enhancer", variable=checkbox_var)
checkbox.pack()
def main():
    arch = 'clean'
    channel_multiplier = 2
    model_path = 'GFPGANv1.4.pth'
    restorer = GFPGANer(
        model_path=model_path,
        upscale=2,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=None
    )
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--face', help='use this face', dest='face', default="face.jpg")
    parser.add_argument('-t', '--target', help='replace this face. If camera, use integer like 0',default=0, dest='target_path')
    args = {}
    providers = rt.get_available_providers()
    for name, value in vars(parser.parse_args()).items():
        args[name] = value
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
    try:
        input_face = cv2.imread(args['face'])
        source_face = sorted(face_analyser.get(input_face), key=lambda x: x.bbox[0])[0]
    except:
        print("You forgot to add the input face")
        exit()
    cap = cv2.VideoCapture(args['target_path'])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    def face_analyser_thread(frame):
        faces = face_analyser.get(frame)
        bboxes = []
        for face in faces:
            bboxes.append(face.bbox)
            frame = face_swapper.get(frame, face, source_face, paste_back=True)    
        return bboxes, frame

    with tqdm() as progressbar:
        temp = []
        bbox = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            temp.append(ThreadWithReturnValue(target=face_analyser_thread, args=(frame,)))
            temp[-1].start()
            while len(temp) >= 3:
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

                
            cv2.imshow('Face Detection', frame)
            progressbar.update(1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
threading.Thread(target=main).start()
root.mainloop()