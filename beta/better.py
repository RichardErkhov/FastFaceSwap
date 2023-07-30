import argparse
import os
import globalsx
import copy
import magic    #pip install python-magic-bin https://github.com/Yelp/elastalert/issues/1927
# prepare some arguments 
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
globalsx.args = {}
for name, value in vars(parser.parse_args()).items():
    globalsx.args[name] = value
if globalsx.args['batch'] != "" and not globalsx.args['batch'].endswith(".mp4"):
    globalsx.args['batch'] += '.mp4'
#if args['extract_target'] != '':
#    os.makedirs(args['extract_target'])
if globalsx.args['extract_output'] != '':
    os.makedirs(globalsx.args['extract_output'])
# =====================================================
from types import NoneType
from threading import Thread
from tqdm import tqdm
import numpy as np
from gfpgan import GFPGANer
import threading, os, torch, time, cv2
from plugins.codeformer_app_cv2 import inference_app as codeformer
from utilities import *
import time
if not globalsx.args['lowmem']:
    import tensorflow as tf
    if not globalsx.args['nocuda']:
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    #TODO EITHER LIMIT THE VRAM, OR JUST CONVERT THE MODEL TO ONNX TO REMOVE TENSORFLOW
    #tf.config.experimental.set_virtual_device_configuration(
    #        physical_devices[0],
    #        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
    prepare()

if globalsx.args['experimental']:
    try:
        from imutils.video import FileVideoStream
    except ImportError:
        print("In the experimental mode, you have to pip install imutils")
        exit()
#video consists of:
#video_type: can be image (0) or video (1)
#cap: cap or image (maybe pointer for less ram usage?)
#path: path to file
#filename: filename
#path_to_output: path to output
#filename_output: filename output
#fps: FPS (if video, else 1)
#length: video length (in seconds, 0 for image)
#frame_amount: total frames (1 for image)
#current_frame_original (None)
#current_frame_swapped
#frame_position: what frame is video on
#TODO settings for each video has to be separate, for now everything is at the same settings, sorry
videos = []
 #id of current video, so the player knows what video to show
videos.append({"video_type": 0,
               "cap": cv2.imread("C:/data_256/seed0000.png"),
               "path": "C:/data_256/",
               "filename": "seed0000.png",
               "path_to_output": "C:/output/",
               "filename_output": "seed0000_processed.png",
               "fps": 1,
               "length": 0,
               "frame_amount": 1,
               "current_frame_original":cv2.imread("C:/data_256/seed0000.png"),
               "current_frame_swapped": cv2.imread("C:/data_256/seed0000.png"),
               "frame_position":0,
               "currently_processing": 0})
for i in range(100):
    videos.append({"video_type": 0,
                "cap": cv2.imread("C:/data_256/seed0001.png"),
                "path": "C:/data_256/",
                "filename": "seed0001.png",
                "path_to_output": "C:/output/",
                "filename_output": "seed0001_processed.png",
                "fps": 1,
                "length": 0,
                "frame_amount": 1,
                "current_frame_original":cv2.imread("C:/data_256/seed0001.png"),
                "current_frame_swapped": cv2.imread("C:/data_256/seed0001.png"),
                "frame_position":0,
                "currently_processing": 0})


prepare_models()
mime = magic.Magic(mime=True)
#TODO ability to delete from videos
# sorry, I don't know what Im doing, TODO make that thing smaller
#! /usr/bin/env python3
#  -*- coding: utf-8 -*-
#
# GUI module generated by PAGE version 7.6
#  in conjunction with Tcl version 8.6
#    Jul 21, 2023 06:43:56 PM AST  platform: Windows NT

import sys
import tkinter as tk
import tkinter.ttk as ttk
from tkinter.constants import *
import os.path
from PIL import Image, ImageTk
from tkinter.filedialog import asksaveasfilename, askdirectory, askopenfilename
_script = sys.argv[0]
_location = os.path.dirname(_script)


_bgcolor = '#d9d9d9'  # X11 color: 'gray85'
_fgcolor = '#000000'  # X11 color: 'black'
_compcolor = 'gray40' # X11 color: #666666
_ana1color = '#c3c3c3' # Closest X11 color: 'gray76'
_ana2color = 'beige' # X11 color: #f5f5dc
_tabfg1 = 'black' 
_tabfg2 = 'black' 
_tabbg1 = 'grey75' 
_tabbg2 = 'grey89' 
_bgmode = 'light' 

_style_code_ran = 0
def _style_code():
    global _style_code_ran
    if _style_code_ran:
       return
    style = ttk.Style()
    if sys.platform == "win32":
       style.theme_use('winnative')
    style.configure('.',background=_bgcolor)
    style.configure('.',foreground=_fgcolor)
    style.configure('.',font='TkDefaultFont')
    style.map('.',background =
       [('selected', _compcolor), ('active',_ana2color)])
    if _bgmode == 'dark':
       style.map('.',foreground =
         [('selected', 'white'), ('active','white')])
    else:
       style.map('.',foreground =
         [('selected', 'black'), ('active','black')])
    style.configure('Vertical.TScrollbar',  background=_bgcolor,
        arrowcolor= _fgcolor)
    style.configure('Horizontal.TScrollbar',  background=_bgcolor,
        arrowcolor= _fgcolor)
    _style_code_ran = 1


# The following code is added to facilitate the Scrolled widgets you specified.
class AutoScroll(object):
    '''Configure the scrollbars for a widget.'''
    def __init__(self, master):
        #  Rozen. Added the try-except clauses so that this class
        #  could be used for scrolled entry widget for which vertical
        #  scrolling is not supported. 5/7/14.
        try:
            vsb = ttk.Scrollbar(master, orient='vertical', command=self.yview)
        except:
            pass
        hsb = ttk.Scrollbar(master, orient='horizontal', command=self.xview)
        try:
            self.configure(yscrollcommand=self._autoscroll(vsb))
        except:
            pass
        self.configure(xscrollcommand=self._autoscroll(hsb))
        self.grid(column=0, row=0, sticky='nsew')
        try:
            vsb.grid(column=1, row=0, sticky='ns')
        except:
            pass
        hsb.grid(column=0, row=1, sticky='ew')
        master.grid_columnconfigure(0, weight=1)
        master.grid_rowconfigure(0, weight=1)
        # Copy geometry methods of master  (taken from ScrolledText.py)
        methods = tk.Pack.__dict__.keys() | tk.Grid.__dict__.keys() \
                  | tk.Place.__dict__.keys()
        for meth in methods:
            if meth[0] != '_' and meth not in ('config', 'configure'):
                setattr(self, meth, getattr(master, meth))

    @staticmethod
    def _autoscroll(sbar):
        '''Hide and show scrollbar as needed.'''
        def wrapped(first, last):
            first, last = float(first), float(last)
            if first <= 0 and last >= 1:
                sbar.grid_remove()
            else:
                sbar.grid()
            sbar.set(first, last)
        return wrapped

    def __str__(self):
        return str(self.master)

def _create_container(func):
    '''Creates a ttk Frame with a given master, and use this new frame to
    place the scrollbars and the widget.'''
    def wrapped(cls, master, **kw):
        container = ttk.Frame(master)
        container.bind('<Enter>', lambda e: _bound_to_mousewheel(e, container))
        container.bind('<Leave>', lambda e: _unbound_to_mousewheel(e, container))
        return func(cls, container, **kw)
    return wrapped

class ScrolledListBox(AutoScroll, tk.Listbox):
    '''A standard Tkinter Listbox widget with scrollbars that will
    automatically show/hide as needed.'''
    @_create_container
    def __init__(self, master, **kw):
        tk.Listbox.__init__(self, master, **kw)
        AutoScroll.__init__(self, master)
    def size_(self):
        sz = tk.Listbox.size(self)
        return sz

import platform
def _bound_to_mousewheel(event, widget):
    child = widget.winfo_children()[0]
    if platform.system() == 'Windows' or platform.system() == 'Darwin':
        child.bind_all('<MouseWheel>', lambda e: _on_mousewheel(e, child))
        child.bind_all('<Shift-MouseWheel>', lambda e: _on_shiftmouse(e, child))
    else:
        child.bind_all('<Button-4>', lambda e: _on_mousewheel(e, child))
        child.bind_all('<Button-5>', lambda e: _on_mousewheel(e, child))
        child.bind_all('<Shift-Button-4>', lambda e: _on_shiftmouse(e, child))
        child.bind_all('<Shift-Button-5>', lambda e: _on_shiftmouse(e, child))

def _unbound_to_mousewheel(event, widget):
    if platform.system() == 'Windows' or platform.system() == 'Darwin':
        widget.unbind_all('<MouseWheel>')
        widget.unbind_all('<Shift-MouseWheel>')
    else:
        widget.unbind_all('<Button-4>')
        widget.unbind_all('<Button-5>')
        widget.unbind_all('<Shift-Button-4>')
        widget.unbind_all('<Shift-Button-5>')

def _on_mousewheel(event, widget):
    if platform.system() == 'Windows':
        widget.yview_scroll(-1*int(event.delta/120),'units')
    elif platform.system() == 'Darwin':
        widget.yview_scroll(-1*int(event.delta),'units')
    else:
        if event.num == 4:
            widget.yview_scroll(-1, 'units')
        elif event.num == 5:
            widget.yview_scroll(1, 'units')

def _on_shiftmouse(event, widget):
    if platform.system() == 'Windows':
        widget.xview_scroll(-1*int(event.delta/120), 'units')
    elif platform.system() == 'Darwin':
        widget.xview_scroll(-1*int(event.delta), 'units')
    else:
        if event.num == 4:
            widget.xview_scroll(-1, 'units')
        elif event.num == 5:
            widget.xview_scroll(1, 'units')
faces_to_swap_window = None
add_target_videos_window = None
additional_settings_window = None
open_codeformer_settings_window = None
def choose_faces_to_swap():
    global faces_to_swap_window, origin_face, globalsx
    def on_slider_change(val):
        this_cap.set(cv2.CAP_PROP_POS_FRAMES, int(val))
        ret, globalsx.this_frame = this_cap.read()
        xframe = globalsx.this_frame.copy()
        bboxes = get_face_boxes(globalsx.this_frame)
        for i in bboxes:
            cv2.rectangle(xframe, (i[0], i[2]), (i[1], i[3]), (0, 255, 0), 2 )
        cv2.imshow("Choose the face", xframe)
    def get_mouse_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Left mouse button clicked at coordinates (x={x}, y={y})")
            bboxes = get_face_boxes(globalsx.this_frame)
            for it, i in enumerate(bboxes):
                if x > i[0] and x < i[1] and y > i[2] and y < i[3]:
                    print('x')
                    globalsx.to_swap.append([i[4], origin_face])
                    cv2.destroyWindow("Choose the face")
                    #print(f"got face {it}")



    def get_face_boxes(image):
        faces = sorted(face_analyzer.get(image), key=lambda x: x.bbox[0])
        bboxes = []
        for i in faces:
            embedding = i.normed_embeddings
            i = i.bbox
            bboxes.append([int(i[0]),int(i[1]),int(i[2]),int(i[3]), embedding])
        return bboxes
    #show_error("sorry, but it's in development, I hope next version will implement it. For now use old method. Sorry")
    #return
    height, width, channels = videos[globalsx.current_video]['current_frame_original'].shape
    if height < 640 or width < 640:
        lowest = min(height, width)
        face_analyzer = prepare_models(custom_return=True, det_size=(128, 128))
    else:
        face_analyzer = prepare_models(custom_return=True, det_size=(640, 640))
    if not faces_to_swap_window or not faces_to_swap_window.winfo_exists():
        filex = askopenfilename(title="Select a face that you want to swap with")
        if not filex:
            return
        try:
            video_type = mime.from_file(filex)
        except Exception as e:
            print(f"{filex} is not image or video, error from video_type: {e}")
            return
        if not video_type.startswith('image'):
            return 
        origin_face = cv2.imread(filex)
        cv2.namedWindow("Choose the face")
        cv2.setMouseCallback("Choose the face", get_mouse_click)
        if videos[globalsx.current_video]['video_type'] == 1: #if we have video, we need to create a slider, so we need to create a window
            this_cap = open_cap(os.path.join(videos[globalsx.current_video]['path'],videos[globalsx.current_video]['filename']))
            faces_to_swap_window = tk.Toplevel(top)
            faces_to_swap_window.title("choose faces to swap")
            faces_to_swap_window.configure(borderwidth="10")
            faces_to_swap_window.configure(background=background_color)
            Labelframe1 = tk.LabelFrame(faces_to_swap_window, text='Choose face enhancer')
            Labelframe1.place(relx=0.085, rely=0.034, relheight=0.909, relwidth=0.905)
            Labelframe1.configure(foreground=text_color, background=box_background_color, relief='groove')
            slider = tk.Scale(Labelframe1, from_=0, to=videos[globalsx.current_video]["frame_amount"]-1, command=on_slider_change, resolution=1)
            slider.pack()
        else:
            globalsx.this_frame = videos[globalsx.current_video]['current_frame_original']
            cv2.imshow("Choose the face", globalsx.this_frame)
            cv2.waitKey(0)
    else:
        faces_to_swap_window.lift()
        faces_to_swap_window.focus()

def current_swapping_thread():
    while True:
        time.sleep(0.01)
        try:
            if len(globalsx.to_swap) > 0:
                current_frame = videos[globalsx.current_video]['frame_position']
                current_frame += globalsx.frame_move
                if current_frame >= videos[globalsx.current_video]['frame_amount']:
                    current_frame = videos[globalsx.current_video]['frame_amount'] -1
                if current_frame < 0:
                    current_frame = 0
                videos[globalsx.current_video]['frame_position'] = current_frame
                videos[globalsx.current_video]['current_frame_swapped'] = swap_frame(videos[globalsx.current_video]['current_frame_original'])
        except Exception as e:
            print(e)
threading.Thread(target=current_swapping_thread).start()
def select_target(batch=False):
    global args
    if not batch:
        filex = askopenfilename(title="Select a target")
    else:
        filex = askdirectory(title="Select output folder")
    if not filex:
        return
    if not batch:
        filex2 = asksaveasfilename(initialfile="test.mp4",title="Select output folder")
    else:
        filex2 = askdirectory(title="Select output folder")
    if not filex2:
        return
    files = []
    if batch:
        for i in os.listdir(filex):
            files.append(i)
    else:
        files.append(filex)
    for file_path in files:
        file_path = os.path.join(filex, file_path)
        try:
            video_type = mime.from_file(file_path)
        except Exception as e:
            print(f"{file_path} is not image or video, error from video_type: {e}")
            continue
        folder, filename = os.path.split(file_path)
        if not batch:
            folder_output, filename_output = os.path.split(filex2)
        else:
            folder_output = filex2
            filename_output = f"{filename}_processed.mp4"

        if video_type.startswith('image'):
            video_type = 0
            cap = cv2.imread(file_path)
            current_frame = cap
            fps = 1
            length = 0
            frame_amount = 1
        elif video_type.startswith("video"):
            video_type = 1
            cap = cv2.VideoCapture(file_path)
            ret, current_frame = cap.read()
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_amount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            length = frame_amount / fps
        else:
            continue
        videos.append({"video_type": video_type,
                    "cap": cap,
                    "path": folder,
                    "filename": filename,
                    "path_to_output": folder_output,
                    "filename_output": filename_output,
                    "fps": fps,
                    "length": length, #in seconds
                    "frame_amount": frame_amount,
                    "current_frame_original": current_frame,
                    "current_frame_swapped": current_frame,
                    "frame_position":0,
                    "currently_processing": 0})
    
def add_target_videos():
    global add_target_videos_window
    if not add_target_videos_window or not add_target_videos_window.winfo_exists():
        add_target_videos_window = tk.Toplevel(top)
        add_target_videos_window.title("add target videos")
        add_target_videos_window.configure(borderwidth="10")
        add_target_videos_window.configure(background=background_color)
        check = tk.IntVar()
        Labelframe1 = tk.LabelFrame(add_target_videos_window, text='Choose videos to add')
        Labelframe1.place(relx=0.085, rely=0.034, relheight=0.909, relwidth=0.905)
        Labelframe1.configure(foreground=text_color, background=box_background_color, relief='groove')
        addvideobutton = tk.Button(Labelframe1, text='add video', command= lambda:select_target(check.get()))
        addvideobutton.place(relx=0.1, rely=0.1, relheight=0.2, relwidth=0.4, bordermode='ignore')
        addvideobutton.configure(activebackground=button_color, activeforeground=text_color)
        addvideobutton.configure(background=button_color, compound='left')
        addvideobutton.configure(disabledforeground=text_color, foreground=text_color)
        addvideobutton.configure(highlightbackground="#d9d9d9", highlightcolor="black")
        addvideobutton.configure(pady="0")
        Checkbutton1 = tk.Checkbutton(Labelframe1, text='Add folder', variable=check)
        Checkbutton1.place(relx=0.085, rely=0.3, relheight=0.1, relwidth=0.9)
        Checkbutton1.configure(activebackground=background_color, activeforeground=button_color,
                            background=background_color, foreground=button_color)
    else:
        add_target_videos_window.lift()
        add_target_videos_window.focus()

def additional_settings():
    global additional_settings_window
    if not additional_settings_window or not additional_settings_window.winfo_exists():
        additional_settings_window = tk.Toplevel(top)
        additional_settings_window.title("additional settings")
        additional_settings_window.configure(borderwidth="50")
        additional_settings_window.configure(background="#090020")
    else:
        additional_settings_window.lift()
        additional_settings_window.focus()

def open_codeformer_settings():
    global open_codeformer_settings_window
    if not open_codeformer_settings_window or not open_codeformer_settings_window.winfo_exists():
        open_codeformer_settings_window = tk.Toplevel(top)
        open_codeformer_settings_window.title('open codeformer settings')
        open_codeformer_settings_window.configure(borderwidth="50")
        open_codeformer_settings_window.configure(background="#090020")
    else:
        open_codeformer_settings_window.lift()
        open_codeformer_settings_window.focus()

def queue_processor():
    def process(frame):
        frame = swap_frame(frame)
        return frame
    while True:
        #wait if queue is empty
        while len(globalsx.render_queue) == 0:
            time.sleep(0.1)
        print("aa")
        current_video_id = globalsx.render_queue.pop(0)
        if videos[current_video_id]['video_type'] == 0:
            show_error("sorry, images is not yet supported in this button, please use \"save frame button on player\"")
            continue
        videos[current_video_id]['cap'].set(cv2.CAP_PROP_POS_FRAMES, 0)
        temp = []
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(videos[current_video_id]['cap'].get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(videos[current_video_id]['cap'].get(cv2.CAP_PROP_FRAME_HEIGHT))
        name = os.path.join(videos[current_video_id]['path_to_output'],videos[current_video_id]['filename_output']+"_temp.mp4")
        out = cv2.VideoWriter(name, fourcc, videos[current_video_id]['fps'], (width, height))
        with tqdm(total=videos[current_video_id]['frame_amount']) as pbar:
            start = time.time()
            while True:
                ret, frame = videos[current_video_id]['cap'].read()
                if ret:
                    t = ThreadWithReturnValue(target=process, args=(frame,))
                    t.start()
                    temp.append(t)
                if len(temp) >= 4 or not ret:
                    image = temp.pop(0).join()
                if len(temp) < 4 and ret:
                    continue
                out.write(image)
                pbar.update(1)
                if len(temp) == 0 and not ret:
                    break
            out.release()
            print(f"time taken for processing of the video: {time.time() - start} seconds")
        
threading.Thread(target=queue_processor).start()   

#we just add to queue
def export_video():
    videos[globalsx.current_video]['currently_processing'] = 1
    globalsx.render_queue.append(globalsx.current_video)
def export_batch():
    pass
def check_queue():
    pass

def resize_image(image, max_width, max_height):
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    original_width, original_height = image.size

    # Calculate the aspect ratio of the original image
    aspect_ratio = original_width / original_height

    # Calculate the desired aspect ratio for the maximum dimensions
    max_aspect_ratio = max_width / max_height

    # Compare the aspect ratios to determine the new dimensions while maintaining aspect ratio
    if aspect_ratio > max_aspect_ratio:
        new_width = max_width
        new_height = int(max_width / aspect_ratio)
    else:
        new_height = max_height
        new_width = int(max_height * aspect_ratio)

    # Resize the image
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return image

def video_length_converter(seconds):
    if seconds == 0:
        return "it's an image"
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    hours = minutes // 60
    remaining_minutes = minutes % 60

    return f"{hours}:{remaining_minutes}:{remaining_seconds}"

def update_current_frame():
    while True:
        if videos[globalsx.current_video]['video_type'] == 1:
            if videos[globalsx.current_video]['currently_processing'] == 0:
                videos[globalsx.current_video]['current_frame_original'] = get_nth_frame(videos[globalsx.current_video]['cap'], videos[globalsx.current_video]['frame_position'])
        time.sleep(0.01)
threading.Thread(target=update_current_frame).start()
def update_gui(old_video_len=0, old_sel=0):
    sel = Scrolledlistbox1.curselection()
    if len(sel) != 0:
        sel = sel[0]
    else:
        sel = 0
    globalsx.current_video = sel
    if len(videos) > 0:
        width = Originalvideopreview.winfo_width() - 20
        height = Originalvideopreview.winfo_height() - 10
        image = ImageTk.PhotoImage(resize_image(videos[globalsx.current_video]['current_frame_original'], max_width=width, max_height=height))
        Originalvideopreview_image.configure(image=image)
        Originalvideopreview_image.image = image
        width = Processedvideopreview.winfo_width() - 20
        height = Processedvideopreview.winfo_height() - 10
        image = ImageTk.PhotoImage(resize_image(videos[globalsx.current_video]['current_frame_swapped'], max_width=width, max_height=height))
        Processedvideopreview_image.configure(image=image)
        Processedvideopreview_image.image = image
        top.update()
        top.update_idletasks()
    if old_video_len != len(videos):
        Scrolledlistbox1.delete('0','end')
        for i in videos:
            Scrolledlistbox1.insert("end", i['filename'])
    if old_sel != sel:
        Label4.configure(text=f'Input filename: {videos[globalsx.current_video]["filename"]}')
        Label5.configure(text=f'Output filename: {videos[globalsx.current_video]["filename_output"]}')
        Label6.configure(text=f'FPS: {videos[globalsx.current_video]["fps"]}')
        Label7.configure(text=f'Video length: {video_length_converter(videos[globalsx.current_video]["length"])}')
        Label8.configure(text=f'Total frames: {videos[globalsx.current_video]["frame_amount"]}')
    globalsx.enhancer_choice = enhancer_choice.get()
    globalsx.alpha = float(Scale2.get())
    top.after(50, update_gui, len(videos), sel)
background_color = "#19122b"
box_background_color = "#251b3f"
button_color = "#a304cf"
text_color = "#f1f2f0"
top= tk.Tk()

top.geometry("1244x826+681+73")
top.minsize(1244, 826)
#top.maxsize(1924, 1061)
top.resizable(1, 1)
top.title("Toplevel 0")
top.configure(borderwidth="50")
top.configure(background=background_color)

che56 = tk.IntVar()

Checkbutton1 = tk.Checkbutton(top, text='Swap all faces with one face', variable=che56)
Checkbutton1.place(relx=0.009, rely=0.95, relheight=0.031, relwidth=0.175)
Checkbutton1.configure(activebackground=background_color, activeforeground=button_color,
                       background=background_color, foreground=button_color)

Button2 = tk.Button(top, text='Help')
Button2.place(relx=0.912, rely=0.961, height=24, width=77)
Button2.configure(activebackground=button_color, activeforeground=text_color, background=button_color,
                  foreground=text_color)

Button3 = tk.Button(top, text='Info')
Button3.place(relx=0.822, rely=0.961, height=24, width=77)
Button3.configure(activebackground=button_color, activeforeground=text_color, background=button_color,
                  foreground=text_color)

Labelframe1 = tk.LabelFrame(top, text='Choose face enhancer')
Labelframe1.place(relx=0.185, rely=0.834, relheight=0.099, relwidth=0.175)
Labelframe1.configure(foreground=text_color, background=box_background_color, relief='groove')
enhancer_choice = tk.StringVar(value='none')
choices = ['fastface enhancer', 'gfpgan', 'codeformer', 'gfpgan onnx']

#if not globalsx.args['lowmem']:
#    choices.remove('fastface enhancer')

dropdown = tk.OptionMenu(Labelframe1, enhancer_choice, enhancer_choice.get(), *choices)
dropdown.configure(background=button_color, foreground=text_color,activeforeground=text_color, activebackground=button_color)
dropdown.pack()
Labelframe2 = tk.LabelFrame(top, text='Choose face swapper model', background=box_background_color)
Labelframe2.place(relx=0.008, rely=0.834, relheight=0.099, relwidth=0.175)
Labelframe2.configure(foreground=text_color, relief='groove')

Labelframe3 = tk.LabelFrame(top, text='Frame control')
Labelframe3.place(relx=0.008, rely=0.462, relheight=0.167, relwidth=0.621)
Labelframe3.configure(foreground=text_color, background=box_background_color, relief='groove')

Scale1 = tk.Scale(Labelframe3, from_=0.0, to=100.0, resolution=1.0, orient="horizontal")
Scale1.place(relx=0.0, rely=0.638, relheight=0.304, relwidth=0.994)
Scale1.configure(activebackground=box_background_color, background=box_background_color, foreground=text_color)
Scale1.configure(highlightbackground=box_background_color, highlightcolor="black")
Scale1.configure(length="666", troughcolor="#d9d9d9")
def change_frame_index(value):
    if videos[globalsx.current_video]['video_type'] == 1:
        t = videos[globalsx.current_video]['frame_position'] + value
        if t > -1 and t < videos[globalsx.current_video]['frame_amount']:
            videos[globalsx.current_video]['frame_position'] += value

def change_frame_move(value):
    if videos[globalsx.current_video]['video_type'] == 1:
        globalsx.frame_move = value
    else:
        globalsx.frame_move = 0

framesavebutton = tk.Button(Labelframe3, text='💾', command= lambda:change_frame_index(-1))
framesavebutton.place(relx=0.237, rely=0.159, height=34, width=47, bordermode='ignore')
framesavebutton.configure(activebackground=button_color, activeforeground=text_color)
framesavebutton.configure(background=button_color, compound='left')
framesavebutton.configure(disabledforeground=text_color, foreground=text_color)
framesavebutton.configure(highlightbackground="#d9d9d9", highlightcolor="black")
framesavebutton.configure(pady="0")

framebackbutton = tk.Button(Labelframe3, text='<', command= lambda:change_frame_index(-1))
framebackbutton.place(relx=0.313, rely=0.159, height=34, width=47, bordermode='ignore')
framebackbutton.configure(activebackground=button_color, activeforeground=text_color)
framebackbutton.configure(background=button_color, compound='left')
framebackbutton.configure(disabledforeground=text_color, foreground=text_color)
framebackbutton.configure(highlightbackground="#d9d9d9", highlightcolor="black")
framebackbutton.configure(pady="0")

frameforwardbutton = tk.Button(Labelframe3, text='>', command=lambda:change_frame_index(1))
frameforwardbutton.place(relx=0.611, rely=0.159, height=34, width=47, bordermode='ignore')
frameforwardbutton.configure(activebackground=button_color, activeforeground=text_color)
frameforwardbutton.configure(background=button_color, compound='left')
frameforwardbutton.configure(disabledforeground=text_color, foreground=text_color)
frameforwardbutton.configure(highlightbackground="#d9d9d9", highlightcolor="black")
frameforwardbutton.configure(pady="0")

backplaybutton = tk.Button(Labelframe3, text='◀', command=lambda:change_frame_move(-1))
backplaybutton.place(relx=0.389, rely=0.159, height=34, width=47, bordermode='ignore')
backplaybutton.configure(activebackground=button_color, activeforeground=text_color)
backplaybutton.configure(background=button_color, compound='left')
backplaybutton.configure(disabledforeground="#a3a3a3", foreground=text_color)
backplaybutton.configure(highlightbackground="#d9d9d9", highlightcolor="black")
backplaybutton.configure(pady="0")

pausebutton = tk.Button(Labelframe3, text='⏸️', command=lambda:change_frame_move(0))
pausebutton.place(relx=0.462, rely=0.159, height=34, width=47, bordermode='ignore')
pausebutton.configure(activebackground=button_color, activeforeground="black")
pausebutton.configure(background=button_color, compound='left')
pausebutton.configure(disabledforeground="#a3a3a3", foreground=text_color)
pausebutton.configure(highlightbackground="#d9d9d9", highlightcolor="black")
pausebutton.configure(pady="0")

Forwardplaybutton = tk.Button(Labelframe3, text='▶', command=lambda:change_frame_move(1))
Forwardplaybutton.place(relx=0.538, rely=0.159, height=34, width=47, bordermode='ignore')
Forwardplaybutton.configure(activebackground="beige", activeforeground="black")
Forwardplaybutton.configure(background=button_color, compound='left')
Forwardplaybutton.configure(disabledforeground="#a3a3a3", foreground=text_color)
Forwardplaybutton.configure(highlightbackground="#d9d9d9", highlightcolor="black")
Forwardplaybutton.configure(justify='right')
Forwardplaybutton.configure(pady="0")
framerefreshbutton = tk.Button(Labelframe3, text='↻')
framerefreshbutton.place(relx=0.684, rely=0.159, height=34, width=47, bordermode='ignore')
framerefreshbutton.configure(activebackground="beige", activeforeground="black")
framerefreshbutton.configure(background=button_color, compound='left')
framerefreshbutton.configure(disabledforeground="#a3a3a3", foreground=text_color)
framerefreshbutton.configure(highlightbackground="#d9d9d9", highlightcolor="black")
framerefreshbutton.configure(pady="0")

Label2 = tk.Label(Labelframe3, text='frame: 0/0')
Label2.place(relx=0.446, rely=0.478, height=34, width=85, bordermode='ignore')
Label2.configure(background=box_background_color, compound='left')
Label2.configure(disabledforeground="#a3a3a3", foreground=text_color)

Originalvideopreview = tk.LabelFrame(top, text='Original video preview')
Originalvideopreview.place(relx=0.009, rely=0.013, relheight=0.442, relwidth=0.474)
Originalvideopreview.configure(background=box_background_color, relief='groove', foreground=text_color)

Originalvideopreview_image = tk.Label(Originalvideopreview)
Originalvideopreview_image.pack()
'''image = ImageTk.PhotoImage(resize_image("C:/data_256/seed0000.png"))
Originalvideopreview_image.configure(image=image)'''
Processedvideopreview = tk.LabelFrame(top, text='Processed video preview')
Processedvideopreview.place(relx=0.503, rely=0.013, relheight=0.443, relwidth=0.472)
Processedvideopreview.configure(background=box_background_color, relief='groove', foreground=text_color)
Processedvideopreview_image = tk.Label(Processedvideopreview)
Processedvideopreview_image.pack()

Labelframe4 = tk.LabelFrame(top, text='Blend upscale and swapped face')
Labelframe4.place(relx=0.289, rely=0.642, relheight=0.126, relwidth=0.219)
Labelframe4.configure(background=box_background_color, relief='groove', foreground=text_color)

Scale2 = tk.Scale(Labelframe4, from_=0.0, to=1.0, resolution=0.1)
Scale2.place(relx=0.026, rely=0.066, relheight=0.532, relwidth=0.941)
Scale2.configure(activebackground=box_background_color, background=box_background_color, foreground=text_color)
Scale2.configure(highlightbackground="#d9d9d9", highlightcolor="black")
Scale2.configure(length="376", orient="horizontal", troughcolor="#d9d9d9")

# Assuming _style_code() is a function that configures the style
#_style_code()

Scrolledlistbox1 = ScrolledListBox(top)
Scrolledlistbox1.place(relx=0.816, rely=0.462, relheight=0.47, relwidth=0.157)
Scrolledlistbox1.configure(background=box_background_color, font="TkFixedFont")
Scrolledlistbox1.configure(foreground=text_color, highlightbackground=button_color, highlightcolor=button_color)
Scrolledlistbox1.configure(selectbackground=button_color, selectforeground=text_color)

Labelframe5 = tk.LabelFrame(top, text='Video information')
Labelframe5.place(relx=0.009, rely=0.642, relheight=0.172, relwidth=0.271)
Labelframe5.configure(background=box_background_color, relief='groove', foreground=text_color)

Label4 = tk.Label(Labelframe5, text='Input filename:')
Label4.pack(anchor='w')
Label4.configure(background=box_background_color, compound='left', disabledforeground="#a3a3a3")
Label4.configure(foreground=text_color)

Label5 = tk.Label(Labelframe5, text='Output filename:')
Label5.pack(anchor='w')
Label5.configure(background=box_background_color, compound='left', disabledforeground="#a3a3a3")
Label5.configure(foreground=text_color)

Label6 = tk.Label(Labelframe5, text='FPS:')
Label6.pack(anchor='w')
Label6.configure(background=box_background_color, compound='left')
Label6.configure(disabledforeground="#a3a3a3", foreground=text_color)

Label7 = tk.Label(Labelframe5, text='Video length:')
Label7.pack(anchor='w')
Label7.configure(background=box_background_color, compound='left')
Label7.configure(disabledforeground="#a3a3a3", foreground=text_color)

Label8 = tk.Label(Labelframe5, text='Total frames:')
Label8.pack(anchor='w')
Label8.configure(background=box_background_color, compound='left', disabledforeground="#a3a3a3")
Label8.configure(foreground=text_color)

Labelframe6 = tk.LabelFrame(top, text='Processing threads')
Labelframe6.place(relx=0.519, rely=0.642, relheight=0.062, relwidth=0.101)
Labelframe6.configure(background=box_background_color, relief='groove', foreground=text_color,)
def adjust_threads():
    globalsx.args['threads'] = int(Spinbox2.get())
Spinbox2 = tk.Spinbox(Labelframe6, from_=1, to=24, command=adjust_threads)
Spinbox2.pack(fill='both', expand=True)
Spinbox2.configure(activebackground=box_background_color, background=box_background_color, buttonbackground=button_color)
Spinbox2.configure(disabledforeground=box_background_color, font="TkDefaultFont", foreground=text_color)
Spinbox2.configure(highlightbackground=box_background_color, highlightcolor=text_color, insertbackground=text_color)
Spinbox2.configure(selectbackground=box_background_color, selectforeground=text_color)

Labelframe7 = tk.LabelFrame(top, text='Progress')
Labelframe7.place(relx=0.369, rely=0.834, relheight=0.096, relwidth=0.432) 
Labelframe7.configure(background=box_background_color, relief='groove', foreground=text_color)

Label1 = tk.Label(Labelframe7, text='''Progress bar here''')
Label1.pack(fill='both', expand=True)
Label1.configure(anchor='w')
Label1.configure(background=box_background_color)
Label1.configure(compound='left')
Label1.configure(disabledforeground="#a3a3a3")
Label1.configure(foreground=text_color)

Frame1 = tk.Frame(top, relief='groove', borderwidth="2")
Frame1.place(relx=0.634, rely=0.462, relheight=0.165, relwidth=0.177)
Frame1.configure(background=box_background_color, relief='groove')

Button1 = tk.Button(Frame1, text='Choose faces to swap', command=choose_faces_to_swap)
Button1.pack(side='top', fill='both', expand=True)
Button1.configure(background=button_color, compound='left')
Button1.configure(disabledforeground="#a3a3a3", foreground=text_color)

Button1 = tk.Button(Frame1, text='Add target videos', command=add_target_videos)
Button1.pack(side='top', fill='both', expand=True)
Button1.configure(background=button_color, compound='left')
Button1.configure(disabledforeground="#a3a3a3", foreground=text_color)

Button7 = tk.Button(Frame1, text='Additional settings', command=additional_settings)
Button7.pack(side='top', fill='both', expand=True)
Button7.configure(background=button_color, compound='left')
Button7.configure(disabledforeground="#a3a3a3", foreground=text_color)

Button8 = tk.Button(Frame1, text='Open codeformer settings', command=open_codeformer_settings)
Button8.pack(side='top', fill='both', expand=True)
Button8.configure(background=button_color, compound='left')
Button8.configure(disabledforeground="#a3a3a3", foreground=text_color)
Frameexport = tk.Frame(top, relief='groove', borderwidth="2")
Frameexport.place(relx=0.634, rely=0.642, relheight=0.165/4*3, relwidth=0.177)
Frameexport.configure(background=box_background_color, relief='groove')


exportButton1 = tk.Button(Frameexport, text='Export video/image', command=export_video)
exportButton1.pack(side='top', fill='both', expand=True)
exportButton1.configure(background=button_color, compound='left')
exportButton1.configure(disabledforeground="#a3a3a3", foreground=text_color)

exportButton2 = tk.Button(Frameexport, text='Export batch', command=export_batch)
exportButton2.pack(side='top', fill='both', expand=True)
exportButton2.configure(background=button_color, compound='left')
exportButton2.configure(disabledforeground="#a3a3a3", foreground=text_color)
exportButton3 = tk.Button(Frameexport, text='Check export queue', command=check_queue)
exportButton3.pack(side='top', fill='both', expand=True)
exportButton3.configure(background=button_color, compound='left')
exportButton3.configure(disabledforeground="#a3a3a3", foreground=text_color)
top.update_idletasks() #sometimes dies at the start, needs this to stay alive
Scrolledlistbox1.delete('0','end')
for i in videos:
    Scrolledlistbox1.insert("end", i['filename'])
top.update_idletasks() #sometimes dies at the start, needs this to stay alive
top.after(100, update_gui)


top.mainloop()
