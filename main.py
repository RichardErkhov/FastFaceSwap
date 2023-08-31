import time
import argparse
import os
import globalsz
mode = 2  # 1 for side-by-side, 2 for stacked
background_color = "#222831"
button_color = "#0E8388"
text_color = "#EEEEEE"
tick_color = "#222831"
tick_background_color = "#EEEEEE"
border_color = "#444A53"
selector_color = "#0E8388"
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--face', help='use this face', dest='face', default="face.jpg")
parser.add_argument('-t', '--target', help='replace this face. If camera, use integer like 0',default="0", dest='target_path')
parser.add_argument('-o', '--output', help='path to output of the video',default="video.mp4", dest='output')
parser.add_argument('-cam-fix', '--camera-fix', help='fix for logitech cameras that start for 40 seconds in default mode.', dest='camera_fix', action='store_true')
parser.add_argument('-res', '--resolution', help='camera resolution, given in format WxH (ex 1920x1080). Is set for camera mode only',default="1920x1080", dest='resolution')
parser.add_argument('--threads', help='amount of gpu threads',default="4", dest='threads')
parser.add_argument('--image', help='Include if the target is image', dest='image', action='store_true')
parser.add_argument('--cli', help='run in cli mode, turns off preview and now accepts switch of face enhancer from the command', dest='cli', action='store_true')
parser.add_argument('--face-enhancer', help='face enhancer, choice works only in cli mode. In gui mode, you need to choose from gui', dest='face_enhancer', default='none', choices=['none','gfpgan', 'ffe', 'codeformer', 'gpfgan_onnx', 'real_esrgan'])
parser.add_argument('--no-face-swapper', '--no-swapper', help='disables face swapper', dest='no_faceswap', action='store_true')
#parser.add_argument('--preview-mode', help='experimental: preview mode', dest='preview', action='store_true')
parser.add_argument('--experimental', help='experimental mode, enables features like buffered video reader', dest='experimental', action='store_true')
parser.add_argument('--nocuda','--no-cuda', help='no cuda should be used', dest='nocuda', action='store_true')
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
parser.add_argument('--optimization', help='choose the mode of the model: fp32 (default), fp16 (smaller, might be faster), int8 (doesnt work properly on old gpus, I dont know about new once, please test. On old gpus it uses cpu)', dest='optimization', default='fp32', choices=['fp32','fp16', 'int8'])
parser.add_argument('--fast-load', help='try to load as fast as possible, may be delays in the work, shouldnt affect the speed of processing', dest='fastload', action='store_true')
parser.add_argument("--bbox-adjust", help='adjustements to do for the box: x1,y1 coords of left top corner and x2,y2 are bottom right. Give in the form x1xy1xx2xy2 (default: 50x50x50x50)', default='50x50x50x50',dest='bbox_adjust')
parser.add_argument("-vcam", "--virtual-camera", help='allows to use OBS virtual camera as output source', action='store_true', dest="vcam")
parser.add_argument("--apple", help='just in case you are an apple user, you can finally use FFS', action='store_true', dest="apple")
args = {}
for name, value in vars(parser.parse_args()).items():
    args[name] = value
width, height = args['resolution'].split('x')
globalsz.width, globalsz.height = int(width), int(height)
if args['batch'] != "" and not args['batch'].endswith(".mp4"):
    args['batch'] += '.mp4'
#if args['extract_target'] != '':
#    os.makedirs(args['extract_target'])
if args['extract_output'] != '':
    os.makedirs(args['extract_output'])
if args['vcam']:
    try:
        import pyvirtualcam
    except:
        print("pip install pyvirtualcam to support output to OBS virtual camera")
        exit()
alpha = float(args['alpha'])
frame = None #so tkinter doesn't die 
#if args['cli']:
    #testx = input("Are you sure you want to extract frames from videos? It will be done in the background (yes for yes and anything else for no):")
    #if testx == 'yes':
        #if args['batch'] == ''
#just a fix, sometimes speeds up things
os.environ['OMP_NUM_THREADS'] = '1'
globalsz.args = args
#from types import NoneType
NoneType = type(None)
import threading, os, time
if not args['fastload']:
    from plugins.codeformer_app_cv2 import inference_app as codeformer
globalsz.lowmem = args['lowmem']
from utils import *
videos = []
current_video = 0 #id of video
target_embedding = None
clip_neg_prompt = ""
clip_pos_prompt = ""
'''
how video is
types:
0 - image
1 - video
if image:
    original_image: numpy image (None if not loaded)
    swapped_image: numpy image (None if not swapped)
if video:
    cap: cv2 videocapture (some magic thing in future)
    original_image: numpy image (current frame, None if not loaded)
    swapped_image: numpy image (swapped frame corresponded to original_image, None if not swapped)
target_path: str (or os path object) (if camera, it's int)
save_path: str (or os path object)
save_temp_path: str (or os path object)
settings: dict (shown later)
faces_to_swap: list or None (if None, swapping all faces)
current_frame_index: int
total_frames: int (-1 for camera, -1 for image)
rendering: bool,
width: int,
heigh: int,
fps: int (-1 for image)

settings:
{"threads": int,
"enable_swapper":bool,
"enable_enhancer":bool,
"enhancer_choice":str (name of the enhancer)
"bbox_adjust": [x1, y1, x2, y2],
"codeformer_fidelity": float,
"blender": float,
"codeformer_skip_if_no_face": bool,
"codeformer_upscale_face": bool,
"codeformer_enhancer_background": bool,
"codeformer_upscale_amount":int,
}
faces_to_swap: [[target_embedding, face_to_swap_with],]

'''


class simulate:
    def __init__(self, bbox, kps, det_score, embedding, normed_embedding):
        self.bbox = bbox
        self.kps = kps
        self.det_score = det_score
        self.embedding=embedding
        self.normed_embedding = normed_embedding
def kill_ui():
    global root
    root.destroy()
def get_source_face():
    global current_video
    #if isinstance(globalsz.source_face, NoneType):
        #try:
            #globalsz.source_face = sorted(face_analysers[0].get(cv2.imread(args['face'])), key=lambda x: x.bbox[0])[0]
        #except Exception as e:
        #    print(f"HUSTON, WE HAVE A PROBLEM. WE CAN'T DETECT THE FACE IN THE IMAGE YOU PROVIDED! ERROR: {e}")
        #    if not args['cli']:
        #        show_error_custom(text = f"HUSTON, WE HAVE A PROBLEM. WE CAN'T DETECT THE FACE IN THE IMAGE YOU PROVIDED! ERROR: {e}")
        #        kill_ui()
    return videos[current_video]['face'] #globalsz.source_face
def start_swapper(sw):
    import pickle
    with open('ll.pkl', 'rb') as file:
        loaded_data = pickle.load(file)
    frame = face_swappers[sw].get(cv2.imread(args['face']), loaded_data, loaded_data, paste_back=True)
    return frame
def start_analyser(sw):
    x = sorted(face_analysers[sw].get(cv2.imread(args['face'])), key=lambda x: x.bbox[0])[0]
    return x
def startx():
    global face_swappers, face_analysers
    face_swappers, face_analysers = prepare_swappers_and_analysers(args)
    if not args['cli']:
        threads = []
        #threads_2 = []
        for i in range(len(face_swappers)):
            t = threading.Thread(target=start_swapper, args=[i,])
            t.start()
            threads.append(t)
        #for i in range(len(face_swappers)):
            t = threading.Thread(target=start_analyser, args=[i,])
            t.start()
            threads.append(t)
        #for i in threads:
        #    i.join()
        return threads
if args['fastload'] and args['cli']:
    tx = threading.Thread(target=startx)
    tx.start()
elif args['fastload'] and not args['cli']:
    tx = startx()

#    tx, tx2 = startx()
from tqdm import tqdm
from PIL import Image
if not args['cli']:
    from PIL import ImageTk
if not args['lowmem']:
    if not args['fastload']:
        import tensorflow as tf
        prepare()
    
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

if not args['fastload']:
    if not globalsz.args['nocuda'] and not args['apple']:
        device = torch.device(0)
        gpu_memory_total = round(torch.cuda.get_device_properties(device).total_memory / 1024**3,2)  # Convert bytes to GB

def on_closing():
    print("thank you for using FFS")
    print("If you didn't, please join our discord. Thank you")
    os._exit(0)

while True:
    if not args['cli']:
        from thatrandomfilethatneverwillbedeleted import ScrolledListBox, ScrolledListBox_horizontal
        import tkinter as tk
        from tkinter import ttk
        from tkinter.filedialog import asksaveasfilename, askdirectory, askopenfilename
        '''def finish(menu):
            global thread_amount_temp
            thread_amount_temp = thread_amount_input.get()
            menu.destroy()
        menu = tk.Tk()
        menu.geometry("500x500")
        menu.configure(bg=background_color)
        menu.title("FFS")
        
        menu.protocol("WM_DELETE_WINDOW", on_closing)
        button_start_program = tk.Button(menu, text="Start Program",bg=button_color, fg=text_color, command=lambda: finish(menu))
        button_start_program.pack()
        select_face_label = tk.Label(menu,text=f'Face filename: {args["face"]}', fg=text_color, bg=background_color)
        select_face_label.pack()
        button_select_face = tk.Button(menu, text='Select face',bg=button_color, fg=text_color, command=select_face)
        button_select_face.pack()
        select_target_label = tk.Label(menu,text=f'Target filename: {args["target_path"]}', fg=text_color, bg=background_color)
        select_target_label.pack()
        button_select_target = tk.Button(menu, text='Select target',bg=button_color, fg=text_color, command=select_target)
        button_select_target.pack()
        button_select_camera = tk.Button(menu, text='run from camera',bg=button_color, fg=text_color, command=select_camera)
        button_select_camera.pack()
        select_output_label = tk.Label(menu, text=f'output filename: {args["output"]}', fg=text_color, bg=background_color)
        select_output_label.pack()
        button_select_output = tk.Button(menu, text='Select output',bg=button_color, fg=text_color, command=select_output)
        button_select_output.pack()
        thread_amount_label = tk.Label(menu, text='Select the number of threads', fg=text_color, bg=background_color)
        thread_amount_label.pack()
        thread_amount_input = tk.Entry(menu)
        thread_amount_input.pack()
        menu.mainloop()
        if thread_amount_temp != "":
            args['threads'] = int(thread_amount_temp)'''

    if not isinstance(args['target_path'], int):
        if (args['target_path'].isdigit()):
            args['target_path'] = int(args['target_path'])

    adjust_x1, adjust_y1, adjust_x2, adjust_y2 = args['bbox_adjust'].split('x')
    adjust_x1, adjust_y1, adjust_x2, adjust_y2 = int(adjust_x1), int(adjust_y1), int(adjust_x2), int(adjust_y2)
    '''adjust_x1 = 50
    adjust_y1 = 50
    adjust_x2 = 50
    adjust_y2 = 50'''

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


    frame_index = 0
    frame_move = 0
    
    if not args['cli']:
        
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
        def _create_container(func):
            '''Creates a ttk Frame with a given master, and use this new frame to
            place the scrollbars and the widget.'''
            def wrapped(cls, master, **kw):
                container = ttk.Frame(master)
                container.bind('<Enter>', lambda e: _bound_to_mousewheel(e, container))
                container.bind('<Leave>', lambda e: _unbound_to_mousewheel(e, container))
                return func(cls, container, **kw)
            return wrapped
        """class ScrolledListBox(AutoScroll, tk.Listbox):
            '''A standard Tkinter Listbox widget with scrollbars that will
            automatically show/hide as needed.'''
            @_create_container
            def __init__(self, master, **kw):
                tk.Listbox.__init__(self, master, **kw)
                AutoScroll.__init__(self, master)
            def size_(self):
                sz = tk.Listbox.size(self)
                return sz"""

        root = tk.Tk()
        root.state('zoomed')
        style = ttk.Style()
        # Set the theme to "clam"
        style.theme_use("clam")

        # Configure the Checkbutton style
        style.configure("TCheckbutton", 
                        indicatorbackground=tick_background_color, 
                        indicatorforeground=tick_color,
                        background=background_color, 
                        foreground=text_color)

        # Ensure the Checkbutton doesn't change appearance when active
        style.map("TCheckbutton", 
                indicatorbackground=[("active", tick_background_color)], 
                indicatorforeground=[("active", tick_color)],
                background=[("active", background_color)], 
                foreground=[("active", text_color)])
        #if not args['preview']:
        #    root.geometry("1000x750")
        #else:
        row_counter = 0
        root.geometry("1000x970")
        root.configure(bg=background_color)
        root.title("FFS")
        root.protocol("WM_DELETE_WINDOW", on_closing)
        left_frame = tk.Frame(root, bg=background_color)
        left_frame.grid(row=0, column=3, rowspan=2, sticky="ns")
        faceswapper_checkbox_var = tk.IntVar(value=1)
        faceswapper_checkbox = ttk.Checkbutton(left_frame, text="Face swapper", variable=faceswapper_checkbox_var, style="TCheckbutton")
        faceswapper_checkbox.grid(row=row_counter, column=0)
        row_counter += 1
        
        checkbox_var = tk.IntVar()
        checkbox = ttk.Checkbutton(left_frame, text="Face enhancer", variable=checkbox_var, style="TCheckbutton")
        checkbox.grid(row=row_counter, column=0)
        row_counter += 1
        
        enhancer_choice = tk.StringVar(value='fastface enhancer')
        choices = ['fastface enhancer', 'gfpgan', 'codeformer', 'gfpgan onnx', "real esrgan"]

        if args['lowmem']:
            choices.remove('fastface enhancer')

        dropdown = ttk.OptionMenu(left_frame, enhancer_choice, enhancer_choice.get(), *choices)
        dropdown.grid(row=row_counter, column=0)
        row_counter += 1

        show_bbox_var = tk.IntVar()
        show_bbox = ttk.Checkbutton(left_frame, text="draw bounding box around faces", variable=show_bbox_var, style="TCheckbutton")
        show_bbox.grid(row=row_counter, column=0)
        row_counter += 1
        
        realtime_updater_var = tk.IntVar()
        realtime_updater = ttk.Checkbutton(left_frame, text="realtime updater", variable=realtime_updater_var, style="TCheckbutton")
        realtime_updater.grid(row=row_counter, column=0)
        row_counter += 1
        if not isinstance(args['target_path'], int):
            progress_label = tk.Label(left_frame, fg=text_color, bg=background_color)
            progress_label.grid(row=row_counter, column=0)
            row_counter += 1
        
        usage_label1 = tk.Label(left_frame, fg=text_color, bg=background_color)
        usage_label1.grid(row=row_counter, column=0)
        row_counter += 1
        
        if not args['nocuda'] and not args['apple']:
            usage_label2 = tk.Label(left_frame, fg=text_color, bg=background_color)
            usage_label2.grid(row=row_counter, column=0)
            row_counter += 1
        
        #if args['preview']:
        def on_slider_move(value):
            global videos
            videos[current_video]["current_frame_index"] = int(value)
        
        def edit_index(amount):
            global videos
            mini = 0
            maxi = videos[current_video]['frame_number']
            videos[current_video]["current_frame_index"] += amount
            slider.set(videos[current_video]["current_frame_index"])
        
        
        def edit_play(amount):
            global videos, frame_move
            frame_move = amount
        frame_amount = count_frames(args['target_path'])
        label = tk.Label(left_frame, text="frame number", fg=text_color, bg=background_color)
        label.grid(row=row_counter, column=0)
        row_counter += 1
        
        slider = tk.Scale(left_frame, from_=1, to=frame_amount, fg=text_color, bg=background_color, orient=tk.HORIZONTAL, command=on_slider_move)
        slider.grid(row=row_counter, column=0, sticky="ew")
        row_counter += 1
        
        frame_count_label = tk.Label(left_frame, text=f"total frames: {frame_amount}", fg=text_color, bg=background_color)
        frame_count_label.grid(row=row_counter, column=0, sticky="ew")
        row_counter += 1
        
        button_width = left_frame.winfo_width() // 2
        
        #label = tk.Label(left_frame, text = "frame back, frame forward, backplay, pause, play", fg=text_color, bg=background_color)
        #label.grid(row=row_counter, column=0, sticky="ew")
        #row_counter += 1
        
        button_frame = tk.Frame(left_frame, bg=background_color)
        button_frame.grid(row=row_counter, column=0, pady=10, sticky="ew")
        row_counter += 1
        
        frame_back_button = tk.Button(button_frame, text='<', bg=button_color, fg=text_color, width=button_width, command=lambda: edit_index(-1), anchor="center")
        frame_back_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        frame_back_button = tk.Button(button_frame, text='◀', bg=button_color, fg=text_color, width=button_width, command=lambda: edit_play(-1), anchor="center")
        frame_back_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        frame_back_button = tk.Button(button_frame, text='⏸', bg=button_color, fg=text_color, width=button_width, command=lambda: edit_play(0), anchor="center")
        frame_back_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        frame_back_button = tk.Button(button_frame, text='▶', bg=button_color, fg=text_color, width=button_width, command=lambda: edit_play(1), anchor="center")
        frame_back_button.pack(side=tk.LEFT, fill=tk.X, expand=True)

        frame_forward_button = tk.Button(button_frame, text='>', bg=button_color, fg=text_color, width=button_width, command=lambda: edit_index(1), anchor="center")
        frame_forward_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
        def run_it_please():
            global count, videos, current_video
            videos[current_video]['rendering'] = True
            videos[current_video]['current_frame_index'] = 0
            videos[current_video]['count'] = -1
            render_button.config(state=tk.DISABLED)
            stop_rendering_button.config(state=tk.ACTIVE)
            videos[current_video]['cap'].set(cv2.CAP_PROP_POS_FRAMES, 0)
        def not_run_it_please():
            global count, videos, current_video
            videos[current_video]['rendering'] = False
            videos[current_video]['current_frame_index'] = 0
            videos[current_video]['count'] = -1
            render_button.config(state=tk.ACTIVE)
            stop_rendering_button.config(state=tk.DISABLED)
            videos[current_video]['cap'].set(cv2.CAP_PROP_POS_FRAMES, 0)
            videos[current_video]['out'].release()
            videos[current_video]['out'] = cv2.VideoWriter(videos[current_video]['out_settings_for_resetting']['name_temp'], videos[current_video]['out_settings_for_resetting']['fourcc'], 
                                                           videos[current_video]['out_settings_for_resetting']['fps'], (videos[current_video]['out_settings_for_resetting']['width'], videos[current_video]['out_settings_for_resetting']['height']))

        
        original_image_second_open = False
        swapped_image_second_open = False

        def on_closing_original_image_second():
            global original_image_second_window, original_image_second_open
            original_image_second_open = False
            original_image_second_window.destroy()
        def on_closing_swapped_image_second():
            global swapped_image_second_window, swapped_image_second_open
            swapped_image_second_open = False
            swapped_image_second_window.destroy()
        def open_original_image_second():
            global original_image_label_second, original_image_second_window,original_image_second_open
            if not original_image_second_open:
                original_image_second_window = tk.Toplevel(root, bg=background_color)
                original_image_second_window.protocol("WM_DELETE_WINDOW", on_closing_original_image_second)
                original_image_second_window.geometry("640x360")
                original_image_second_window.title("Original image")
                original_image_label_second = tk.Label(original_image_second_window,bg=background_color)
                original_image_label_second.pack(fill=tk.BOTH, expand=True)
                original_image_second_open = True
            else:
                original_image_second_window.lift()
        def open_swapped_image_second():
            global swapped_image_label_second, swapped_image_second_window ,swapped_image_second_open
            if not swapped_image_second_open:
                swapped_image_second_window = tk.Toplevel(root, bg=background_color)
                swapped_image_second_window.protocol("WM_DELETE_WINDOW", on_closing_swapped_image_second)
                swapped_image_second_window.geometry("640x360")
                swapped_image_second_window.title("Swapped image")
                swapped_image_label_second = tk.Label(swapped_image_second_window,bg=background_color)
                swapped_image_label_second.pack(fill=tk.BOTH, expand=True)
                swapped_image_second_open = True
            else:
                swapped_image_second_window.lift()
        open_frames_frame = tk.Frame(left_frame, bg=background_color)
        open_frames_frame.grid(row=row_counter, column=0, pady=10, sticky="ew")
        row_counter += 1
        open_original_image_second_button = tk.Button(open_frames_frame, text='open original preview', bg=button_color, fg=text_color, command=open_original_image_second)
        open_original_image_second_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        open_swapped_image_second_button = tk.Button(open_frames_frame, text='open swapped preview', bg=button_color, fg=text_color, command=open_swapped_image_second)
        open_swapped_image_second_button.pack(side=tk.LEFT, fill=tk.X, expand=True)

        '''show_external_swapped_preview_var = tk.IntVar()
        show_external_swapped_preview = ttk.Checkbutton(left_frame, text="Show swapped frame in another window", variable=show_external_swapped_preview_var, style="TCheckbutton")
        show_external_swapped_preview.grid(row=row_counter, column=0)
        show_external_swapped_preview_var.set(0)
        row_counter += 1'''
        def unselect_face():
            global target_embedding, old_index, args, videos
            args['selective'] = ''
            target_embedding = None
            videos[current_video]['old_number'] = -1
        face_selector_var = tk.IntVar()
        face_selector_check = ttk.Checkbutton(left_frame, text="Face selector mode", variable=face_selector_var, style="TCheckbutton")
        face_selector_check.grid(row=row_counter, column=0)
        face_selector_var.set(0)
        row_counter += 1
        #unselect_face_button = tk.Button(left_frame, text='unselect the face button', bg=button_color, fg=text_color, command=unselect_face)
        #unselect_face_button.grid(row=row_counter, column=0)
        #row_counter += 1
        def toggle_menu():
            global menu_visible
            if menu_visible:
                advanced_section_frame.grid_remove()
            else:
                advanced_section_frame.grid(row=menu_counter, column=0, pady=10)
            menu_visible = not menu_visible
        occluder_checkbox_var = tk.IntVar()
        occluder_checkbox = ttk.Checkbutton(left_frame, text="Use occluder", variable=occluder_checkbox_var, style="TCheckbutton")
        occluder_checkbox.grid(row=row_counter, column=0)
        occluder_checkbox_var.set(1)
        row_counter += 1
        advanced_face_detector_var = tk.IntVar()
        advanced_face_detector_checkbox = ttk.Checkbutton(left_frame, text="advanced face detector", variable=advanced_face_detector_var, style="TCheckbutton")
        advanced_face_detector_checkbox.grid(row=row_counter, column=0)
        row_counter += 1
        
        toggle_frame_ = tk.Frame(left_frame, bg=background_color)
        toggle_frame_.grid(row=row_counter, column=0, sticky="ew")
        row_counter += 1
        show_advanced_settings = tk.Button(toggle_frame_, text='Toggle advanced settings', bg=button_color, fg=text_color, command=toggle_menu)
        show_advanced_settings.pack(side=tk.LEFT, fill=tk.X, expand=True)
        row_counter += 1
        
        def toggle_clip_menu():
            global clip_menu_visible
            if clip_menu_visible:
                clip_frame.grid_remove()
            else:
                clip_frame.grid(row=clip_menu_counter, column=0, pady=10)
            clip_menu_visible = not clip_menu_visible
            
        show_clip_settings = tk.Button(toggle_frame_, text='Toggle CLIP settings', bg=button_color, fg=text_color, command=toggle_clip_menu)
        show_clip_settings.pack(side=tk.LEFT, fill=tk.X, expand=True)
        row_counter += 1
        advanced_section_frame = tk.LabelFrame(left_frame, text="Advanced settings", bg=background_color, fg=text_color)
        advanced_section_frame.grid(row=row_counter, column=0, pady=10, sticky="nsew")
        advanced_section_frame.grid_columnconfigure(0, weight=1)
        menu_visible = True
        menu_counter = row_counter
        row_counter += 1



        label = tk.Label(advanced_section_frame, text="bounding box adjustment", fg=text_color, bg=background_color)
        label.grid(row=row_counter, column=0)
        row_counter += 1

        up_frame = tk.Frame(advanced_section_frame, bg=background_color)
        up_frame.grid(row=row_counter, column=0, rowspan=1, sticky="ew")
        row_counter += 1
        label = tk.Label(up_frame, text="up", fg=text_color, bg=background_color, width=10)
        label.pack(side=tk.LEFT, fill=tk.X)#, expand=True)
        entry_y1 = tk.Entry(up_frame)
        entry_y1.pack(side=tk.LEFT, fill=tk.X, expand=True)
        entry_y1.insert(0, adjust_y1)
        
        right_frame = tk.Frame(advanced_section_frame, bg=background_color)
        right_frame.grid(row=row_counter, column=0, rowspan=1, sticky="ew")
        label = tk.Label(right_frame, text="right", fg=text_color, bg=background_color, width=10)
        label.pack(side=tk.LEFT, fill=tk.X)#, expand=True)
        row_counter += 1
        
        entry_x2 = tk.Entry(right_frame)
        entry_x2.pack(side=tk.LEFT, fill=tk.X, expand=True)
        entry_x2.insert(0, adjust_x2)
        
        left__frame = tk.Frame(advanced_section_frame, bg=background_color)
        left__frame.grid(row=row_counter, column=0, rowspan=1, sticky="ew")
        label = tk.Label(left__frame, text="left", fg=text_color, bg=background_color, width=10)
        label.pack(side=tk.LEFT, fill=tk.X)#, expand=True)
        row_counter += 1
        
        entry_x1 = tk.Entry(left__frame)
        entry_x1.pack(side=tk.LEFT, fill=tk.X, expand=True)
        entry_x1.insert(0, adjust_x1)
        
        down_frame = tk.Frame(advanced_section_frame, bg=background_color)
        down_frame.grid(row=row_counter, column=0, rowspan=1, sticky="ew")
        label = tk.Label(down_frame, text="down", fg=text_color, bg=background_color, width=10)
        label.pack(side=tk.LEFT, fill=tk.X)#, expand=True)
        row_counter += 1
        
        entry_y2 = tk.Entry(down_frame)
        entry_y2.pack(side=tk.LEFT, fill=tk.X, expand=True)
        entry_y2.insert(0, adjust_y2)
        
        button = tk.Button(advanced_section_frame, text="Set Values", bg=button_color, fg=text_color, command=set_adjust_value)
        button.grid(row=row_counter, column=0)
        row_counter += 1
        
        label = tk.Label(advanced_section_frame, text="for these settings you need codeformer to be enabled", fg=text_color, bg=background_color)
        label.grid(row=row_counter, column=0)
        row_counter += 1
        
        label = tk.Label(advanced_section_frame, text="and tick on the face enhancer", fg=text_color, bg=background_color)
        label.grid(row=row_counter, column=0)
        row_counter += 1
        
        codeformer_fidelity = 0.1
        def on_codeformer_slider_move(value):
            global codeformer_fidelity
            codeformer_fidelity = float(value)
        
        label = tk.Label(advanced_section_frame, text="Codeformer fidelity", fg=text_color, bg=background_color)
        label.grid(row=row_counter, column=0)
        row_counter += 1
        
        codeformer_slider = tk.Scale(advanced_section_frame, from_=0.1, to=2.0, resolution=0.1,  orient=tk.HORIZONTAL, fg=text_color, bg=background_color, command=on_codeformer_slider_move)
        codeformer_slider.grid(row=row_counter, column=0, sticky='ew', padx=10)
        row_counter += 1
        
        alpha = 1.0
        def alpha_slider_move(value):
            global alpha
            alpha = float(value)
        
        label = tk.Label(advanced_section_frame, text="original/final blend", fg=text_color, bg=background_color)
        label.grid(row=row_counter, column=0)
        row_counter += 1
        
        alpha_slider = tk.Scale(advanced_section_frame, from_=0.0, to=1.0, resolution=0.1, fg=text_color, bg=background_color,  orient=tk.HORIZONTAL, command=alpha_slider_move)
        alpha_slider.grid(row=row_counter, column=0, sticky='ew', padx=10)
        alpha_slider.set(1.0)
        row_counter += 1
        label = tk.Label(advanced_section_frame, text="swapped/upscaled blend", fg=text_color, bg=background_color)
        label.grid(row=row_counter, column=0)
        row_counter += 1
        
        alpha2 = 1.0
        def alpha_slider_move(value):
            global alpha2
            alpha2 = float(value)
        alpha_slider2 = tk.Scale(advanced_section_frame, from_=0.0, to=1.0, resolution=0.1, fg=text_color, bg=background_color,  orient=tk.HORIZONTAL, command=alpha_slider_move)
        alpha_slider2.grid(row=row_counter, column=0, sticky='ew', padx=10)
        alpha_slider2.set(1.0)
        row_counter += 1
        
        codeformer_skip_if_no_face_var = tk.IntVar()
        codeformer_skip_if_no_face = ttk.Checkbutton(advanced_section_frame, text="Skip codeformer if not face is found", variable=codeformer_skip_if_no_face_var, style="TCheckbutton")
        codeformer_skip_if_no_face.grid(row=row_counter, column=0)
        row_counter += 1
        
        codeformer_upscale_face_var = tk.IntVar()
        codeformer_upscale_face = ttk.Checkbutton(advanced_section_frame, text="Upscale face using codeformer", variable=codeformer_upscale_face_var, style="TCheckbutton")
        codeformer_upscale_face.grid(row=row_counter, column=0)
        codeformer_upscale_face_var.set(1)
        row_counter += 1
        
        codeformer_enhance_background_var = tk.IntVar()
        codeformer_enhance_background = ttk.Checkbutton(advanced_section_frame, text="Enhance background using codeformer", variable=codeformer_enhance_background_var, style="TCheckbutton")
        codeformer_enhance_background.grid(row=row_counter, column=0)
        row_counter += 1
        
        codeformer_upscale_amount_value = 1
        def codeformer_upscale_amount_move(value):
            global codeformer_upscale_amount_value
            codeformer_upscale_amount_value = int(value)
        
        codeformer_upscale_amount = tk.Scale(advanced_section_frame, from_=1, to=3, resolution=1, fg=text_color, bg=background_color, orient=tk.HORIZONTAL, command=codeformer_upscale_amount_move)
        codeformer_upscale_amount.grid(row=row_counter, column=0, sticky='ew', padx=10)
        codeformer_upscale_amount.set(1)
        row_counter += 1
        
        #label = tk.Label(advanced_section_frame, text="codeformer settings finished", fg=text_color, bg=background_color)
        #label.grid(row=row_counter, column=0)
        #row_counter += 1
        clean_cache_button = tk.Button(advanced_section_frame, text="Clean VRAM", bg=button_color, fg=text_color, command=lambda:torch.cuda.empty_cache())
        clean_cache_button.grid(row=row_counter, column=0)
        row_counter += 1
        clip_frame = tk.LabelFrame(left_frame, text="CLIP settings", bg=background_color, fg=text_color)
        clip_frame.grid(row=row_counter, column=0, pady=10, sticky="nsew")
        clip_menu_visible = True
        clip_menu_counter = row_counter
        row_counter += 1

        enable_clip_var = tk.IntVar()
        enable_clip_ckeck = ttk.Checkbutton(clip_frame, text="Enable clip", variable=enable_clip_var, style="TCheckbutton")
        enable_clip_ckeck.pack(expand=True)#).grid(row=row_counter, column=0, sticky='ew')
        row_counter += 1
        
        label = tk.Label(clip_frame, text="positive prompt:", fg=text_color, bg=background_color)
        label.pack(expand=True)#.grid(row=row_counter, column=0, sticky='ew')
        row_counter += 1
        
        entry_clip_pos = tk.Entry(clip_frame)
        entry_clip_pos.pack(expand=True, fill=tk.X)#.grid(row=row_counter, column=0, sticky='ew')
        row_counter += 1
        
        label = tk.Label(clip_frame, text="negative prompt:", fg=text_color, bg=background_color)
        label.pack(expand=True)#.grid(row=row_counter, column=0, sticky='ew')
        row_counter += 1
        
        entry_clip_neg = tk.Entry(clip_frame)
        entry_clip_neg.pack(expand=True, fill=tk.X)#.grid(row=row_counter, column=0, sticky='ew')
        row_counter += 1

        def update_clip_values():
            global clip_neg_prompt, clip_pos_prompt
            clip_neg_prompt = entry_clip_neg.get()
            clip_pos_prompt = entry_clip_pos.get()
        button = tk.Button(clip_frame, text="Update CLIP values", bg=button_color, fg=text_color, command=update_clip_values)
        button.pack(expand=True)#.grid(row=row_counter, column=0, sticky='ew')
        row_counter += 1



        expander = tk.Label(left_frame, text=f"⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀", fg=text_color, bg=background_color)
        expander.grid(row=row_counter, column=0, sticky="ew")
        row_counter += 1
        
        def open_face_chooser():
            face_chooser_window.deiconify()
        def close_face_chooser():
            face_chooser_window.withdraw()
       
        face_chooser_window = tk.Toplevel(root, bg=background_color)
        face_chooser_window.attributes("-topmost", True)
        face_chooser_window.geometry("640x560")
        face_chooser_window.grid_rowconfigure(0, weight=1)
        face_chooser_window.grid_columnconfigure(0, weight=1)
        face_chooser_window.grid_rowconfigure(1, weight=1)

        video_faces = tk.Frame(face_chooser_window, bg=background_color)
        video_faces.grid(row=0, column=0, sticky="ew")
        video_faces.grid_rowconfigure(0, weight=1)
        video_faces.grid_columnconfigure(0, weight=1)
        faces_listbox = ScrolledListBox_horizontal(video_faces)
        faces_listbox.configure(background=background_color)
        faces_listbox.selector_color = selector_color
        faces_listbox.text_color = text_color
        faces_listbox.grid(row=0, column=0, sticky="nsew")
        added_faces = tk.Frame(face_chooser_window, bg=background_color)
        added_faces.grid(row=1, column=0, sticky="ew")
        added_faces.grid_rowconfigure(0, weight=1)
        added_faces.grid_columnconfigure(0, weight=1)
        faces_listbox2 = ScrolledListBox_horizontal(added_faces)
        faces_listbox2.configure(background=background_color)
        faces_listbox2.selector_color = selector_color
        faces_listbox2.text_color = text_color
        faces_listbox2.grid(row=0, column=0, sticky="nsew")
        faces_listbox2.other_widget = faces_listbox
        faces_listbox.other_widget = faces_listbox2
        faces_listbox2.order = 1
        face_chooser_window.update_idletasks()
        '''# Load some sample image pairs
        left_image_paths = ["face.jpg", "rick.png"]
        right_image_paths = left_image_paths[::-1]
        
        right_images = [Image.open(image_path) for image_path in left_image_paths]
        left_images = ["1", "2", "3"]
        
        # Insert data into the ScrolledImageList
        image_pair_list = list(zip(left_images, right_images))
        faces_listbox.insert_data(image_pair_list)
        faces_listbox2.insert_data(image_pair_list)
        faces_listbox2.add_item("banana", right_images[0])
        faces_listbox2.add_item("banana2", right_images[0])
        faces_listbox2.add_item("banana3", right_images[0])'''
        def find_and_add_faces():
            try:
                image = original_image_label.image
                image_width = image.width()
                image_height = image.height()
                #print(relative_x, relative_y)
                bboxes = []
                faces = face_analysers[0].get(videos[current_video]['original_image'])
                for face in faces:    
                    bboxes.append([face.bbox, face])
                height, width = videos[current_video]['original_image'].shape[:2]
                images = []
                for bbox, face in bboxes:
                    images.append([Image.fromarray(cv2.cvtColor(videos[current_video]['original_image'][int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])], cv2.COLOR_BGR2RGB)), face])
                good_images = []
                for i, f in images:
                    allowed = True
                    a = f.normed_embedding
                    for (_, _, tt) in faces_listbox.data_list:
                        _, allow = compute_cosine_distance(a,tt.normed_embedding , 0.75)
                        if allow:
                            allowed = False
                            break
                    if allowed:
                        faces_listbox.add_item("unselected", i, f)
                
            #for i in image_pair_list:
            #    faces_listbox2.add_item(*i)
            #faces_listbox2.insert_data(image_pair_list)
            except Exception as e:
                print(f"HUSTON, WE HAD A PROBLEM AT FINDING FACES: {e}")
        def add_faces():
            filex = askopenfilename(title="Select a face")
            if filex:
                im = cv2.imread(filex)
                bboxes = []
                faces = face_analysers[0].get(im)
                for face in faces:    
                    bboxes.append([face.bbox, face])
                images = []
                for bbox, face in bboxes:
                    images.append([Image.fromarray(cv2.cvtColor(im[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])], cv2.COLOR_BGR2RGB)), face])
                for i, f in images:
                    a = f.normed_embedding
                    faces_listbox2.add_item(len(faces_listbox2.data_list), i, f)


        find_faces_in_frame_button = tk.Button(face_chooser_window, text="Find faces", bg=button_color, fg=text_color, command=find_and_add_faces)
        find_faces_in_frame_button.grid(row=2, column=0)
        add_faces_in_frame_button = tk.Button(face_chooser_window, text="Add faces", bg=button_color, fg=text_color, command=add_faces)
        add_faces_in_frame_button.grid(row=3, column=0)
        unselect_face_button = tk.Button(face_chooser_window, text="unselect chosen face", bg=button_color, fg=text_color, command=faces_listbox.unselect)
        unselect_face_button.grid(row=4, column=0)
        face_chooser_window.withdraw()
        face_chooser_window.protocol("WM_DELETE_WINDOW", close_face_chooser)


        buttonxx = tk.Button(left_frame, text="Choose faces", bg=button_color, fg=text_color, command=open_face_chooser)
        buttonxx.grid(row=row_counter, column=0)
        row_counter += 1
        
        #=============================================
        #render_button.grid(row=34, column=0)
        right_frame1 = tk.LabelFrame(root, text="Original frame", bg=background_color, highlightthickness=2, highlightbackground=border_color, fg=text_color)
        right_frame2 = tk.LabelFrame(root, text="Swapped frame", bg=background_color, highlightthickness=2, highlightbackground=border_color, fg=text_color)
        ui_vertical = 0
        show_orig = 1
        show_swapped = 1
        def update_ui(clicked=0):
            global right_frame1, right_frame2, ui_vertical, show_orig, show_swapped
            if clicked == 0:
                ui_vertical = not ui_vertical
            if clicked == 1:
                show_orig = not show_orig
            if clicked == 2:
                show_swapped = not show_swapped
            right_frame1.grid_remove()
            right_frame2.grid_remove()
            left_frame.grid_remove()
            if show_orig:
                show_hide_original_button.configure(text="Hide original video")
                right_frame1.grid(row=0, column=1, columnspan=1+(not ui_vertical), rowspan = 1 +ui_vertical, sticky="nsew")
            else:
                show_hide_original_button.configure(text="Show original video")
            if show_swapped:
                show_hide_swapped_button.configure(text="Hide swapped video")
                right_frame2.grid(row=0+(not ui_vertical), column=1+ui_vertical, columnspan=1+(not ui_vertical), rowspan = 1 +ui_vertical, sticky="nsew")
            else:
                show_hide_swapped_button.configure(text="Show swapped video")
            left_frame.grid(row=0, column=3, rowspan=2, sticky="ns")
            right_frame1.grid_propagate(True)
            right_frame2.grid_propagate(True)
            if ui_vertical:   
                root.grid_columnconfigure(0, weight=1)
                root.grid_columnconfigure(1, weight=4*show_orig)
                root.grid_columnconfigure(2, weight=4*show_swapped)
                root.grid_columnconfigure(3, weight=1)
                root.grid_rowconfigure(0, weight=1) 
                root.grid_rowconfigure(1, weight=1)             
            
            else:
                root.grid_columnconfigure(0, weight=1)
                root.grid_columnconfigure(1, weight=4)
                root.grid_columnconfigure(2, weight=4)
                root.grid_columnconfigure(3, weight=1)
                root.grid_rowconfigure(0, weight=show_orig) 
                root.grid_rowconfigure(1, weight=show_swapped)
            right_frame1.grid_propagate(False)
            right_frame2.grid_propagate(False)
            root.update_idletasks()
            root.update()
        
        UI_button_frame = tk.Frame(left_frame, bg=background_color)
        UI_button_frame.grid(row=row_counter, column=0, sticky="ew")
        row_counter += 1
        make_vertical_button = tk.Button(UI_button_frame, text="vertical/horizontal video", bg=button_color, fg=text_color, command=lambda:update_ui(clicked=0))
        make_vertical_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        show_hide_original_button = tk.Button(UI_button_frame, text="Hide original video", bg=button_color, fg=text_color, command=lambda:update_ui(clicked=1))
        show_hide_original_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        show_hide_swapped_button = tk.Button(UI_button_frame, text="Hide swapped video", bg=button_color, fg=text_color, command=lambda:update_ui(clicked=2))
        show_hide_swapped_button.pack(side=tk.LEFT, fill=tk.X, expand=True)


        original_image_label = tk.Label(right_frame1, bg=background_color)#, text="Original frame placeholder")
        swapped_image_label = tk.Label(right_frame2, bg=background_color)#, text="Swapped frame placeholder")
        right_frame1.grid_propagate(False)
        right_frame2.grid_propagate(False)
        if mode == 1:
            # Side by side configuration
            right_frame1.grid(row=0, column=1, sticky="nsew")
            original_image_label.pack(padx=15, pady=15)

            right_frame2.grid(row=0, column=2, sticky="nsew")
            swapped_image_label.pack(padx=15, pady=15)
            
            # Configure column weights
            root.grid_columnconfigure(0, weight=1)
            root.grid_columnconfigure(1, weight=4)
            root.grid_columnconfigure(2, weight=4)
            root.grid_columnconfigure(3, weight=1)
            root.grid_rowconfigure(0, weight=1)

        else:
            # Stacked configuration
            right_frame1.grid(row=0, column=1, columnspan=2, sticky="nsew")
            original_image_label.grid(sticky="nsew", padx=15, pady=10)

            right_frame2.grid(row=1, column=1, columnspan=2, sticky="nsew")
            swapped_image_label.grid(sticky="nsew", padx=15, pady=10)
            
            # Configure column and row weights
            root.grid_columnconfigure(0, weight=1)
            root.grid_columnconfigure(1, weight=4)
            root.grid_columnconfigure(2, weight=4)
            root.grid_columnconfigure(3, weight=1)
            root.grid_rowconfigure(0, weight=1) 
            root.grid_rowconfigure(1, weight=1)
        def on_image_click(event):
            global target_embedding, old_index, args
            if face_selector_var.get() == 1:
                image = original_image_label.image
                image_width = image.width()
                image_height = image.height()
                
                relative_x = event.x / image_width
                relative_y = event.y / image_height
                #print(relative_x, relative_y)
                bboxes = []
                faces = face_analysers[0].get(videos[current_video]['original_image'])
                for face in faces:    
                    bboxes.append(face.bbox)
                height, width = videos[current_video]['original_image'].shape[:2]
                real_x = height*relative_y
                real_y = width*relative_x
                for bbox in bboxes:
                    if real_y > bbox[0] and real_y < bbox[2] and real_x > bbox[1] and real_x < bbox[3]:
                        this_face = videos[current_video]['original_image'][int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                        args['selective'] = True
                        target_embedding = get_embedding(this_face)[0]
                        videos[current_video]['old_number'] = -1
                        cv2.imshow("cropped face", this_face)
                        cv2.waitKey(0)
                        try:
                            cv2.destroyWindow("cropped face")
                        except:
                            break
                        break
        original_image_label.bind("<Button-1>", on_image_click)
        def left_arrow_click(event):
            global current_video, slider, frame_count_label
            #current_video = max(0, current_video-1)
            #slider.config(to=videos[current_video]["frame_number"])
            #frame_count_label.config(text=f"total frames: {videos[current_video]['frame_number']}") 
            #root.update_idletasks()
            edit_index(-1)
        def right_arrow_click(event):
            global current_video, slider, frame_count_label
            #maxi = len(videos)-1
            #current_video = min(maxi, current_video + 1)
            #slider.config(to=videos[current_video]["frame_number"])
            #frame_count_label.config(text=f"total frames: {videos[current_video]['frame_number']}") 
            #root.update_idletasks()
            edit_index(1)
        def space_click(event):
            global frame_move
            if frame_move == 0:
                frame_move = 1
            else:
                frame_move = 0
        def update_wraplength(event):
            select_face_label.config(wraplength=left_frame.winfo_width() - 20)  # Subtract a small padding
            select_target_label.config(wraplength=left_frame.winfo_width() - 20)  # Subtract a small padding
            select_output_label.config(wraplength=left_frame.winfo_width() - 20)  # Subtract a small padding

        root.bind("<Left>", left_arrow_click)
        root.bind("<Right>", right_arrow_click)
        root.bind("<space>", space_click)
        root.bind("<Configure>", update_wraplength)
        right_control_frame = tk.Frame(root, bg=background_color)
        right_control_frame.grid(row=0, column=0, rowspan=2, sticky="ns")
        def add():
            global videos
            facex = sorted(face_analysers[0].get(cv2.imread(args["face"])), key=lambda x: x.bbox[0])[0]
            videos.append(create_new_cap(args['target_path'], facex, args['output'],))
        def delete_current_video():
            global videos, current_video
            videos.pop(current_video)
            current_video = 0
        #yes, one day Im going to release that thing
        button_select_face = tk.Button(right_control_frame, text='Select face',bg=button_color, fg=text_color, command=select_face)
        button_select_face.grid(row=0, column=0, pady=3, sticky='ew')
        select_face_label = tk.Label(right_control_frame, text=f'Face filename: {args["face"]}', fg=text_color, bg=background_color)
        select_face_label.grid(row=1, column=0, pady=3)
        canvas = tk.Canvas(right_control_frame, height=2, bg=border_color, highlightthickness=0)
        canvas.grid(row=2, column=0, columnspan=2, sticky='ew', padx=0, pady=4)
        button_select_target = tk.Button(right_control_frame, text='Select target',bg=button_color, fg=text_color, command=select_target)
        button_select_target.grid(row=3, column=0, pady=3, sticky='ew')
        select_target_label = tk.Label(right_control_frame, text=f'Target filename: {args["target_path"]}', fg=text_color, bg=background_color)
        select_target_label.grid(row=4, column=0, pady=3)
        canvas2 = tk.Canvas(right_control_frame, height=2, bg=border_color, highlightthickness=0)
        canvas2.grid(row=5, column=0, columnspan=2, sticky='ew', padx=0, pady=4)
        button_select_output = tk.Button(right_control_frame, text='Select output',bg=button_color, fg=text_color, command=select_output)
        button_select_output.grid(row=6, column=0, pady=3, sticky='ew')
        select_output_label = tk.Label(right_control_frame, text=f'output filename: {args["output"]}', fg=text_color, bg=background_color)
        select_output_label.grid(row=7, column=0, pady=3)
        canvas3 = tk.Canvas(right_control_frame, height=2, bg=border_color, highlightthickness=0)
        canvas3.grid(row=8, column=0, columnspan=2, sticky='ew', padx=0, pady=4)
        start_video_frame = tk.Frame(right_control_frame, bg=background_color)
        start_video_frame.grid(row=9, column=0, sticky="ew")
        button_start_program = tk.Button(start_video_frame, text="Add this video",bg=button_color, fg=text_color, command=add)
        button_start_program.pack(side=tk.LEFT, fill=tk.X, expand=True)#.grid(row=10, column=0, pady=3, sticky='ew')
        button_select_camera = tk.Button(start_video_frame, text='run from camera',bg=button_color, fg=text_color, command=select_camera)
        button_select_camera.pack(side=tk.LEFT, fill=tk.X, expand=True)#.grid(row=9, column=0, pady=3, sticky='ew')
        #thread_amount_label = tk.Label(right_control_frame, text='Select the number of threads', fg=text_color, bg=background_color)
        #thread_amount_label.grid(row=8, column=0)
        #thread_amount_input = tk.Entry(right_control_frame)
        #thread_amount_input.grid(row=9, column=0)
        selector_frame = tk.Frame(right_control_frame, bg=background_color)
        selector_frame.grid(row=11, column=0, sticky="nsew", pady=15)
        
        Scrolledlistbox1 = ScrolledListBox(selector_frame)
        Scrolledlistbox1.configure(background=background_color)
        Scrolledlistbox1.selector_color = selector_color
        Scrolledlistbox1.text_color = text_color
        Scrolledlistbox1.grid(row=0, column=0, sticky="nsew")
        right_control_frame.grid_rowconfigure(11, weight=10)
        button_select_output = tk.Button(right_control_frame, text='Delete selected video',bg=button_color, fg=text_color, command=delete_current_video)
        button_select_output.grid(row=12, column=0, sticky='ew', pady=4)
        render_button_frame = tk.Frame(right_control_frame, bg=background_color)
        render_button_frame.grid(row=13, column=0, pady=10, sticky="ew")
        row_counter += 1
        render_button = tk.Button(render_button_frame, text='render', bg=button_color, fg=text_color, command=run_it_please)
        render_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        stop_rendering_button = tk.Button(render_button_frame, text='stop render', bg=button_color, fg=text_color, command=not_run_it_please)
        stop_rendering_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        stop_rendering_button.config(state=tk.DISABLED)
        def update_progress_bar(length, progress, total, gpu_usage, vram_usage, total_vram):
            global videos
            try:
                if videos[current_video]['rendering'] and not isinstance(args['target_path'], int):
                    filled_length = int(length * progress // total)
                    bar = '█' * filled_length + '—' * (length - filled_length)
                    percent = round(100.0 * progress / total, 1)
                    progress_text = f'Progress: |{bar}| {percent}% {progress}/{total}'
                    progress_label['text'] = progress_text
                if not args['nocuda'] and not args['apple']:
                    usage_label1['text'] = f"gpu usage: {gpu_usage}%|VRAM usage: {vram_usage}/{total_vram}GB"
                    ram_usage, total_ram, cpu_usage = get_system_usage()
                    usage_label2['text'] = f"cpu usage: {cpu_usage}%|RAM usage: {ram_usage}/{total_ram}GB"
                else:
                    ram_usage, total_ram, cpu_usage = get_system_usage()
                    usage_label1['text'] = f"cpu usage: {cpu_usage}%|RAM usage: {ram_usage}/{total_ram}GB"
                #progress_var.set(text=progress_text)
                root.update()
            except:
                return
    def face_analyser_thread(frame, sw):
        global alpha2, codeformer
        original_frame = frame.copy()
        if not args['cli']:
            test1 = alpha != 0
        else:
            test1 = args['alpha'] != 0
        _upscaled = False
        if test1:
            advanced_search = False
            if not args['cli']:
                advanced_search = advanced_face_detector_var.get()
            else:
                pass
            
            if not advanced_search:
                faces = face_analysers[sw].get(frame)
                bboxes = []
                for face in faces:
                    if args['selective'] != '':
                        #a = target_embedding.normed_embedding
                        allowed_list = []
                        for xx in range(len(faces_listbox.data_list)):
                            a = faces_listbox.data_list[xx][2].normed_embedding
                            b = face.normed_embedding
                            _, allow = compute_cosine_distance(a,b , 0.75)
                            allowed_list.append(allow)
                        if len(allowed_list) == 0:
                            continue
                        try:
                            indexx = allowed_list.index(True)
                        except ValueError:
                            continue
                            #if not allow:
                            #    break
                        if faces_listbox.data_list[indexx][0] == "unselected":
                            continue
                    bboxes.append(face.bbox)
                    ttest1 = False
                    if not args['cli']:
                        if faceswapper_checkbox_var.get() == True:
                            ttest1=True
                    if not args['no_faceswap'] and (ttest1 == True or args['cli']):
                        occluder_works= False
                        if not args['cli']:
                            occluder_works = int(occluder_checkbox_var.get())
                            #print(occluder_works)
                        clip_works = False
                        if not args['cli']:
                            clip_works = int(enable_clip_var.get())
                        if args['selective'] != '':
                            try:
                                fff = faces_listbox2.data_list[int(faces_listbox.data_list[indexx][0])][2]
                            except:
                                print("errorrr")
                                fff = get_source_face()
                        else:
                            fff = get_source_face()
                        frame = face_swappers[sw].get(frame, face, fff,occluder_works, clip_works, [clip_pos_prompt, clip_neg_prompt], paste_back=True)
                        swapped_frame = frame.copy()
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
                            x2 = min(x2+adjust_x2, videos[current_loop_video]['width'])
                            y2 = min(y2+adjust_y2, videos[current_loop_video]['height'])
                            facer = frame[y1:y2, x1:x2]
                            if not args['cli']:
                                enhancer_choice_value = enhancer_choice.get()
                            else:
                                enhancer_choice_value = args['face_enhancer']
                            if enhancer_choice_value == "fastface enhancer" or (enhancer_choice_value == "ffe" and not args['lowmem']):
                                facex = upscale_image(facer, load_generator())
                            elif enhancer_choice_value == "gfpgan":
                                facex = restorer_enhance(facer)
                            elif enhancer_choice_value == "gfpgan onnx" or enhancer_choice_value == "gpfgan_onnx":
                                facex = load_gfpganonnx().forward(facer)
                            elif enhancer_choice_value == "real esrgan" or enhancer_choice_value == "real_esrgan":
                                facex = realesrgan_enhance(facer)
                            facex = cv2.resize(facex, ((x2-x1), (y2-y1)))
                            frame[y1:y2, x1:x2] = facex
                            _upscaled = True
                        except Exception as e:
                            print(f"ee: {e}")
            else:
                init_advanced_face_detector() #will not do anything if loaded
                rotation_angles = calculate_rotation_angles(frame)
                for it in range(len(rotation_angles)):
                    #a = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    #cv2.imshow('a', a)
                    #cv2.waitKey(0)
                    #rotation_angle = calculate_rotation_angle(a)
                    print(rotation_angles[it])
                    ss = frame.shape
                    frame, rotation_matrix = rotate_image(frame, -rotation_angles[it])
                    
                    rotated=False
                    faces = face_analysers[sw].get(frame)
                    for face in faces:
                        if args['selective'] != '':
                            a = target_embedding.normed_embedding
                            b = face.normed_embedding
                            _, allow = compute_cosine_distance(a,b , 0.75)
                            if not allow:
                                continue
                        #bboxes.append(face.bbox)
                        ttest1 = False
                        if not args['cli']:
                            if faceswapper_checkbox_var.get() == True:
                                ttest1=True
                        if not args['no_faceswap'] and (ttest1 == True or args['cli']):
                            occluder_works= False
                            if not args['cli']:
                                occluder_works = int(occluder_checkbox_var.get())
                                #print(occluder_works)
                            clip_works = False
                            if not args['cli']:
                                clip_works = int(enable_clip_var.get())
                            frame = face_swappers[sw].get(frame, face, get_source_face(),occluder_works, clip_works, [clip_pos_prompt, clip_neg_prompt], paste_back=True)
                            cv2.imshow('a', frame)
                            #frame = frame[0:videos[current_loop_video]['height'], 0:videos[current_loop_video]['width']]
                            rotated = True
                            swapped_frame = rotate_back(frame, rotation_matrix, ss).copy()
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
                                x2 = min(x2+adjust_x2, videos[current_loop_video]['width'])
                                y2 = min(y2+adjust_y2, videos[current_loop_video]['height'])
                                facer = frame[y1:y2, x1:x2]
                                if not args['cli']:
                                    enhancer_choice_value = enhancer_choice.get()
                                else:
                                    enhancer_choice_value = args['face_enhancer']
                                if enhancer_choice_value == "fastface enhancer" or (enhancer_choice_value == "ffe" and not args['lowmem']):
                                    facex = upscale_image(facer, load_generator())
                                elif enhancer_choice_value == "gfpgan":
                                    facex = restorer_enhance(facer)
                                elif enhancer_choice_value == "gfpgan onnx" or enhancer_choice_value == "gpfgan_onnx":
                                    facex = load_gfpganonnx().forward(facer)
                                elif enhancer_choice_value == "real esrgan" or enhancer_choice_value == "real_esrgan":
                                    facex = realesrgan_enhance(facer)
                                facex = cv2.resize(facex, ((x2-x1), (y2-y1)))
                                frame[y1:y2, x1:x2] = facex
                                _upscaled = True
                            except Exception as e:
                                print(f"ee: {e}")

                    frame = rotate_back(frame, rotation_matrix, ss)#[0:videos[current_loop_video]['height'], 0:videos[current_loop_video]['width']]


            if not args['cli']:
                if enhancer_choice.get() == "codeformer" and checkbox_var.get() == 1 : 
                    if args['fastload']:
                        from plugins.codeformer_app_cv2 import inference_app as codeformer
                    #frame, background enhance bool true, face upscample bool true, upscale int 2,
                    # codeformer fidelity float 0.8, skip_if_no_face bool false 
                    frame = codeformer(frame, codeformer_enhance_background_var.get(), codeformer_upscale_face_var.get(), codeformer_upscale_amount_value, codeformer_fidelity, codeformer_skip_if_no_face_var.get())
            else:
                if args['face_enhancer'] == 'codeformer':
                    if args['fastload']:
                        from plugins.codeformer_app_cv2 import inference_app as codeformer
                    frame = codeformer(frame, args['codeformer_background_enhance'], args['codeformer_face_upscale'], args['codeformer_upscale'], float(args['codeformer_fidelity']), args['codeformer_skip_if_no_face'])
            
            if not args['cli']:
                test1 = alpha2 != 1
            else:
                test1 = args['alpha'] != 1
            if test1:
                #print(alpha)
                if _upscaled:
                    frame = merge_face(frame, swapped_frame, alpha2)
            if not args['cli']:
                test1 = alpha != 1
            else:
                test1 = args['alpha'] != 1
            if test1:
                #print(alpha)
                frame = merge_face(frame, original_frame, alpha)
            return [], frame, original_frame
        return [], frame, original_frame

    def cv2_image_to_tkinter(cv2_image, target_width, target_height, pad_width=30, pad_height=50):
        """Convert a cv2 image to a tkinter compatible format and resize it to fit target dimensions."""
        cv2_img_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cv2_img_rgb)

        target_width -= pad_width
        target_height -= pad_height
        
        # Resize the image while maintaining its aspect ratio
        image_aspect = pil_image.width / pil_image.height
        target_aspect = target_width / target_height

        if image_aspect > target_aspect:
            # Image is wider than target, fit to width
            width = target_width
            height = int(target_width / image_aspect)
        else:
            # Image is taller or equal to target, fit to height
            height = target_height
            width = int(target_height * image_aspect)

        pil_image_resized = pil_image.resize((width, height), Image.Resampling.LANCZOS)
        
        return ImageTk.PhotoImage(pil_image_resized)
    def frame_updater(xx=True):
        try:
            if not isinstance(videos[current_video]['original_image'], NoneType) and not isinstance(videos[current_video]['swapped_image'], NoneType):
                    sizex1, sizey1 = right_frame1.winfo_width(), right_frame1.winfo_height()
                    sizex2, sizey2 = right_frame2.winfo_width(), right_frame2.winfo_height()
                    tk_imagex = cv2_image_to_tkinter(videos[current_video]['original_image'], sizex1, sizey1)
                    original_image_label.configure(image=tk_imagex)
                    original_image_label.image = tk_imagex  # Keep a reference to prevent garbage collection
                    tk_image = cv2_image_to_tkinter(videos[current_video]['swapped_image'], sizex2, sizey2)
                    swapped_image_label.configure(image=tk_image)
                    swapped_image_label.image = tk_image
                    if original_image_second_open:
                        sizex1, sizey1 = original_image_label_second.winfo_width(), original_image_label_second.winfo_height()
                        tk_imagex = cv2_image_to_tkinter(videos[current_video]['original_image'], sizex1, sizey1,0,0)
                        original_image_label_second.configure(image=tk_imagex)
                        original_image_label_second.image = tk_imagex  # Keep a reference to prevent garbage collection
                    if swapped_image_second_open:
                        sizex2, sizey2 = swapped_image_second_window.winfo_width(), swapped_image_second_window.winfo_height()
                        tk_image = cv2_image_to_tkinter(videos[current_video]['swapped_image'], sizex2, sizey2, 0,0)
                        swapped_image_label_second.configure(image=tk_image)
                        swapped_image_label_second.image = tk_image  # Keep a reference to prevent garbage collection
                        
            else:
                    original_image_label.configure(image=None)
                    original_image_label.image = None  # Keep a reference to prevent garbage collection
                    swapped_image_label.configure(image=None)
                    swapped_image_label.image = None
                    if original_image_second_open:
                        original_image_label_second.configure(image=None)
                        original_image_label_second.image = None  # Keep a reference to prevent garbage collection
                    if swapped_image_second_open:
                        swapped_image_label_second.configure(image=None)
                        swapped_image_label_second.image = None  # Keep a reference to prevent garbage collection

        except:
            
            original_image_label.configure(image=None)
            original_image_label.image = None  # Keep a reference to prevent garbage collection
            swapped_image_label.configure(image=None)
            swapped_image_label.image = None
            if original_image_second_open:
                original_image_label_second.configure(image=None)
                original_image_label_second.image = None  # Keep a reference to prevent garbage collection
            if swapped_image_second_open:
                swapped_image_label_second.configure(image=None)
                swapped_image_label_second.image = None  # Keep a reference to prevent garbage collection
            pass
        if xx:
            root.after(30, frame_updater)
    def get_embedding(face_image):
        try:
            return face_analysers[0].get(face_image)
        except IndexError:
            return None

    def process_image(input_path, output_path, sw):
        image = cv2.imread(input_path)
        bbox, image, original_frame = face_analyser_thread(image,sw)
        try:
            test1 = checkbox_var.get() == 1
        except:
            test1 = False
        if test1 or (args['face_enhancer'] != 'none' and args['cli']):
            image = restorer_enhance(image)
        cv2.imwrite(output_path, image)

    def just_preload_them(sw, frame):
        for i in range(int(args['threads'])):
            threading.Thread(target=load, args=(sw, frame)).start()
    def load(sw,frame):
        faces = face_analysers[sw].get(frame)
        face = list(faces)[0]
        face_swappers[sw].get(frame, face, source_face, paste_back=True)

    def source_face_creator(input_face):
        global source_face
        source_face = sorted(face_analysers[0].get(input_face), key=lambda x: x.bbox[0])[0]





    def optimize_saver():
        import pickle
        x = face_analysers[0].get(cv2.imread(args['face']))[0] #sorted(face_analysers[0].get(cv2.imread(args['face'])), key=lambda x: x.bbox[0])[0]
        #print(x)
        ll = {}
        for key, value in x.items():
            ll[key] = value
        ll = simulate(ll['bbox'], ll['kps'],ll['det_score'],ll['embedding'], ll['embedding'])
            #print(key, ":", value)
        #frame = face_swappers[0].get(cv2.imread(args['face']), face, get_source_face(), paste_back=True)
        
        
        #ll.bbox = x.get('bbox')
        #ll.kps = x.kps
        #ll.det_score = x.det_score
        #ll.embedding = x.embedding
        
        with open('ll.pkl', 'wb') as file:
            pickle.dump(ll, file)
            
        with open('ll.pkl', 'rb') as file:
            loaded_data = pickle.load(file)
        frame = face_swappers[0].get(cv2.imread(args['face']), loaded_data, get_source_face(), paste_back=True)
        print('nice')
        exit()

    def open_second_window():
        def run_it_please():
            global videos
            videos[current_video]['rendering'] = 0
            second_window.destroy()
        second_window = tk.Toplevel(root)
        label = tk.Label(second_window, text='press the button to start rendering')
        label.pack()
        button = tk.Button(second_window, text='Start', command=run_it_please)
        button.pack()

    def open_video_settings():
        
        button_select_face = tk.Button(right_control_frame, text='Select face',bg=button_color, fg=text_color, command=select_face)
        button_select_face.grid(row=0, column=0, pady=3, sticky='ew')
        select_face_label = tk.Label(right_control_frame, text=f'Face filename: {args["face"]}', fg=text_color, bg=background_color)
        select_face_label.grid(row=1, column=0, pady=3)
        canvas = tk.Canvas(right_control_frame, height=2, bg=border_color, highlightthickness=0)
        canvas.grid(row=2, column=0, columnspan=2, sticky='ew', padx=0, pady=4)
        button_select_camera = tk.Button(right_control_frame, text='run from camera',bg=button_color, fg=text_color, command=select_camera)
        button_select_camera.grid(row=3, column=0, pady=3, sticky='ew')
        button_select_target = tk.Button(right_control_frame, text='Select target',bg=button_color, fg=text_color, command=select_target)
        button_select_target.grid(row=4, column=0, pady=3, sticky='ew')
        select_target_label = tk.Label(right_control_frame, text=f'Target filename: {args["target_path"]}', fg=text_color, bg=background_color)
        select_target_label.grid(row=5, column=0, pady=3)
        canvas2 = tk.Canvas(right_control_frame, height=2, bg=border_color, highlightthickness=0)
        canvas2.grid(row=6, column=0, columnspan=2, sticky='ew', padx=0, pady=4)
        button_select_output = tk.Button(right_control_frame, text='Select output',bg=button_color, fg=text_color, command=select_output)
        button_select_output.grid(row=7, column=0, pady=3, sticky='ew')
        select_output_label = tk.Label(right_control_frame, text=f'output filename: {args["output"]}', fg=text_color, bg=background_color)
        select_output_label.grid(row=8, column=0, pady=3)
        canvas3 = tk.Canvas(right_control_frame, height=2, bg=border_color, highlightthickness=0)
        canvas3.grid(row=9, column=0, columnspan=2, sticky='ew', padx=0, pady=4)
        button_start_program = tk.Button(right_control_frame, text="Add this video",bg=button_color, fg=text_color, command=add)
        button_start_program.grid(row=10, column=0, pady=3, sticky='ew')

    def main():
        global current_video, videos,old_index, args, width, height, frame_index, face_analysers,frame_move, face_swappers, source_face, progress_var, target_embedding, count, frame_number, listik, frame, cap,current_loop_video
        #start = time.time()
        if not args['fastload']:
            face_swappers, face_analysers = prepare_swappers_and_analysers(args)
        #optimize_saver()
        #if args['fastload']:
        #    source_face_thread = threading.Thread(target=source_face_creator, args=(input_face,))
        #    source_face_thread.start()
        #else:
        #    source_face = sorted(face_analysers[0].get(input_face), key=lambda x: x.bbox[0])[0]
        gpu_usage = 0
        vram_usage = 0
        if args['selective'] != '':
            if args['selective'] != True:
                im = cv2.imread(args['selective'])
                #im = cv2.resize(im, (640, 640))
                target_embedding = get_embedding(im)[0]
        if args['image'] == True :
            images = []
            if args['batch'] != "":
                for i in os.listdir(args['target_path']):
                    if is_picture_file(i):
                        images.append([os.path.join(args['target_path'], i), os.path.join(args['output'], f"{i}{args['batch']}.png")])
            else:
                images.append([args['target_path'], args['output']])
            original_threads = threading.active_count()
            image_amount = len(images)
            for it, i in tqdm(enumerate(images)):
                if not args['nocuda'] and not args['apple']:
                    vram_usage, gpu_usage = round(gpu_memory_total - torch.cuda.mem_get_info(device)[0] / 1024**3,2), torch.cuda.utilization(device=device)
                    listik = [it, image_amount, gpu_usage, vram_usage, gpu_memory_total]
                else:
                    listik = [it, image_amount, 0, 0, 0]
                threading.Thread(target=process_image, args=(i[0], i[1], it%len(face_swappers))).start()
                while threading.active_count() > (int(args['threads']) + original_threads):
                    time.sleep(0.01)
            while threading.active_count() > original_threads:
                time.sleep(0.01)
            print("image processing finished")
            exit()
        caps = []
        #if args['batch'] == '':
        #    videos.append(create_new_cap(args['target_path']))
        #    #caps.append(create_cap())
        #else:
        #    for file in os.listdir(args['target_path']):
        #        if is_video_file(file):
        #            caps.append(create_batch_cap(file))
        #if args['fastload']:
        #    source_face_thread.join()
        #print(time.time()-start)
        if args['fastload'] and args['cli']:
            tx.join()
        elif args['fastload'] and not args['cli']:
            for t in tx:
                t.join()

        #videos[current_video]['rendering'] = int(args['cli'])
        #if not args['cli'] and not args['preview']:
        #    open_second_window()
        #for cap, fps, width, height, out, name, file, frame_number in caps:
        while 1 == 1:
            while len(videos) == 0:
                time.sleep(0.01)
            #root.after(1, update_progress_length, frame_number)
            #update_progress_bar( 10, 0, frame_number)
            count = -1
            #videos[current_video]['current_frame_index'] = count
            if args['vcam'] and videos[current_video]["type"] == 1:
                cam = pyvirtualcam.Camera(width=width, height=height, fps=videos[current_video]["fps"])
            progressbar = tqdm(total=videos[current_video]["frame_number"])
            bbox = []
            start = time.time()
            '''if not args['preview']:
                for i in range(int(args['threads'])):
                    if args['experimental']:
                        frame = cap.read()
                        if isinstance(frame, NoneType):
                            break
                    else:
                        ret, frame = cap.read()
                        if not ret:
                            break
                    temp.append(ThreadWithReturnValue(target=face_analyser_thread, args=(frame,count%len(face_swappers))))
                    temp[-1].start()
                    count += 1'''
            xxs = True
            
            try:
                while True:
                    try:
                        #print(videos[current_video]['rendering'])
                        current_loop_video = current_video
                        if videos[current_loop_video]['rendering'] and ((videos[current_loop_video]['rendering'] and not args['cli']) or args['cli']):
                            
                            videos[current_loop_video]['count'] += 1
                            #print(videos[current_video]['count'])
                            #if not isinstance(args['target_path'], int):
                            videos[current_loop_video]['current_frame_index'] = videos[current_loop_video]['count']
                            if videos[current_loop_video]['count'] == 0:
                                progressbar.reset()
                            if args['experimental']:
                                frame = videos[current_loop_video]["cap"].read()
                                if isinstance(frame, NoneType): #== None:
                                    break
                            else:
                                ret, frame = videos[current_loop_video]["cap"].read()
                                if not ret:
                                    break
                            #print("red cap")
                            if videos[current_loop_video]['count'] % 1000 == 999:
                                torch.cuda.empty_cache()
                            videos[current_loop_video]['temp'].append(ThreadWithReturnValue(target=face_analyser_thread, args=(frame,videos[current_loop_video]['count']%len(face_swappers))))
                            videos[current_loop_video]['temp'][-1].start()
                            if len(videos[current_loop_video]['temp']) < int(args['threads']) * len(face_swappers) and ret:
                                continue
                            while len(videos[current_loop_video]['temp']) >= int(args['threads']) * len(face_swappers):
                                bbox, videos[current_loop_video]["swapped_image"], videos[current_loop_video]['original_image'] = videos[current_loop_video]['temp'].pop(0).join()
                            xxs = True
                        else:
                            
                            if not videos[current_loop_video]['current_frame_index'] == videos[current_loop_video]['old_number'] or int(realtime_updater_var.get()):
                                #if not videos[current_loop_video]['rendering']:
                                if not isinstance(videos[current_loop_video]['target_path'], int):
                                    videos[current_loop_video]['old_number'] = videos[current_loop_video]['current_frame_index']
                                    videos[current_loop_video]['current_frame_index'] += frame_move
                                    if videos[current_loop_video]['current_frame_index'] < 1:
                                        videos[current_loop_video]['current_frame_index'] = 1
                                    elif videos[current_loop_video]['current_frame_index'] > videos[current_loop_video]["frame_number"]:
                                        videos[current_loop_video]['current_frame_index'] = videos[current_loop_video]["frame_number"]
                                else:videos[current_loop_video]['current_frame_index'] = 0
                                bbox, videos[current_loop_video]["swapped_image"], videos[current_loop_video]['original_image'] = face_analyser_thread(get_nth_frame(videos[current_loop_video]["cap"], videos[current_loop_video]['current_frame_index']-1), count%len(face_swappers))
                            xxs = False
                        if not args['cli']:
                            if show_bbox_var.get() == 1 or face_selector_var.get() == 1:
                                for i in bbox: 
                                    x1, y1, x2, y2 = int(i[0]),int(i[1]),int(i[2]),int(i[3])
                                    x1 = max(x1-adjust_x1, 0)
                                    y1 = max(y1-adjust_y1, 0)
                                    x2 = min(x2+adjust_x2, videos[current_loop_video]['width'])
                                    y2 = min(y2+adjust_y2, videos[current_loop_video]['height'])
                                    color = (0, 255, 0)  # Green color (BGR format)
                                    thickness = 2  # Line thickness
                                    if show_bbox_var.get() == 1:
                                        cv2.rectangle(videos[current_loop_video]["swapped_image"], (x1,y1), (x2,y2), color, thickness)
                                    cv2.rectangle(videos[current_loop_video]["original_image"], (x1,y1), (x2,y2), color, thickness)
                        if time.time() - start > 1:
                            start = time.time()
                            if not args['nocuda'] and not args['apple']:
                                vram_usage, gpu_usage = round(gpu_memory_total - torch.cuda.mem_get_info(device)[0] / 1024**3,2), torch.cuda.utilization(device=device)
                                progressbar.set_description(f"VRAM: {vram_usage}/{gpu_memory_total} GB, usage: {gpu_usage}%")
                        
                        if not args['cli']:
                            if not args['nocuda'] and not args['apple']:
                                listik = [videos[current_loop_video]['count'], videos[current_loop_video]["frame_number"],gpu_usage, vram_usage,gpu_memory_total]
                            else:
                                listik = [videos[current_loop_video]['count'], videos[current_loop_video]["frame_number"], 0, 0, 0]
                            #videos[current_video]['swapped_image'] = videos[current_video]["swapped_image"]
                            #cv2.imshow('Face Detection', frame)
                        
                        #if not args['cli']:
                        #    if show_external_swapped_preview_var.get() == 1:
                        #        cv2.imshow('swapped frame', videos[current_loop_video]["swapped_image"])
                        if videos[current_loop_video]['rendering'] and ((videos[current_loop_video]['rendering'] and not args['cli']) or args['cli']) and xxs:
                            videos[current_loop_video]['out'].write(videos[current_loop_video]["swapped_image"])
                        
                        if args['vcam']:
                            cam.send(cv2.cvtColor(videos[current_loop_video]["swapped_image"], cv2.COLOR_RGB2BGR))

                            
                        if args['extract_output'] != '':
                            cv2.imwrite(os.path.join(args['extract_output'], os.path.basename(videos[current_loop_video]["target_path"]), f"frame_{videos[current_loop_video]['count']:05d}.png"), videos[current_loop_video]["swapped_image"])
                        if videos[current_loop_video]['rendering'] and ((videos[current_loop_video]['rendering'] and not args['cli']) or args['cli']):
                            progressbar.update(1)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    except KeyboardInterrupt:
                        break
                    #except Exception as e:
                    #    if "main thread is not in main loop" in str(e):
                    #        return
                    #    if "list index out of range" in str(e):
                    #        print('index')
                    #        break
                    #    if "'NoneType' object has no attribute 'shape'" in str(e) and isinstance(videos[current_loop_video]['target_path'], int):
                    #        videos[current_loop_video]['cap'] = cv2.VideoCapture(videos[current_loop_video]['target_path'])
                    #        videos[current_loop_video]['cap'].set(cv2.CAP_PROP_FRAME_WIDTH, globalsz.width)
                    #        videos[current_loop_video]['cap'].set(cv2.CAP_PROP_FRAME_HEIGHT, globalsz.height)
                    #    print(f"HUSTON, WE HAD AN EXCEPTION, PROCEED WITH CAUTION, SEND RICHARD THIS: {e}. Line 947")
                for i in videos[current_video]['temp']:
                    bbox, videos[current_video]["swapped_image"], videos[current_video]['original_image'] = i.join()
                    if not args['cli']:
                        if show_bbox_var.get() == 1 :
                            for i in bbox: 
                                x1, y1, x2, y2 = int(i[0]),int(i[1]),int(i[2]),int(i[3])
                                x1 = max(x1-adjust_x1, 0)
                                y1 = max(y1-adjust_y1, 0)
                                x2 = min(x2+adjust_x2, videos[current_loop_video]['width'])
                                y2 = min(y2+adjust_y2, videos[current_loop_video]['height'])
                                color = (0, 255, 0)  # Green color (BGR format)
                                thickness = 2  # Line thickness
                                cv2.rectangle(videos[current_video]["swapped_image"], (x1,y1), (x2,y2), color, thickness)
                    if time.time() - start > 1:
                        start = time.time()
                        if not args['nocuda'] and not args['apple']:
                            vram_usage, gpu_usage = round(gpu_memory_total - torch.cuda.mem_get_info(device)[0] / 1024**3,2), torch.cuda.utilization(device=device)
                            progressbar.set_description(f"VRAM: {vram_usage}/{gpu_memory_total} GB, usage: {gpu_usage}%")
                    
                    if videos[current_video]['rendering'] and ((videos[current_video]['rendering'] and not args['cli']) or args['cli']):
                        videos[current_video]['out'].write(videos[current_video]["swapped_image"])
                    if args['vcam']:
                        cam.send(cv2.cvtColor(videos[current_video]["swapped_image"], cv2.COLOR_RGB2BGR))
                    if args['extract_output'] != '':
                        cv2.imwrite(os.path.join(args['extract_output'], os.path.basename(videos[current_video]["target_path"]), f"frame_{videos[current_video]['count']:05d}.png"), videos[current_video]["swapped_image"])
                    progressbar.update(1)
                    #if not args['cli']:
                    #    if show_external_swapped_preview_var.get() == 1:
                    #        cv2.imshow('swapped frame', videos[current_video]["swapped_image"])
                    if not videos[current_video]['rendering']:
                        old_number = videos[current_video]['current_frame_index']
                        while videos[current_video]['current_frame_index'] == old_number:
                            time.sleep(0.01)
                        
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                videos[current_video]['out'].release()
                #videos[current_video]["cap"].release()
                cv2.destroyAllWindows()
                progressbar.close()
                if args['batch'] != '':
                    try:
                        add_audio_from_video(videos[current_video]["save_temp_path"],videos[current_video]["target_path"], videos[current_video]['save_path'])
                        os.remove(videos[current_video]["save_temp_path"])
                    except Exception as e:
                        print(f"SOMETHING WENT WRONG DURING THE ADDING OF THE AUDIO TO THE VIDEO!file: {videos[current_video]['save_path']}, error:{e}")
                else:
                    if not isinstance(args['target_path'], int):
                        try:
                            add_audio_from_video(videos[current_video]["save_temp_path"], videos[current_video]["target_path"], videos[current_video]['save_path'])
                            os.remove(videos[current_video]["save_temp_path"])
                        except Exception as e:
                            print(f"failed to add audio: {e}")
                videos[current_video]["cap"] = reset_cap(videos[current_video]["cap"])
                if not args['cli']:
                    not_run_it_please()
                continue
        
            except KeyboardInterrupt:
                break
            #except Exception as e:
            #    if "main thread is not in main loop" in str(e):
            #        return
            #    #if "list index out of range" in str(e):
            #    #    break
            #    print(f"HUSTON, WE HAD AN EXCEPTION, PROCEED WITH CAUTION, SEND RICHARD THIS: {e}. Line 1229")
            
            
        print("Processing finished, you may close the window now")
        if args['cli']:
        #    root.destroy()
            #del root
            #del menu
        #else:
            os._exit(0)
    ##########################
    if args['batch'] != '':
        os.makedirs(args['output'], exist_ok=True)
    globalsz.args = args
    if args['fastload']:
        wait_thread.join()
    try:
        if not args['cli']:
            listik = [0, 1, 0, 0, 0]
            threading.Thread(target=main).start()
            def update_gui(old_index=0):
                global frame_index
                try:
                    update_progress_bar(7, listik[0], listik[1], listik[2], listik[3], listik[4])
                    
                    #if args['preview']:
                    if old_index != videos[current_video]['current_frame_index']:  
                        slider.set(videos[current_video]['current_frame_index'])
                        old_index = videos[current_video]['current_frame_index']
                    try:
                        slider.config(to=videos[current_video]["frame_number"])
                        frame_count_label.config(text=f"total frames: {videos[current_video]['frame_number']}") 
                    except:
                        pass

                except:
                    pass
                root.after(300, update_gui, old_index)
            def update_selector(old_len = 0):
                global current_video
                try:
                    new_len = len(videos)
                    if old_len != new_len: 
                        #Scrolledlistbox1.delete('0','end')
                        #for i in videos:
                        #    Scrolledlistbox1.insert("end", i['target_path'])
                        Scrolledlistbox1.delete_all()
                        v = []
                        for i in videos:
                            v.append([os.path.basename(str(i['target_path'])), Image.fromarray(cv2.cvtColor(i["first_frame"], cv2.COLOR_BGR2RGB))])
                        Scrolledlistbox1.insert_data(v)
                        root.update_idletasks()
                        if len(videos) > 0:
                            videos[current_video]['old_number'] = -1
                        frame_updater(False)
                    #sel = Scrolledlistbox1.curselection()
                    #if len(sel) != 0:
                    #    sel = sel[0]
                    #else:
                    #    sel = 0
                    sel = Scrolledlistbox1.get_selected_id()
                    if sel == None:
                        sel = 0
                    if sel != current_video:
                        if videos[sel]['rendering'] == True:
                            render_button.config(state=tk.DISABLED)
                            stop_rendering_button.config(state=tk.ACTIVE)
                        else:
                            render_button.config(state=tk.ACTIVE)
                            stop_rendering_button.config(state=tk.DISABLED)
                    current_video = sel
                except:
                    pass
                try:
                    if face_selector_var.get() == 1:
                        args['selective'] = True
                    else:
                        args['selective'] = ''
                except:
                    pass
                root.after(30, update_selector, new_len)
            update_gui()
            update_selector()
            frame_updater()
            root.after(1000, toggle_menu)
            root.after(1000, toggle_clip_menu)
            
            root.mainloop()
        else:
            main()
    except Exception as e:
        print(e)
        os._exit(1)
    finally:
        if not args['cli']:
            globalsz.source_face = None
            try:
                root.destroy()
            except:
                continue
            continue
        os._exit(0)