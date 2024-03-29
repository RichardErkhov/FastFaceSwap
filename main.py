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
original_frame = None
swapped_frame = None
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
    if isinstance(globalsz.source_face, NoneType):
        try:
            globalsz.source_face = sorted(face_analysers[0].get(cv2.imread(args['face'])), key=lambda x: x.bbox[0])[0]
        except Exception as e:
            print(f"HUSTON, WE HAVE A PROBLEM. WE CAN'T DETECT THE FACE IN THE IMAGE YOU PROVIDED! ERROR: {e}")
            if not args['cli']:
                show_error_custom(text = f"HUSTON, WE HAVE A PROBLEM. WE CAN'T DETECT THE FACE IN THE IMAGE YOU PROVIDED! ERROR: {e}")
                kill_ui()
    return globalsz.source_face
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
        import tkinter as tk
        from tkinter import ttk
        from tkinter.filedialog import asksaveasfilename, askdirectory, askopenfilename
        def finish(menu):
            global thread_amount_temp
            thread_amount_temp = thread_amount_input.get()
            menu.destroy()
        menu = tk.Tk()
        menu.geometry("500x500")
        menu.configure(bg=background_color)
        
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
            args['threads'] = int(thread_amount_temp)

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
        root = tk.Tk()

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
        root.geometry("1000x970")
        root.configure(bg=background_color)
        left_frame = tk.Frame(root, bg=background_color)
        left_frame.grid(row=0, column=0, rowspan=2, sticky="ns")
        
        faceswapper_checkbox_var = tk.IntVar(value=1)
        faceswapper_checkbox = ttk.Checkbutton(left_frame, text="Face swapper", variable=faceswapper_checkbox_var, style="TCheckbutton")
        faceswapper_checkbox.grid(row=0, column=0)
        
        checkbox_var = tk.IntVar()
        checkbox = ttk.Checkbutton(left_frame, text="Face enhancer", variable=checkbox_var, style="TCheckbutton")
        checkbox.grid(row=1, column=0)
        
        enhancer_choice = tk.StringVar(value='fastface enhancer')
        choices = ['fastface enhancer', 'gfpgan', 'codeformer', 'gfpgan onnx', "real esrgan"]

        if args['lowmem']:
            choices.remove('fastface enhancer')

        dropdown = ttk.OptionMenu(left_frame, enhancer_choice, enhancer_choice.get(), *choices)
        dropdown.grid(row=2, column=0)

        show_bbox_var = tk.IntVar()
        show_bbox = ttk.Checkbutton(left_frame, text="draw bounding box around faces", variable=show_bbox_var, style="TCheckbutton")
        show_bbox.grid(row=3, column=0)
        
        label = tk.Label(left_frame, text="bounding box adjustment", fg=text_color, bg=background_color)
        label.grid(row=4, column=0)
        
        label = tk.Label(left_frame, text="up", fg=text_color, bg=background_color)
        label.grid(row=5, column=0)
        
        entry_y1 = tk.Entry(left_frame)
        entry_y1.grid(row=6, column=0)
        entry_y1.insert(0, adjust_y1)
        
        label = tk.Label(left_frame, text="right", fg=text_color, bg=background_color)
        label.grid(row=7, column=0)
        
        entry_x2 = tk.Entry(left_frame)
        entry_x2.grid(row=8, column=0)
        entry_x2.insert(0, adjust_x2)
        
        label = tk.Label(left_frame, text="left", fg=text_color, bg=background_color)
        label.grid(row=9, column=0)
        
        entry_x1 = tk.Entry(left_frame)
        entry_x1.grid(row=10, column=0)
        entry_x1.insert(0, adjust_x1)
        
        label = tk.Label(left_frame, text="down", fg=text_color, bg=background_color)
        label.grid(row=11, column=0)
        
        entry_y2 = tk.Entry(left_frame)
        entry_y2.grid(row=12, column=0)
        entry_y2.insert(0, adjust_y2)
        
        button = tk.Button(left_frame, text="Set Values", bg=button_color, fg=text_color, command=set_adjust_value)
        button.grid(row=13, column=0)
        
        label = tk.Label(left_frame, text="for these settings you need codeformer to be enabled", fg=text_color, bg=background_color)
        label.grid(row=14, column=0)
        
        label = tk.Label(left_frame, text="and tick on the face enhancer", fg=text_color, bg=background_color)
        label.grid(row=15, column=0)
        
        codeformer_fidelity = 0.1
        def on_codeformer_slider_move(value):
            global codeformer_fidelity
            codeformer_fidelity = float(value)
        
        label = tk.Label(left_frame, text="Codeformer fidelity", fg=text_color, bg=background_color)
        label.grid(row=16, column=0)
        
        codeformer_slider = tk.Scale(left_frame, from_=0.1, to=2.0, resolution=0.1,  orient=tk.HORIZONTAL, fg=text_color, bg=background_color, command=on_codeformer_slider_move)
        codeformer_slider.grid(row=17, column=0)
        
        alpha = 0.0
        def alpha_slider_move(value):
            global alpha
            alpha = float(value)
        
        label = tk.Label(left_frame, text="blender", fg=text_color, bg=background_color)
        label.grid(row=18, column=0)
        
        alpha_slider = tk.Scale(left_frame, from_=0.0, to=1.0, resolution=0.1, fg=text_color, bg=background_color,  orient=tk.HORIZONTAL, command=alpha_slider_move)
        alpha_slider.grid(row=19, column=0)
        alpha_slider.set(1.0)
        
        codeformer_skip_if_no_face_var = tk.IntVar()
        codeformer_skip_if_no_face = ttk.Checkbutton(left_frame, text="Skip codeformer if not face is found", variable=codeformer_skip_if_no_face_var, style="TCheckbutton")
        codeformer_skip_if_no_face.grid(row=20, column=0)
        
        codeformer_upscale_face_var = tk.IntVar()
        codeformer_upscale_face = ttk.Checkbutton(left_frame, text="Upscale face using codeformer", variable=codeformer_upscale_face_var, style="TCheckbutton")
        codeformer_upscale_face.grid(row=21, column=0)
        codeformer_upscale_face_var.set(1)
        
        codeformer_enhance_background_var = tk.IntVar()
        codeformer_enhance_background = ttk.Checkbutton(left_frame, text="Enhance background using codeformer", variable=codeformer_enhance_background_var, style="TCheckbutton")
        codeformer_enhance_background.grid(row=22, column=0)
        
        codeformer_upscale_amount_value = 1
        def codeformer_upscale_amount_move(value):
            global codeformer_upscale_amount_value
            codeformer_upscale_amount_value = int(value)
        
        codeformer_upscale_amount = tk.Scale(left_frame, from_=1, to=3, resolution=1, fg=text_color, bg=background_color, orient=tk.HORIZONTAL, command=codeformer_upscale_amount_move)
        codeformer_upscale_amount.grid(row=23, column=0)
        codeformer_upscale_amount.set(1)
        
        label = tk.Label(left_frame, text="codeformer settings finished", fg=text_color, bg=background_color)
        label.grid(row=24, column=0)
        
        if not isinstance(args['target_path'], int):
            progress_label = tk.Label(left_frame, fg=text_color, bg=background_color)
            progress_label.grid(row=25, column=0)
        
        usage_label1 = tk.Label(left_frame, fg=text_color, bg=background_color)
        usage_label1.grid(row=26, column=0)
        
        if not args['nocuda'] and not args['apple']:
            usage_label2 = tk.Label(left_frame, fg=text_color, bg=background_color)
            usage_label2.grid(row=27, column=0)
        
        #if args['preview']:
        def on_slider_move(value):
            global frame_index
            frame_index = int(value)
        
        def edit_index(amount):
            global frame_index
            frame_index += amount
            slider.set(frame_index)
        
        
        def edit_play(amount):
            global frame_move
            frame_move = amount
            #slider.set(frame_move)
        frame_amount = count_frames(args['target_path'])
        label = tk.Label(left_frame, text="frame number", fg=text_color, bg=background_color)
        label.grid(row=28, column=0)
        
        slider = tk.Scale(left_frame, from_=1, to=frame_amount, fg=text_color, bg=background_color, orient=tk.HORIZONTAL, command=on_slider_move)
        slider.grid(row=29, column=0, sticky="ew")
        
        frame_count_label = tk.Label(left_frame, text=f"total frames: {frame_amount}", fg=text_color, bg=background_color)
        frame_count_label.grid(row=30, column=0, sticky="ew")
        
        button_width = left_frame.winfo_width() // 2
        
        label = tk.Label(left_frame, text = "frame back, frame forward, backplay, pause, play", fg=text_color, bg=background_color)
        label.grid(row=31, column=0, sticky="ew")
        
        button_frame = tk.Frame(left_frame, bg=background_color)
        button_frame.grid(row=32, column=0, pady=10, sticky="ew")
        
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
            global runnable, frame_index, count
            runnable = 0
            frame_index = 0
            count = -1
            render_button.config(state=tk.DISABLED)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        show_external_swapped_preview_var = tk.IntVar()
        show_external_swapped_preview = ttk.Checkbutton(left_frame, text="Show swapped face in another window", variable=show_external_swapped_preview_var, style="TCheckbutton")
        show_external_swapped_preview.grid(row=33, column=0)
        show_external_swapped_preview_var.set(0)
        render_button = tk.Button(left_frame, text='render', bg=button_color, fg=text_color, command=run_it_please)
        render_button.grid(row=34, column=0)
        face_selector_var = tk.IntVar()
        face_selector_check = ttk.Checkbutton(left_frame, text="Face selector mode", variable=face_selector_var, style="TCheckbutton")
        face_selector_check.grid(row=35, column=0)
        face_selector_var.set(0)
        def unselect_face():
            global target_embedding, old_index, args
            args['selective'] = ''
            target_embedding = None
            old_index = -1
        unselect_face_button = tk.Button(left_frame, text='unselect the face button', bg=button_color, fg=text_color, command=unselect_face)
        unselect_face_button.grid(row=36, column=0)
        
        right_frame1 = tk.Frame(root, bg=background_color, highlightthickness=2, highlightbackground=border_color)
        right_frame2 = tk.Frame(root, bg=background_color, highlightthickness=2, highlightbackground=border_color)
        original_image_label = tk.Label(right_frame1, text="Image 1 Placeholder")
        swapped_image_label = tk.Label(right_frame2, text="Image 2 Placeholder")

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
            original_image_label.grid(sticky="nsew", padx=15, pady=15)

            right_frame2.grid(row=1, column=1, columnspan=2, sticky="nsew")
            swapped_image_label.grid(sticky="nsew", padx=15, pady=15)
            
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
                print(relative_x, relative_y)
                bboxes = []
                faces = face_analysers[0].get(original_frame)
                for face in faces:    
                    bboxes.append(face.bbox)
                height, width = original_frame.shape[:2]
                real_x = height*relative_y
                real_y = width*relative_x
                for bbox in bboxes:
                    if real_y > bbox[0] and real_y < bbox[2] and real_x > bbox[1] and real_x < bbox[3]:
                        old_index = -1
                        this_face = original_frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                        args['selective'] = True
                        target_embedding = get_embedding(this_face)[0]
                        cv2.imshow("cropped face", this_face)
                        cv2.waitKey(0)
                        try:
                            cv2.destroyWindow("cropped face")
                        except:
                            break
                        break
        original_image_label.bind("<Button-1>", on_image_click)
        #yes, one day Im going to release that thing
        '''right_control_frame = tk.Frame(root, bg=background_color)
        right_control_frame.grid(row=0, column=3, rowspan=2, sticky="ns")
        button_start_program = tk.Button(right_control_frame, text="Add this video",bg=button_color, fg=text_color, command=lambda: finish(menu))
        button_start_program.pack()
        select_face_label = tk.Label(right_control_frame, text=f'Face filename: {args["face"]}', fg=text_color, bg=background_color)
        select_face_label.pack()
        button_select_face = tk.Button(right_control_frame, text='Select face',bg=button_color, fg=text_color, command=select_face)
        button_select_face.pack()
        select_target_label = tk.Label(right_control_frame, text=f'Target filename: {args["target_path"]}', fg=text_color, bg=background_color)
        select_target_label.pack()
        button_select_target = tk.Button(right_control_frame, text='Select target',bg=button_color, fg=text_color, command=select_target)
        button_select_target.pack()
        button_select_camera = tk.Button(right_control_frame, text='run from camera',bg=button_color, fg=text_color, command=select_camera)
        button_select_camera.pack()
        select_output_label = tk.Label(right_control_frame, text=f'output filename: {args["output"]}', fg=text_color, bg=background_color)
        select_output_label.pack()
        button_select_output = tk.Button(right_control_frame, text='Select output',bg=button_color, fg=text_color, command=select_output)
        button_select_output.pack()
        thread_amount_label = tk.Label(right_control_frame, text='Select the number of threads', fg=text_color, bg=background_color)
        thread_amount_label.pack()
        thread_amount_input = tk.Entry(right_control_frame)
        thread_amount_input.pack()'''

        def update_progress_bar(length, progress, total, gpu_usage, vram_usage, total_vram):
            try:
                if not runnable and not isinstance(args['target_path'], int):
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
        global alpha, codeformer
        original_frame = frame
        if not args['cli']:
            test1 = alpha != 0
        else:
            test1 = args['alpha'] != 0
        if test1:        
            faces = face_analysers[sw].get(frame)
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
                    frame = face_swappers[sw].get(frame, face, get_source_face(), paste_back=True)
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
                        x2 = min(x2+adjust_x2, int(width))
                        y2 = min(y2+adjust_y2, int(height))
                        facer = frame[y1:y2, x1:x2]
                        if not args['cli']:
                            if enhancer_choice.get() == "fastface enhancer":
                                facex = upscale_image(facer, load_generator())
                            elif enhancer_choice.get() == "gfpgan":
                                facex = restorer_enhance(facer)
                            elif enhancer_choice.get() == "gfpgan onnx":
                                facex, _ = load_gfpganonnx().forward(facer)
                            elif enhancer_choice.get() == "real esrgan":
                                facex = realesrgan_enhance(facer)
                        else:
                            if args['face_enhancer'] == 'gfpgan':
                                facex = restorer_enhance(facer)
                            elif args['face_enhancer'] == 'ffe' and not args['lowmem']:
                                facex = upscale_image(facer, load_generator())
                            elif args['face_enhancer'] == "gpfgan_onnx":
                                facex, _ = load_gfpganonnx().forward(facer)
                            elif args['face_enhancer'] == "real_esrgan":
                                facex = realesrgan_enhance(facer)
                        facex = cv2.resize(facex, ((x2-x1), (y2-y1)))
                        frame[y1:y2, x1:x2] = facex
                    except Exception as e:
                        print(e)
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
                test1 = alpha != 1
            else:
                test1 = args['alpha'] != 1
            if test1:
                print(alpha)
                frame = merge_face(frame, original_frame, alpha)
            return bboxes, frame, original_frame
        return [], frame, original_frame

    def cv2_image_to_tkinter(cv2_image, target_width, target_height):
        """Convert a cv2 image to a tkinter compatible format and resize it to fit target dimensions."""
        cv2_img_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cv2_img_rgb)

        target_width -= 30
        target_height -= 30
        
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
    def frame_updater():
        try:
            if not isinstance(original_frame, NoneType) and not isinstance(swapped_frame, NoneType):
                #original_frame,swapped_frame
                    sizex1, sizey1 = right_frame1.winfo_width(), right_frame1.winfo_height()
                    sizex2, sizey2 = right_frame2.winfo_width(), right_frame2.winfo_height()
                    tk_image = cv2_image_to_tkinter(original_frame, sizex1, sizey1)
                    original_image_label.configure(image=tk_image)
                    original_image_label.image = tk_image  # Keep a reference to prevent garbage collection
                    tk_image = cv2_image_to_tkinter(swapped_frame, sizex2, sizey2)
                    swapped_image_label.configure(image=tk_image)
                    swapped_image_label.image = tk_image
            else:
                    original_image_label.configure(image=None)
                    original_image_label.image = None  # Keep a reference to prevent garbage collection
                    swapped_image_label.configure(image=None)
                    swapped_image_label.image = None

        except:
            pass
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
            global runnable
            runnable = 0
            second_window.destroy()
        second_window = tk.Toplevel(root)
        label = tk.Label(second_window, text='press the button to start rendering')
        label.pack()
        button = tk.Button(second_window, text='Start', command=run_it_please)
        button.pack()

    def main():
        global old_index, runnable, args, width, height, frame_index, face_analysers,frame_move, face_swappers, source_face, progress_var, target_embedding, count, frame_number, listik, frame, original_frame,swapped_frame, cap
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
        play = 0
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
        if args['batch'] == '':
            caps.append(create_cap())
        else:
            for file in os.listdir(args['target_path']):
                if is_video_file(file):
                    caps.append(create_batch_cap(file))
        #if args['fastload']:
        #    source_face_thread.join()
        #print(time.time()-start)
        if args['fastload'] and args['cli']:
            tx.join()
        elif args['fastload'] and not args['cli']:
            for t in tx:
                t.join()

        runnable = not int(args['cli'])
        #if not args['cli'] and not args['preview']:
        #    open_second_window()
        for cap, fps, width, height, out, name, file, frame_number in caps:
            #root.after(1, update_progress_length, frame_number)
            #update_progress_bar( 10, 0, frame_number)
            count = -1
            frame_index = count
            if args['vcam']:
                cam = pyvirtualcam.Camera(width=width, height=height, fps=fps)
            with tqdm(total=frame_number) as progressbar:
                temp = []
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
                old_index = 0
                while True:
                    try:
                        if runnable == 0 and ((not runnable and not args['cli']) or args['cli']):
                            count += 1
                            #if not isinstance(args['target_path'], int):
                            frame_index = count
                            if count == 0:
                                progressbar.reset()
                            if args['experimental']:
                                frame = cap.read()
                                if isinstance(frame, NoneType): #== None:
                                    break
                            else:
                                ret, frame = cap.read()
                                if not ret:
                                    break
                            temp.append(ThreadWithReturnValue(target=face_analyser_thread, args=(frame,count%len(face_swappers))))
                            temp[-1].start()
                            if count % 1000 == 999:
                                torch.cuda.empty_cache()
                            if len(temp) < int(args['threads']) * len(face_swappers) and ret:
                                continue
                            while len(temp) >= int(args['threads']) * len(face_swappers):
                                bbox, frame, original_frame = temp.pop(0).join()
                            xxs = True
                        else:
                            if not frame_index == old_index:
                                bbox, frame, original_frame = face_analyser_thread(get_nth_frame(cap, frame_index-1), count%len(face_swappers))
                            xxs = False
                        if not args['cli']:
                            if show_bbox_var.get() == 1:
                                for i in bbox: 
                                    x1, y1, x2, y2 = int(i[0]),int(i[1]),int(i[2]),int(i[3])
                                    x1 = max(x1-adjust_x1, 0)
                                    y1 = max(y1-adjust_y1, 0)
                                    x2 = min(x2+adjust_x2, int(width))
                                    y2 = min(y2+adjust_y2, int(height))
                                    color = (0, 255, 0)  # Green color (BGR format)
                                    thickness = 2  # Line thickness
                                    cv2.rectangle(frame, (x1,y1), (x2,y2), color, thickness)
                        if time.time() - start > 1:
                            start = time.time()
                            if not args['nocuda'] and not args['apple']:
                                vram_usage, gpu_usage = round(gpu_memory_total - torch.cuda.mem_get_info(device)[0] / 1024**3,2), torch.cuda.utilization(device=device)
                                progressbar.set_description(f"VRAM: {vram_usage}/{gpu_memory_total} GB, usage: {gpu_usage}%")
                        
                        if not args['cli']:
                            if not args['nocuda'] and not args['apple']:
                                listik = [count, frame_number,gpu_usage, vram_usage,gpu_memory_total]
                            else:
                                listik = [count, frame_number, 0, 0, 0]
                            swapped_frame = frame
                            #cv2.imshow('Face Detection', frame)
                        
                        if not args['cli']:
                            if show_external_swapped_preview_var.get() == 1:
                                cv2.imshow('swapped frame', frame)
                        if runnable == 0 and ((not runnable and not args['cli']) or args['cli']) and xxs:
                            out.write(frame)
                        
                        if args['vcam']:
                            cam.send(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        if runnable:
                            
                            if not isinstance(args['target_path'], int):
                                frame_index += frame_move
                                if frame_index < 1:
                                    frame_index = 1
                                elif frame_index > frame_number:
                                    frame_index = frame_number
                                old_index = frame_index
                        if args['extract_output'] != '':
                            cv2.imwrite(os.path.join(args['extract_output'], os.path.basename(file), f"frame_{count:05d}.png"), frame)
                        if runnable == 0 and ((not runnable and not args['cli']) or args['cli']):
                            progressbar.update(1)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    except KeyboardInterrupt:
                        break
                    except Exception as e:
                        if "main thread is not in main loop" in str(e):
                            return
                        print(f"HUSTON, WE HAD AN EXCEPTION, PROCEED WITH CAUTION, SEND RICHARD THIS: {e}. Line 947")
                for i in temp:
                    bbox, frame, original_frame = i.join()
                    if not args['cli']:
                        if show_bbox_var.get() == 1:
                            for i in bbox: 
                                x1, y1, x2, y2 = int(i[0]),int(i[1]),int(i[2]),int(i[3])
                                x1 = max(x1-adjust_x1, 0)
                                y1 = max(y1-adjust_y1, 0)
                                x2 = min(x2+adjust_x2, int(width))
                                y2 = min(y2+adjust_y2, int(height))
                                color = (0, 255, 0)  # Green color (BGR format)
                                thickness = 2  # Line thickness
                                cv2.rectangle(frame, (x1,y1), (x2,y2), color, thickness)
                    if time.time() - start > 1:
                        start = time.time()
                        if not args['nocuda'] and not args['apple']:
                            vram_usage, gpu_usage = round(gpu_memory_total - torch.cuda.mem_get_info(device)[0] / 1024**3,2), torch.cuda.utilization(device=device)
                            progressbar.set_description(f"VRAM: {vram_usage}/{gpu_memory_total} GB, usage: {gpu_usage}%")
                    
                    if runnable == 0 and ((not runnable and not args['cli']) or args['cli']):
                        out.write(frame)
                    if args['vcam']:
                        cam.send(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    if args['extract_output'] != '':
                        cv2.imwrite(os.path.join(args['extract_output'], os.path.basename(file), f"frame_{count:05d}.png"), frame)
                    progressbar.update(1)
                    if not args['cli']:
                        if show_external_swapped_preview_var.get() == 1:
                            cv2.imshow('swapped frame', frame)
                    #if not args['cli']:
                        #cv2.imshow('Face Detection', frame)
                        #update_progress_bar(10, count, frame_number)
                    if runnable:
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
                    try:
                        add_audio_from_video(name, args['target_path'], args['output'])
                        os.remove(name)
                    except Exception as e:
                        print(f"failed to add audio: {e}")
            
            
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
                    #if not runnable:
                    update_progress_bar(7, listik[0], listik[1], listik[2], listik[3], listik[4])
                    
                    #if args['preview']:
                    if old_index != frame_index:  
                        slider.set(frame_index)
                        old_index = frame_index

                except:
                    pass
                root.after(300, update_gui, old_index)
            update_gui()
            frame_updater()
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
