import threading
# config
select_face_swapper_gpu = None#[0] #None for all gpus. Or make it a list, like [0, 1, 2] to select gpus to use
select_gfpgan_gpu = 0 #supports only 1 gpu for now
select_realesrgan_gpu = 0 # supports only 1 gpu for now
select_rembg_gpu = None# Leave it like that, it dies anyway
realeasrgan_model = "RealESRGAN_x2plus"
realesrgan_fp16 = False
realesrgan_outscale = 2
rembg_model = "u2net" #https://github.com/danielgatis/rembg/tree/main#models
rembg_color = [0, 255, 0, 0] #RGBA
# used by the program
lowmem = True
generator = None
restorer = None
gfpgan_onnx_model = None
realeasrgan_enhancer = None
THREAD_SEMAPHORE = threading.Semaphore()
realesrgan_lock = threading.Semaphore()
cuda = True # just debugging I think 
source_face = None
advanced_face_detector_lock = threading.Lock()
face_mesh = None
mp_face_mesh = None
rembg_models = None
rembg_lock = threading.Lock()
waitx = False