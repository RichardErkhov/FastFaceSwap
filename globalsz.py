import threading
# config
select_face_swapper_gpu = [0] #None for all gpus. Or make it a list, like [0, 1, 2] to select gpus to use
select_gfpgan_gpu = 0 #supports only 1 gpu for now
select_realesrgan_gpu = 0 # supports only 1 gpu for now
realeasrgan_model = "RealESRGAN_x2plus"
realesrgan_fp16 = False
realesrgan_outscale = 2


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