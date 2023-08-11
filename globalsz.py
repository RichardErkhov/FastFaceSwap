import threading
generator = None
lowmem = True
restorer = None
gfpgan_onnx_model = None
THREAD_SEMAPHORE = threading.Semaphore()
select_gpu = None #None for all gpus. Or make it a list, like [0, 1, 2] to select gpus to use
cuda = True #just debugging I think 