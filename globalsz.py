import threading
generator = None
lowmem = True
restorer = None
gfpgan_onnx_model = None
THREAD_SEMAPHORE = threading.Semaphore()