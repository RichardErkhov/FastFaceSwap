import cv2
import onnxruntime as rt
import insightface
import tqdm
import threading
from threading import Thread
import os
import numpy as np
from numpy.linalg import norm as l2norm
import torch
from swapperfp16 import get_model
threads_per_gpu = 12
if torch.cuda.is_available():
    num_devices = torch.cuda.device_count()
    print(f"Number of available CUDA devices: {num_devices}")

    for i in range(num_devices):
        device_name = torch.cuda.get_device_name(i)
        print(f"Device {i}: {device_name}")

os.environ['OMP_NUM_THREADS'] = '1'
class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return
cap = cv2.VideoCapture("banana.mp4")
swappers = []
analysers = []
#providers = rt.get_available_providers()
for idx in range(num_devices):
    providers = [
        ('CUDAExecutionProvider', {
            'device_id': idx,
        'gpu_mem_limit': 8 * 1024 * 1024 * 1024,
        'gpu_external_alloc': 0,
        'gpu_external_free': 0,
        'gpu_external_empty_cache': 1,
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'cudnn_conv1d_pad_to_nc1d': 1,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'do_copy_in_default_stream': 1,
        'enable_cuda_graph': 0,
        'cudnn_conv_use_max_workspace': 1,
        'tunable_op_enable': 1,
        'enable_skip_layer_norm_strict_mode': 1,
        'tunable_op_tuning_enable': 1
        }),
        'CPUExecutionProvider',
    ]
    sess_options = rt.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL#rt.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
    sess_options.execution_order = rt.ExecutionOrder.PRIORITY_BASED
    #swappers.append(insightface.model_zoo.get_model("inswapper_128.onnx", session_options=sess_options, providers=providers))
    swappers.append(get_model("inswapper_128.fp16.onnx", session_options=sess_options, providers=providers))
    analysers.append(insightface.app.FaceAnalysis(name='buffalo_l', providers=providers, session_options=sess_options))
    analysers[idx].prepare(ctx_id=0, det_size=(256, 256))
'''#providers = rt.get_available_providers()
providers = [
        ('CUDAExecutionProvider', {
            'device_id': idx,
        }),
        #'CPUExecutionProvider',
    ]'''
    
input_face = cv2.imread("face.jpg")
source_face = sorted(analysers[0].get(input_face), key=lambda x: x.bbox[0])[0]
def process(frame, sw):
    faces = analysers[sw].get(frame)
    bboxes = []
    for face in faces:
        bboxes.append(face.bbox)
        frame = swappers[sw].get(frame, face, source_face, paste_back=True)
    return frame
temp = []
pbar = tqdm.tqdm()
current_frame = 0
while True:
    ret, frame = cap.read()
    if len(temp) < threads_per_gpu*len(swappers) and ret:
        t = ThreadWithReturnValue(target=process, args=(frame,current_frame%len(swappers)))
        t.start()
        temp.append(t)
        continue
    if len(temp) >= threads_per_gpu*len(swappers) or not ret:
        frame = temp.pop(0).join()
    #cv2.imshow('Camera', frame)
    pbar.update(1)
    current_frame += 1
    if not ret and len(temp) == 0:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()