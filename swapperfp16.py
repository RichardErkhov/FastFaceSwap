import time
import numpy as np
import onnxruntime
import cv2
import onnx
from onnx import numpy_helper
from insightface.utils import face_align
from numpy.linalg import norm as l2norm
import tqdm
import requests
import os
from skimage import transform as trans

from models.clipseg import CLIPDensePredT
from torchvision import transforms
import torch
import segmentation_models_pytorch as smp
from collections import OrderedDict
import math
from insightface.utils.face_align import norm_crop2
from math import floor, ceil
import threading
lock=threading.Lock()
def load_clip_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")    
    clip_session = CLIPDensePredT(version='ViT-B/16', reduce_dim=64, complex_trans_conv=True)
    clip_session.eval()
    clip_session.load_state_dict(torch.load('weights/rd64-uni-refined.pth', map_location=torch.device('cuda')), strict=False)
    clip_session.to(device)    
    return clip_session, device
def load_occluder_model():            
    
    exists = os.path.exists('weights/occluder.ckpt')
    if not exists:
        os.makedirs('weights', exist_ok=True)
        download(f"https://github.com/RichardErkhov/FastFaceSwap/releases/download/model/occluder.ckpt", 'weights/occluder.ckpt')
    to_tensor = transforms.ToTensor()
    model = smp.Unet(encoder_name='resnet18', encoder_weights='imagenet', classes=1, activation=None)

    weights = torch.load('weights/occluder.ckpt')
    new_weights = OrderedDict()
    for key in weights.keys():
        new_key = '.'.join(key.split('.')[1:])
        new_weights[new_key] = weights[key]

    model.load_state_dict(new_weights)
    model.to('cuda')
    model.eval()
    return model, to_tensor

def create_M2_from_M1(M1, scale_factor):
    M2 = np.copy(M1)
    M2[:, :2] *= scale_factor  # Scale the rotation and scaling part
    M2[:, 2] *= scale_factor  # Scale the translation part
    return M2
class INSwapper():
    def __init__(self, model_file=None, session=None):
        self.model_file = model_file
        self.session = session
        model = onnx.load(self.model_file)
        graph = model.graph
        self.emap = numpy_helper.to_array(graph.initializer[-1])
        self.input_mean = 0.0
        self.input_std = 255.0
        #print('input mean and std:', model_file, self.input_mean, self.input_std)
        if self.session is None:
            self.session = onnxruntime.InferenceSession(self.model_file, None)
        inputs = self.session.get_inputs()
        self.input_names = []
        for inp in inputs:
            self.input_names.append(inp.name)
        outputs = self.session.get_outputs()
        output_names = []
        for out in outputs:
            output_names.append(out.name)
        self.output_names = output_names
        assert len(self.output_names)==1
        output_shape = outputs[0].shape
        input_cfg = inputs[0]
        input_shape = input_cfg.shape
        self.input_shape = input_shape
        self.occluder_model = None
        self.clip_session = None
        print('inswapper-shape:', self.input_shape)
        self.input_size = tuple(input_shape[2:4][::-1])
        self.init_occluder = False
        self.occluder_blur = 25
        self.arcface_dst = np.array(
        [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
         [41.5493, 92.3655], [70.7299, 92.2041]],
        dtype=np.float32)  
        self.mask_top = 0
        self.mask_bottom = 0
        self.mask_left = 0
        self.mask_right = 0
        self.mask_blur = 0
        self.occluder = True
        self.toggle_CLIPs = False
        self.fake_diff_state = False
        self.GFPGAN_state = False
        self.fake_diff_blend = 0
        #for clip
        self.pos_thresh = 0.5
        self.neg_thresh = 0.5
        self.CLIPs = ["", ""]
        self.init_clip = False
        self.occluder_works = False
        self.CLIP_blur = 5
    def load_occluder(self):
        self.init_occluder = True
        self.occluder_model, self.occluder_tensor = load_occluder_model()
    def load_clip(self):
        self.init_clip = True
        self.clip_session, self.cuda_device = load_clip_model()

    def forward(self, img, latent):
        img = (img - self.input_mean) / self.input_std
        pred = self.session.run(self.output_names, {self.input_names[0]: img, self.input_names[1]: latent})[0]
        return pred
    def get(self, img, target_face, source_face, occluder_works=False, clip_works=False, prompts=["", ""], paste_back=True):
        if clip_works:
            self.CLIPs = prompts
        self.toggle_CLIPs = clip_works
        self.occluder_works = occluder_works
        if not self.init_occluder and occluder_works:
            self.load_occluder()
        if not self.init_clip and clip_works:
            self.load_clip()
        if occluder_works or clip_works:
            return self.get_rope(img, target_face, source_face, paste_back)  
        
        else:
            return self.get_old(img, target_face, source_face, paste_back)
        
    def get_rope(self, img, target_face, source_face, paste_back=True):
        kps = target_face.kps
        s_e = source_face.normed_embedding
        bbox = target_face.bbox
        aimg, _ = norm_crop2(img, kps, self.input_size[0])
        blob = cv2.dnn.blobFromImage(aimg, 1.0 / 255.0, self.input_size, (0.0, 0.0, 0.0), swapRB=True)

       #Select source embedding
        n_e = s_e / l2norm(s_e)
        latent = n_e.reshape((1,-1))
        latent = np.dot(latent, self.emap)
        latent /= np.linalg.norm(latent)
        
        '''if self.io_binding:
                            
            io_binding = self.swapper_model.io_binding()            
            io_binding.bind_cpu_input(self.input_names[0], blob)
            io_binding.bind_cpu_input(self.input_names[1], latent)
            io_binding.bind_output(self.output_names[0], "cuda")
               
            self.swapper_model.run_with_iobinding(io_binding)
            ort_outs = io_binding.copy_outputs_to_cpu()
            pred = ort_outs[0]'''
        
        #else:
        pred = self.session.run(self.output_names, {self.input_names[0]: blob, self.input_names[1]: latent})[0]
        #print(latent.shape, latent.dtype, pred.shape)
        img_fake = pred.transpose((0,2,3,1))[0]
        bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:,:,::-1]


        target_img = img

        ratio = 4.0
        diff_x = 8.0*ratio

        dst = self.arcface_dst * ratio
        dst[:,0] += diff_x
        tform = trans.SimilarityTransform()
        tform.estimate(kps, dst)
        M1 = tform.params[0:2, :]
        IM = cv2.invertAffineTransform(M1)
        
        ratio = 2.0
        diff_x = 8.0*ratio

        dst = self.arcface_dst * ratio
        dst[:,0] += diff_x
        tform = trans.SimilarityTransform()
        tform.estimate(kps, dst)
        M2 = tform.params[0:2, :]


        bgr_fake_upscaled = cv2.resize(bgr_fake, (512,512))
        

        img_white = np.full((bgr_fake_upscaled.shape[0],bgr_fake_upscaled.shape[1]), 255, dtype=np.float32)
        img_black = np.full((bgr_fake_upscaled.shape[0],bgr_fake_upscaled.shape[1]), 0, dtype=np.float32)

        img_white[img_white>20] = 255
        img_mask = img_black
        mask_border = 5
        img_mask = cv2.rectangle(img_mask, (mask_border+int(self.mask_left), mask_border+int(self.mask_top)), 
                                (512 - mask_border-int(self.mask_right), 512-mask_border-int(self.mask_bottom)), (255, 255, 255), -1)    
        img_mask = cv2.GaussianBlur(img_mask, (self.mask_blur*2+1,self.mask_blur*2+1), 0)    
        img_mask /= 255
        
        # Occluder
        if self.occluder and self.occluder_works:

            input_image = cv2.warpAffine(target_img, M2, (256, 256), borderValue=0.0)

            data = self.occluder_tensor(input_image).unsqueeze(0)

            data = data.to('cuda')
            with lock:
                with torch.no_grad():
                    pred = self.occluder_model(data)

            occlude_mask = (pred > 0).type(torch.int8)

            occlude_mask = occlude_mask.squeeze().cpu().numpy()*1.0



            occlude_mask = cv2.GaussianBlur(occlude_mask*255, (self.occluder_blur*2+1,self.occluder_blur*2+1), 0)
            
            occlude_mask = cv2.resize(occlude_mask, (512,512))
            
            occlude_mask /= 255
            img_mask *= occlude_mask 

            
       
        if self.toggle_CLIPs:
            clip_mask = img_white
            input_image = cv2.warpAffine(target_img, M1, (512, 512), borderValue=0.0)

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Resize((512, 512)),
            ])
            img = transform(input_image).unsqueeze(0)

            if self.CLIPs[0] != "":
                prompts = self.CLIPs[0].split(',')
                
                lock.acquire()
                with torch.no_grad():
                    preds = self.clip_session(img.repeat(len(prompts),1,1,1), prompts)[0]
                lock.release()
                clip_mask = torch.sigmoid(preds[0][0])
                for i in range(len(prompts)-1):
                    clip_mask += torch.sigmoid(preds[i+1][0])
                clip_mask = clip_mask.data.cpu().numpy()
                np.clip(clip_mask, 0, 1)
                
                clip_mask[clip_mask>self.pos_thresh] = 1.0
                clip_mask[clip_mask<=self.pos_thresh] = 0.0
                kernel = np.ones((5, 5), np.float32)
                clip_mask = cv2.dilate(clip_mask, kernel, iterations=1)
                clip_mask = cv2.GaussianBlur(clip_mask, (self.CLIP_blur*2+1,self.CLIP_blur*2+1), 0)
                
                img_mask *= clip_mask
            
            
            if self.CLIPs[1] != "":
                prompts = self.CLIPs[1].split(',')
                
                lock.acquire()
                with torch.no_grad():
                    preds = self.clip_session(img.repeat(len(prompts),1,1,1), prompts)[0]
                lock.release()
                neg_clip_mask = torch.sigmoid(preds[0][0])
                for i in range(len(prompts)-1):
                    neg_clip_mask += torch.sigmoid(preds[i+1][0])
                neg_clip_mask = neg_clip_mask.data.cpu().numpy()
                np.clip(neg_clip_mask, 0, 1)

                neg_clip_mask[neg_clip_mask>self.neg_thresh] = 1.0
                neg_clip_mask[neg_clip_mask<=self.neg_thresh] = 0.0
                kernel = np.ones((5, 5), np.float32)
                neg_clip_mask = cv2.dilate(neg_clip_mask, kernel, iterations=1)
                neg_clip_mask = cv2.GaussianBlur(neg_clip_mask, (self.CLIP_blur*2+1,self.CLIP_blur*2+1), 0) 
            
                img_mask -= neg_clip_mask
                # np.clip(img_mask, 0, 1)
                img_mask[img_mask<0.0] = 0.0

        if not self.fake_diff_state:
            img_mask_0 = np.reshape(img_mask, [img_mask.shape[0],img_mask.shape[1],1])
            
            img_mask = cv2.warpAffine(img_mask, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
        else:
            fake_diff = bgr_fake.astype(np.float32) - aimg.astype(np.float32)
            fake_diff = np.abs(fake_diff).mean(axis=2)
            fake_diff[:2,:] = 0
            fake_diff[-2:,:] = 0
            fake_diff[:,:2] = 0
            fake_diff[:,-2:] = 0
            fake_diff = cv2.resize(fake_diff, (512,512))
            
            fthresh = int(self.fake_diff_blend)
            fake_diff[fake_diff<fthresh] = 0
            fake_diff[fake_diff>=fthresh] = 255 

            fake_diff = cv2.GaussianBlur(fake_diff, (15*2+1,15*2+1), 0)

            fake_diff /= 255
            
            img_mask_1 = np.reshape(img_mask, [img_mask.shape[0],img_mask.shape[1],1])
            img_mask_0 = np.reshape(fake_diff, [fake_diff.shape[0],fake_diff.shape[1],1]) 
            
            img_mask_0 *= img_mask_1
            
            img_mask = cv2.warpAffine(fake_diff, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)

        if self.GFPGAN_state:  

            temp = bgr_fake_upscaled
            # height, width = temp.shape[0], temp.shape[1]

            # preprocess
            temp = cv2.resize(temp, (512, 512))
            temp = temp / 255.0
            temp = temp.astype('float32')
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
            temp[:,:,0] = (temp[:,:,0]-0.5)/0.5
            temp[:,:,1] = (temp[:,:,1]-0.5)/0.5
            temp[:,:,2] = (temp[:,:,2]-0.5)/0.5
            temp = np.float32(temp[np.newaxis,:,:,:])
            temp = temp.transpose(0, 3, 1, 2)

            ort_inputs = {"input": temp}
            if self.io_binding:
                
                
                io_binding = self.GFPGAN_model.io_binding()            
                io_binding.bind_cpu_input("input", temp)
                io_binding.bind_output("1288", "cuda")
                   
                self.GFPGAN_model.run_with_iobinding(io_binding)
                ort_outs = io_binding.copy_outputs_to_cpu()
            else:
                
                ort_outs = self.GFPGAN_model.run(None, ort_inputs)
            
            output = ort_outs[0][0]

            # postprocess
            output = output.clip(-1,1)
            output = (output + 1) / 2
            output = output.transpose(1, 2, 0)
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            output = (output * 255.0).round()


            inv_soft_mask = np.ones((512, 512, 1), dtype=np.float32)
            output = cv2.resize(output, (512, 512))
     
            output = output.astype(np.uint8)

            temp2 = float(self.GFPGAN_blend)/100.0
            bgr_fake_upscaled = cv2.addWeighted(output, temp2, bgr_fake_upscaled, 1.0-temp2,0)
            #crop = cv2.cvtColor(bgr_fake_upscaled, cv2.COLOR_RGB2BGR) 
            #cv2.imwrite("test.jpg", crop)
        
        fake_merged = img_mask_0* bgr_fake_upscaled
        
        img_mask = np.reshape(img_mask, [img_mask.shape[0],img_mask.shape[1],1])    
        fake_merged = cv2.warpAffine(fake_merged, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0) 
       
        diff_hor = (bbox[2]-bbox[0])*0.3
        diff_vert = (bbox[3]-bbox[1])*0.3
        
        left = floor(bbox[0]-diff_hor)
        if left<0:
            left=0
        top = floor(bbox[1]-diff_vert)
        if top<0: 
            top=0
        right = ceil(bbox[2]+diff_hor)
        if right>target_img.shape[1]:
            right=target_img.shape[1]
        
        bottom = ceil(bbox[3]+diff_vert)
        if bottom>target_img.shape[0]:
            bottom=target_img.shape[0]
        
        fake_merged = fake_merged[top:bottom, left:right, 0:3]
        target_img_a = target_img[top:bottom, left:right, 0:3]
        img_mask = img_mask[top:bottom, left:right, 0:1]
        
        fake_merged = fake_merged + (1-img_mask) * target_img_a.astype(np.float32)
        
        target_img[top:bottom, left:right, 0:3] = fake_merged
        
        fake_merged = target_img.astype(np.uint8)   
        
        # fake_merged = fake_merged + (1-img_mask) * target_img.astype(np.float32)
        # fake_merged = fake_merged.astype(np.uint8)   
            
        return fake_merged    #BGR
    def get_old(self, img, target_face, source_face, paste_back=True):
        print("old")
        aimg, M = face_align.norm_crop2(img, target_face.kps, self.input_size[0])
        blob = cv2.dnn.blobFromImage(aimg, 1.0 / self.input_std, self.input_size,
                                      (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        s_e = source_face.normed_embedding
        n_e = s_e / l2norm(s_e)
        latent = n_e.reshape((1,-1))
        
        latent = np.dot(latent, self.emap)
        latent /= np.linalg.norm(latent)
        pred = self.session.run(self.output_names, {self.input_names[0]: blob, self.input_names[1]: latent})[0]
        #print(latent.shape, latent.dtype, pred.shape)
        img_fake = pred.transpose((0,2,3,1))[0]
        #print("Minimum value:", np.min(img_fake))
        #print("Maximum value:", np.max(img_fake))
        bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:,:,::-1]
        if not paste_back:
            return bgr_fake, M
        else:
            target_img = img
            fake_diff = bgr_fake.astype(np.float32) - aimg.astype(np.float32)
            fake_diff = np.abs(fake_diff).mean(axis=2)
            fake_diff[:2,:] = 0
            fake_diff[-2:,:] = 0
            fake_diff[:,:2] = 0
            fake_diff[:,-2:] = 0
            IM = cv2.invertAffineTransform(M)
            img_mask = np.full((aimg.shape[0],aimg.shape[1]), 255, dtype=np.float32)
            bgr_fake = cv2.warpAffine(bgr_fake, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
            img_mask = cv2.warpAffine(img_mask, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
            fake_diff = cv2.warpAffine(fake_diff, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
            img_mask[img_mask>20] = 255
            fthresh = 10
            fake_diff[fake_diff<fthresh] = 0
            fake_diff[fake_diff>=fthresh] = 255
            #img_mask = img_white
            mask_h_inds, mask_w_inds = np.where(img_mask==255)
            mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
            mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
            mask_size = int(np.sqrt(mask_h*mask_w))
            k = max(mask_size//10, 10)
            #k = max(mask_size//20, 6)
            #k = 6
            kernel = np.ones((k,k),np.uint8)
            img_mask = cv2.erode(img_mask,kernel,iterations = 1)
            kernel = np.ones((2,2),np.uint8)
            fake_diff = cv2.dilate(fake_diff,kernel,iterations = 1)
            k = max(mask_size//20, 5)
            #k = 3
            #k = 3
            kernel_size = (k, k)
            blur_size = tuple(2*i+1 for i in kernel_size)
            img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
            k = 5
            kernel_size = (k, k)
            blur_size = tuple(2*i+1 for i in kernel_size)
            fake_diff = cv2.GaussianBlur(fake_diff, blur_size, 0)
            img_mask /= 255
            fake_diff /= 255
            #img_mask = fake_diff
            img_mask = np.reshape(img_mask, [img_mask.shape[0],img_mask.shape[1],1])
            fake_merged = img_mask * bgr_fake + (1-img_mask) * target_img.astype(np.float32)
            fake_merged = fake_merged.astype(np.uint8)
            return fake_merged
class PickableInferenceSession(onnxruntime.InferenceSession): 
    # This is a wrapper to make the current InferenceSession class pickable.
    def __init__(self, model_path, **kwargs):
        super().__init__(model_path, **kwargs)
        self.model_path = model_path

    def __getstate__(self):
        return {'model_path': self.model_path}

    def __setstate__(self, values):
        model_path = values['model_path']
        self.__init__(model_path)

class ModelRouter:
    def __init__(self, onnx_file):
        self.onnx_file = onnx_file

    def get_model(self, **kwargs):
        session = PickableInferenceSession(self.onnx_file, **kwargs)
        print(f'Applied providers: {session._providers}, with options: {session._provider_options}')
        inputs = session.get_inputs()
        input_cfg = inputs[0]
        input_shape = input_cfg.shape
        outputs = session.get_outputs()
        return INSwapper(model_file=self.onnx_file, session=session)

def get_default_providers():
    return ['CUDAExecutionProvider', 'CPUExecutionProvider']

def get_default_provider_options():
    return None

def download(link, filename):
    response = requests.get(link, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024*16  # 1 KB
    progress_bar = tqdm.tqdm(total=total_size, unit='B', unit_scale=True)

    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

    progress_bar.close()

def check_or_download(filename):
    exists = os.path.exists(filename)
    if not exists:
        download(f"https://github.com/RichardErkhov/FastFaceSwap/releases/download/model/{filename}", filename)
def get_model(name, **kwargs):
    check_or_download(name)
    router = ModelRouter(name)
    providers = kwargs.get('providers', get_default_providers())
    provider_options = kwargs.get('provider_options', get_default_provider_options())
    #session_options = kwargs.get('session_options', None)
    model = router.get_model(providers=providers, provider_options=provider_options)#, session_options = session_options)
    return model