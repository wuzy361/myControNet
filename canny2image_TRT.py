from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
import functools

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from pdb import set_trace
import tensorrt as trt
import os
from cuda import cudart

class hackathon():

    def __init__(self):
        self.model_initialized = False
        self.trt_initialized = False

    def create_trt_context(self, engineString):
        if not os.path.exists(engineString):
            print("Failed finding engineString file!")
            exit()
        
        engine = None
        logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(logger, "")
        with open(engineString, 'rb') as f, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()


        nIO = engine.num_io_tensors
        lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
        nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)
        nOutput = nIO - nInput

        
        bufferD = []
        multiply_tuple = lambda tup: functools.reduce(lambda x, y: x * y, tup)
        #data_width = {}
        
        for i in range(nInput):
            bufferD.append(0)
        
        output_tensor = []
        device = torch.device('cuda:0')
        for i in range(nInput, nIO):
            shape = engine.get_binding_shape(i)
            #size = multiply_tuple(shape)
            tensor = torch.zeros([x for x in shape], device = device)
            bufferD.append(tensor)

        for i in range(nIO):
            if isinstance(bufferD[i], torch.Tensor):
                context.set_tensor_address(lTensorName[i], bufferD[i].data_ptr())
            else:
                context.set_tensor_address(lTensorName[i], bufferD[i])

        context = {"context": context, "bufferD": bufferD}

        return context 
    
    def trt_init(self):
        self.trt_context = []
        trt_models =["control_model.plan"]
        for trt_model in trt_models:
            self.trt_context.append(self.create_trt_context(trt_model))
        return True
   
    def model_init(self):
        self.apply_canny = CannyDetector()
        self.model = create_model('/home/player/ControlNet/models/cldm_v15.yaml').cpu()
        self.model.load_state_dict(load_state_dict('/home/player/ControlNet/models/control_sd15_canny.pth', location='cuda'))
        self.model = self.model.cuda()
        self.ddim_sampler = DDIMSampler(self.model)
        return True

    def initialize(self):
        if not self.model_initialized :
            if self.model_init():
                self.model_initialied = True

        if not self.trt_initialized:
            if self.trt_init():
                self.trt_initialized = True

    def process(self, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold, export_onnx = False, use_trt = True):
        with torch.no_grad():
            img = resize_image(HWC3(input_image), image_resolution)
            H, W, C = img.shape

            detected_map = self.apply_canny(img, low_threshold, high_threshold)
            detected_map = HWC3(detected_map)

            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)

            cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
            un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [self.model.get_learned_conditioning([n_prompt] * num_samples)]}
            shape = (4, H // 8, W // 8)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=True)

            trt_context = None
            if use_trt:
                trt_context = self.trt_context

            self.model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            samples, intermediates = self.ddim_sampler.sample(ddim_steps, num_samples,
                                                        shape, export_onnx,  trt_context, cond, verbose=False, eta=eta,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=un_cond)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)

            x_samples = self.model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

            results = [x_samples[i] for i in range(num_samples)]
        return results
