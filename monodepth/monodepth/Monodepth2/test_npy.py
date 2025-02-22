from __future__ import absolute_import, division, print_function
import sklearn
import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
# from pypfm import PFMLoader

import cv2

import torch
from torchvision import transforms, datasets

import monodepth.Monodepth2.networks as networks
from monodepth.Monodepth2.layers import disp_to_depth
from monodepth.Monodepth2.utils import download_model_if_doesnt_exist
# from evaluate_depth import STEREO_SCALE_FACTOR

def mono_exec(file_path, monodepth_model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    encoder = monodepth_model[0]
    depth_decoder = monodepth_model[1]
    loaded_dict_enc = monodepth_model[2]

    file_path_ext = file_path.split('.')[1]
    
    frames =[]

    # if input is video path
    if file_path_ext == 'mp4':
        cap = cv2.VideoCapture(file_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
    
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    
    if len(frames)==0:
        frames=[cv2.imread(file_path)]

    with torch.no_grad():
        input_image = frames[0]
        original_height, original_width, ch = input_image.shape
        # print (input_image.shape, original_width, original_height, ch)
        input_image = np.resize(input_image, (feed_width, feed_height, 3))
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION

        input_image = input_image.to(device)
        features = encoder(input_image)
        outputs = depth_decoder(features)
            # write_pfm(output_folder+"/"+file_name+".pfm", result.astype(np.float32))

        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Saving numpy file
            # output_name = os.path.splitext(os.path.basename(image_path))[0]
        scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
        rel_depth = depth[0][0].cpu().numpy()
        # Saving colormapped depth image
        #disp_resized_np = disp_resized.squeeze().cpu().numpy()
        #vmax = np.percentile(disp_resized_np, 95)
        #normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
        #mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        #colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
        #im = pil.fromarray(colormapped_im)

    return rel_depth.T

if __name__ == '__main__':

    args = parse_args()
