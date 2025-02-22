import os
import sys
import time
import csv
from pathlib import Path
import glob
import cv2
import torch
import torch.backends.cudnn as cudnn

import numpy as np

import math

import re

from monodepth.Monodepth2.test_npy import mono_exec

import csv
import subprocess
import ast

class MonoLTInferencing(object):
    def __init__(self):
        #self.model = model
        #self.model = Yolov8mModel.load_model()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def LT_avg(self, bboxes_xyxy, monodepth_output):
        monodepth_objects = []    
        weighted_avg_list = []
        result = []
        for item in bboxes_xyxy:
            print("item = ", item)
            x_scale = 1024 / 1280
            y_scale = 320 / 720

            xl = round(item[1][0] * x_scale)
            yl = round(item[1][1] * y_scale)
            xr = round(item[1][2] * x_scale)
            yr = round(item[1][3] * y_scale)

            monodepth_object = monodepth_output[xl:xr, yl:yr]
            monodepth_objects.append(monodepth_object)

            heat_map_object = np.array(monodepth_object)
            heat_map_vals = heat_map_object.flatten()
            sorted_heat_map_vals = sorted(heat_map_vals)

            ten_percent = round(0.1 * len(heat_map_vals))
            avg_depth = np.average(sorted_heat_map_vals[:ten_percent])
            # weighted_avg_list.append(weighted_avg)
            class_id = item[0]
            b_xyxy = item[1]

            if math.isnan(avg_depth):
                result.append([class_id, b_xyxy, 0.0])
            else:
                result.append([class_id, b_xyxy, avg_depth])
            
        return result

    def monodepth(self, monodepth_model, file_path):
        
        encoder = monodepth_model[0]
        depth_decoder = monodepth_model[1]
        loaded_dict_enc = monodepth_model[2]

        monodepth_output = mono_exec(file_path, monodepth_model)

        # inferencing_output = self.LT_avg(bboxes_xyxy, monodepth_output)
    
        return monodepth_output

    #result = bboxes_list
    #return result
 
#yolov5_model = Loadv5Model().load_model()
#detect = Detectv5(yolov5_model)
#for i in range(10):
    #detect.detect([1])




