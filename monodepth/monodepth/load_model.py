import os
import sys
#sys.path.append('../../detection/yolov5')
import time
import argparse
import csv
from pathlib import Path
import glob
import cv2
import torch
import torch.backends.cudnn as cudnn

from ultralytics import YOLO
#sys.path.append('models')
#sys.path.append('weights')

import monodepth.Monodepth2.networks as networks
from monodepth.Monodepth2.layers import disp_to_depth
from monodepth.Monodepth2.utils import download_model_if_doesnt_exist

import torch
#import csv
#from models.common import DetectMultiBackend
#from models import yolo
#from detection.yolov5.utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
#from utils_yolov5.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
#from detection.yolov5.utils.plots import Annotator, colors, save_one_box
#from detection.yolov5.utils.torch_utils import select_device, time_sync

def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    # parser.add_argument('--image_path', type=str, default=input_folder, help='path to a test image or folder of images', required=True)
    parser.add_argument('--model_name', type=str, default="mono_1024x320",
                        help='name of a pretrained model to use',
                        choices=[
                            "mono_640x192",
                            "stereo_640x192",
                            "mono+stereo_640x192",
                            "mono_no_pt_640x192",
                            "stereo_no_pt_640x192",
                            "mono+stereo_no_pt_640x192",
                            "mono_1024x320",
                            "stereo_1024x320",
                            "mono+stereo_1024x320"])
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    parser.add_argument("--pred_metric_depth",
                        help='if set, predicts metric depth instead of disparity. (This only '
                             'makes sense for stereo-trained KITTI models).',
                        action='store_true')
    # parser.add_argument("--file_path", type=str, help='input video file')

    return parser.parse_args()

class LoadMonodepthModel(object):
    def __init__(self):
        pass

    def load_model(self):
        args = parse_args()
        """Function to predict for a single image or folder of images
        """
        assert args.model_name is not None, \
            "You must specify the --model_name parameter; see README.md for an example"

        if torch.cuda.is_available() and not args.no_cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        if args.pred_metric_depth and "stereo" not in args.model_name:
            print("Warning: The --pred_metric_depth flag only makes sense for stereo-trained KITTI "
              "models. For mono-trained models, output depths will not in metric space.")

        download_model_if_doesnt_exist(args.model_name)
        model_path = os.path.join("models", args.model_name)
        # print("-> Loading model from ", model_path)
        encoder_path = os.path.join(model_path, "encoder.pth")
        depth_decoder_path = os.path.join(model_path, "depth.pth")
        encoder = networks.ResnetEncoder(18, False)
        loaded_dict_enc = torch.load(encoder_path, map_location=device)
        #stride = model.stride
        #imgsz = check_img_size(imgsz,s=stride)

        # extract the height and width of image that this model was trained with
        feed_height = loaded_dict_enc['height']
        feed_width = loaded_dict_enc['width']
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
        encoder.load_state_dict(filtered_dict_enc)
        encoder.to(device)
        encoder.eval()

        #print("   Loading pretrained decoder")
        depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

        loaded_dict = torch.load(depth_decoder_path, map_location=device)
        depth_decoder.load_state_dict(loaded_dict)

        depth_decoder.to(device)
        depth_decoder.eval()

        return [encoder, depth_decoder, loaded_dict_enc]

#Loadv5Model().load_model()




"""
    #do some dummy inferencing for GPU to warmup
    #print("doing some dummy inferencing for batch size 1..........")
    dummy_data = torch.zeros(1,3,640,640).to(device)
    #starter,ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for i in range(10):
        t1 = time_sync() 
        _ = model(dummy_data,False,False)
        t2 = time_sync()
        #inference_time, fps = t2-t1, 1/(t2-t1)

    chunk_duration = opt.chunk_dur
    chunks = glob.glob(source)
    chunks = chunks[:100]
    batches = [1,2,3,4,5]
    batches = [i*chunk_duration for i in batches]

    for chunk in chunks:
        for batch in batches:
            cap = cv2.VideoCapture(chunk)
            dummy_data = torch.zeros(batch,3,640,640).to(device)
            #frame_start,frame_end,e2e_start,e2e_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True),\
             #       torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

            while True:
                read_start = time.time()
                ret, frame = cap.read()
                read_end = time.time()

                if ret:
                    start_time = time_sync()
                    outs = model(dummy_data,augment=False)[0]
                    end_time = time_sync()
                    inf_time = end_time-start_time

                    #post processing
                    #pred = non_max_suppression(outs,0.75,0.45,None,False,1)

                    inf_dict = {'inference_time': inf_time, 'batch_size':batch, \
                                    'chunk_size':chunk_duration,'read_time':read_end-read_start}
                    print(inf_dict)
                    with open(output_path, 'a', newline='') as csv_file:
                        dict_writer = csv.DictWriter(csv_file, inf_dict.keys())
                        dict_writer.writerow(inf_dict)
                else:
                    break

            cap.release()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/best_overall.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.75, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', type=str, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--cfg', type=str, default='', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.cfg path')
    parser.add_argument('--chunk-dur', type=int, default=1, help='time duration of each chunk')
    opt = parser.parse_args()

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['']:
                strip_optimizer(opt.weights)
        else:
            detect(opt)
"""
