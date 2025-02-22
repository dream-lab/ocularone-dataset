import os
import sys
import csv
import json
import torch
import cv2
import time
import torch2trt
import trt_pose.coco
import trt_pose.models
import torchvision.transforms as transforms
import PIL.Image
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
from jtop import jtop
import multiprocessing


WIDTH = 224
HEIGHT = 224
MODEL_WEIGHTS = 'tasks/human_pose/resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
OPTIMIZED_MODEL = 'tasks/human_pose/resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
DEVICE = torch.device('cuda')


def jetson_logging(file_path):
    #log jetson stats as the inference is going on
    print("jetson logging...")
    with jtop() as jetson:
        with open(file_path, 'w') as csv_file:
            stats = jetson.stats
            writer = csv.DictWriter(csv_file, fieldnames=stats.keys())
            writer.writeheader()
            writer.writerow(stats)
            while jetson.ok():
                stats = jetson.stats
                writer.writerow(stats)


def torch_to_trt():
    """
    First, let's load the JSON file which describes the human pose task. This is in COCO format, it is the category descriptor pulled from the annotations file. We modify the COCO category slighty    to add a neck keypoint. We will use this task description JSON to create a topology tensor, which is an intermediate data structure that describes the part linkages, as well as which channels     in the part affinity field each linkage corresponds to.
    """
    with open('tasks/human_pose/human_pose.json', 'r') as f:
        human_pose = json.load(f)
    topology = trt_pose.coco.coco_category_to_topology(human_pose)
    
    #load the model. model takes 2 parameters - cmap_channels and paf_channels corresponding to the number of heatmap channels and part affinity field channels. The number of part affinity field      channels is 2x the number of links because each link has a channel corresponding to the x and y direction of the vector field for each link. 
    num_parts = len(human_pose['keypoints'])
    num_links = len(human_pose['skeleton'])

    model = trt_pose.models.densenet121_baseline_att(num_parts, 2 * num_links).cuda().eval()
    model.load_state_dict(torch.load(MODEL_WEIGHTS))

    #create some example data and convert use torch2trt to convert PyTorch model to tensorRT optimised model
    data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()
    model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_batch_size=32, max_workspace_size=1<<25)
    torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)

        
if __name__ == "__main__":
        
    #convert torch model to trt
    #print("Converting model to tensorRT...")
    #torch_to_trt()

    inference_log_csv = 'tasks/human_pose/bodypose_final_2chunk.csv'
    jetson_log_csv = 'tasks/human_pose/resnet_jetson_log.csv'
    
    #add headers to inference csv file
    with open(inference_log_csv,'a') as csv_file:
        dict_writer = csv.DictWriter(csv_file, delimiter=',', \
                            fieldnames=['fps','inference_time','batch_size'])
        dict_writer.writeheader()

    p2 = multiprocessing.Process(target=jetson_logging, args=(jetson_log_csv,))
    p2.start()
    try:
        model_trt = torch2trt.TRTModule()
        model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

        chunk_path = "~/Desktop/video_datasets/drone_1/2sec/test-0000.mp4"
        cap = cv2.VideoCapture(chunk_path)
        for i in range(1,11):
            s = time.time()
            ret,frame = cap.read()
            e = time.time()

            dummy_data = torch.zeros((i, 3, HEIGHT, WIDTH)).cuda()
            for _ in range(20):
                starter,ender = torch.cuda.Event(enable_timing=True),\
                                torch.cuda.Event(enable_timing=True)
                starter.record()
                model_trt(dummy_data)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)/1000
                inf_dict = {'fps':1/curr_time, 'inference_time':curr_time, 'batch_size':i}
                print(inf_dict)
                with open(inference_log_csv,'a',newline='') as csv_file:
                    dict_writer = csv.DictWriter(csv_file,inf_dict.keys())
                    dict_writer.writerow(inf_dict)
    except Exception as e:
        print(e)
        p2.terminate()
    p2.terminate()
