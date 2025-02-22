import os
import time
from tqdm import tqdm
import cv2
import csv
import glob

from monodepth.load_model import LoadMonodepthModel
from monodepth.inference import MonoLTInferencing

# Load the YOLO model
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_time = time.time()

model = LoadMonodepthModel().load_model()
mono = MonoLTInferencing()
# model.to(device)

print(f"Model loaded in {time.time() - start_time:.2f} seconds")

if model is None:
    print(f"Failed to load the model. Please check the model path or name")
    exit(1)


# Base paths
base_path = "/home/sumanraj/monodepth/"
output_dir = os.path.join(base_path, "outputs/bodypose")

os.makedirs(f"{output_dir}/output_inference_time", exist_ok=True)
os.makedirs(f"{output_dir}/results", exist_ok=True)


# Helper function to collect image paths
def collect_image_paths(folder):
    return sorted(glob.iglob(f"{base_path}/{folder}/*.png"))    # Change between jpg, png or jpeg based on images


folder_list = ["bodypose_dataset/images"]

for folder in folder_list:
    print(f"Processing folder: {folder}")

    # Collect image paths
    image_paths = collect_image_paths(folder)
    # print(image_paths[0:5])
    # break
    if not image_paths:
        print(f"No images found in folder: {folder}")
        continue
    
    # Output files
    inference_csv_path = f"{output_dir}/output_inference_time/images.csv"
    # results_csv_path = f"{output_dir}/results/images.csv"

    # with open(inference_csv_path, 'w') as inference_file, open(results_csv_path, 'w') as results_file:
    with open(inference_csv_path, 'w') as inference_file:
        inference_writer = csv.writer(inference_file)
        # results_writer = csv.writer(results_file)

        # Write headers
        inference_writer.writerow(["img_name", "run_time"])
        # results_writer.writerow(["img_name", "result"])
        
        print("Before the loop")

        # Process each image using DataLoader
        for i,image_path in enumerate(tqdm(image_paths)):
            img_name = image_path.split('/')[-1]
            
            print("After the iamge name")
            
            # i1=time.time()
            # image =cv2.imread(image_path)
            # i2=time.time()
            
            print("Image loaded")
            # # converting image to numpy array
            # image=image.numpy().astype('uint8')

            # Perform inference
            infer_start = time.time()
            monodepth_output = mono.monodepth(model, image_path)
            infer_time = time.time() - infer_start
            
            print("After the detect model")
            
            print(monodepth_output)

            inference_writer.writerow([img_name, infer_time])
            # results_writer.writerow([img_name, monodepth_output])

    print(f"Completed folder: {folder}")