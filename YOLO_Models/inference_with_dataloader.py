from ultralytics import YOLO
import argparse
import os
import time
from tqdm import tqdm
import cv2
import csv
import torch
from torch.utils.data import Dataset, DataLoader
import glob

# Argument parser for the model name
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--Name", help="Model name", required=True)
args = parser.parse_args()
model_name = args.Name

# Load the YOLO model
device = 'cuda' if torch.cuda.is_available() else 'cpu'

start_time = time.time()

# Load the saved weights into the model
weights_path = model_name

model = YOLO(weights_path)  # Replace with the actual model architecture if it's different

# Move the model to the appropriate device (GPU if available)
model.to(device)

print(f"Model loaded in {time.time() - start_time:.2f} seconds on {device}")
print("Model weight file :", weights_path)

if model is None:
    print(f"Failed to load the model. Please check the model path or name: {weights_path}")
    exit(1)


# Base paths
base_path = "/media/ssd/IPDPS"
category_path = os.path.join(base_path, "categories")
output_dir = os.path.join(base_path, f"outputs/scratch_{model_name}_100_epochs")

os.makedirs(f"{output_dir}/output_inference_time", exist_ok=True)
os.makedirs(f"{output_dir}/results", exist_ok=True)


# Define a custom dataset
class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        img_name = os.path.basename(image_path).split('.')[0]
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Error reading image: {image_path}")
        
        return img_name, image_path, image


# Helper function to collect image paths
def collect_image_paths(folder):
    if folder == "resized":
        # For the "resized" folder, directly collect image paths
        return sorted(glob.iglob(f"{base_path}/{folder}/*.jpg"))
    else:
        # Collect image paths from subfolders
        subfolders = [
            os.path.join(category_path, folder, subfolder)
            for subfolder in os.listdir(os.path.join(category_path, folder))
            if os.path.isdir(os.path.join(category_path, folder, subfolder))
        ]
        image_paths = []
        for subfolder in subfolders:
            image_paths.extend(glob.iglob(f"{subfolder}/*.jpg"))  # Collect images in subfolder
        return sorted(image_paths)  # Sort all collected image paths


# Process each folder
folder_list = [folder for folder in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, folder))]
folder_list += ["resized"]

for folder in folder_list:
    print(f"Processing folder: {folder}")

    # Collect image paths
    image_paths = collect_image_paths(folder)
    if not image_paths:
        print(f"No images found in folder: {folder}")
        continue

    # Create dataset and dataloader
    dataset = ImageDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # Output files
    inference_csv_path = f"{output_dir}/output_inference_time/{folder}.csv"
    results_csv_path = f"{output_dir}/results/{folder}.csv"

    with open(inference_csv_path, 'w') as inference_file, open(results_csv_path, 'w') as results_file:
        inference_writer = csv.writer(inference_file)
        results_writer = csv.writer(results_file)

        # Write headers
        inference_writer.writerow(["img", "class", "conf", "class-name", "read", "run", "write"])
        results_writer.writerow(["img", "class-dict", "names", "bbox"])

        # Process each image using DataLoader
        with torch.no_grad():  # Disable gradients for inference
            for img_name, image_path, image in tqdm(dataloader, desc=f"Processing {folder}"):
                img_name = img_name[0]  # Dataloader returns batched values
                image = image[0]  # Remove batch dimension

                # Measure read time
                read_start = time.time()

                # Preprocess image
                image = cv2.cvtColor(image.numpy().astype('uint8'), cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                image = image.astype('float32') / 255.0  # Normalize to [0, 1]
                image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).to(device)  # NHWC -> BCHW

                _, _, height, width = image_tensor.shape
                new_height = (height + 31) // 32 * 32  # Make divisible by 32
                new_width = (width + 31) // 32 * 32    # Make divisible by 32

                # Resize image tensor
                image_tensor = torch.nn.functional.interpolate(
                    image_tensor, size=(new_height, new_width), mode='bilinear', align_corners=False
                )

                # Perform inference
                infer_start = time.time()
                results = model.predict(image_tensor, conf=0.5)
                infer_time = time.time() - infer_start

                # Process results
                boxes = results[0].boxes if results and len(results) > 0 else None
                cls = boxes.cls.cpu().numpy() if boxes else []
                conf = boxes.conf.cpu().numpy() if boxes else []
                names = [model.names[int(i)] for i in cls] if len(cls) else []

                cls_str = ';'.join(map(str, cls))
                conf_str = ';'.join(f"{x:.4f}" for x in conf)
                names_str = ';'.join(names)

                # Measure write time
                write_start = time.time()
                inference_writer.writerow([img_name, cls_str, conf_str, names_str, time.time() - read_start, infer_time, time.time() - write_start])
                # results_writer.writerow([img_name, results[0].names, names, boxes.tolist() if boxes else []])
                results_writer.writerow([img_name, results[0].names, names, results[0].boxes.xyxy.cpu().numpy().tolist() if results[0].boxes else []])


                # Free GPU memory
                torch.cuda.empty_cache()

    print(f"Completed folder: {folder}")
