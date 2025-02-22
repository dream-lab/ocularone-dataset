# Monodepth

This repository contains Monodepth2 model for estimating the depth of the image. Follow the instructions below to set up and run the models.

## Prerequisites

- Python 3.8 or higher
- pip
- torch
- trt_pose (https://github.com/NVIDIA-AI-IOT/trt_pose)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/dream-lab/ocularone-dataset.git
    cd monodepth
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Model

1. Install trt_pose by following the steps mentioned [here](https://github.com/NVIDIA-AI-IOT/trt_pose)

2. Create a directory named images and store your image dataset inside this directory

3. Change the [base_path](https://github.com/dream-lab/ocularone-dataset-v0/blob/f6167b3d8fb126bcf082b790d406149880684e27/monodepth/inference_distance_estimation.py#L27C1-L27C11), [output_dir](https://github.com/dream-lab/ocularone-dataset-v0/blob/f6167b3d8fb126bcf082b790d406149880684e27/monodepth/inference_distance_estimation.py#L28) and [folder_list](https://github.com/dream-lab/ocularone-dataset-v0/blob/f6167b3d8fb126bcf082b790d406149880684e27/monodepth/inference_distance_estimation.py#L39) variables in the inference_distance_estimation.py file

4. Run the Monodepth2 model :
    ```bash
    python inference_distance_estimation.py
    ```


## Results

The results will be saved in the `output` directory.


# Plotting

## Inference Time

To visualize the inferencing times, use the monodepth_violin.ipynb file.
You need to set the output path within the script

## Model Summary

To generate a summary of the model (like total number of parameters, layers etc.), run model_summary.py file.
'''bash
python model_summary.py
'''

For any questions or issues, please open an issue on this repository.
