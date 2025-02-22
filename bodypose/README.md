# Bodypose

This repository contains Bodypose model for estimating the depth of the image. Follow the instructions below to set up and run the models.

## Prerequisites

- Python 3.8 or higher
- pip
- torch

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/dream-lab/ocularone-dataset.git
    cd bodypose
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

1. Create a directory named images and store your image dataset inside this directory

2. Change the [base_path](https://github.com/dream-lab/ocularone-dataset-v0/blob/25a0a41957697e2d231016bad90a34576d54ff76/bodypose/inference_bodypose.py#L26), [output_dir](https://github.com/dream-lab/ocularone-dataset-v0/blob/25a0a41957697e2d231016bad90a34576d54ff76/bodypose/inference_bodypose.py#L27) and [folder_list](https://github.com/dream-lab/ocularone-dataset-v0/blob/25a0a41957697e2d231016bad90a34576d54ff76/bodypose/inference_bodypose.py#L38) variables in the inference_bodypose.py file

3. Run the Bodypose model :
    ```bash
    python inference_bodypose.py
    ```


## Results

The results will be saved in the `output` directory.


# Plotting

## Inference Time

To visualize the inferencing times, use the bodypose_violin.ipynb file.
You need to set the output path within the script

## Model Summary

To generate a summary of the model (like total number of parameters, layers etc.), run model_summary.py file.
'''bash
python model_summary.py
'''

For any questions or issues, please open an issue on this repository.