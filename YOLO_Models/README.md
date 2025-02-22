# YOLO Models

This repository contains YOLO models for hazard vest detection. Follow the instructions below to set up and run the models.

## Prerequisites

- Python 3.8 or higher
- pip
- torch

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/dream-lab/ocularone-dataset.git
    cd YOLO_Models
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

1. Download the pre-trained YOLO weights from the hugging face

2. Create a directory named categories and store the occularone dataset inside the directory

3. Change the [base_path](https://github.com/dream-lab/ocularone-dataset-v0/blob/25a0a41957697e2d231016bad90a34576d54ff76/YOLO_Models/inference_with_dataloader.py#L40), [output_dir](https://github.com/dream-lab/ocularone-dataset-v0/blob/25a0a41957697e2d231016bad90a34576d54ff76/YOLO_Models/inference_with_dataloader.py#L42) and [folder_list](https://github.com/dream-lab/ocularone-dataset-v0/blob/25a0a41957697e2d231016bad90a34576d54ff76/YOLO_Models/inference_with_dataloader.py#L87) variables in the inference_with_dataloader.py file

4. Run the YOLO model :
    ```bash
    python inference_with_dataloader.py -n <weight_file_path>
    ```


## Results

The results will be saved in the `output` directory.


# Plotting

## Inference Time

To visualize the inferencing times, use the yolo_violin.ipynb file.
You need to set the output path within the script

## Accuracy

To determine the accuracy of the YOLO models, use the accuracy.ipynb file.
As it is a jupyter notebook, you can run this script interactively.
The output will be a confusion matrix in the form of pdf as well as png.

## Model Summary

To generate a summary of the model (like total number of parameters, layers etc.), run model_summary.py file.
'''bash
python model_summary.py
'''

For any questions or issues, please open an issue on this repository.