python inference_with_dataloader.py -n=yolov8n 2>&1 | tee scratch_yolov8n_100_epochs_resized.txt
python inference_with_dataloader.py -n=yolov8m 2>&1 | tee scratch_yolov8m_100_epochs_resized.txt
python inference_with_dataloader.py -n=yolov8x 2>&1 | tee scratch_yolov8x_100_epochs_resized.txt

python inference_with_dataloader.py -n=yolov11n 2>&1 | tee scratch_yolov11n_100_epochs_resized.txt
python inference_with_dataloader.py -n=yolov11m 2>&1 | tee scratch_yolov11m_100_epochs_resized.txt
python inference_with_dataloader.py -n=yolov11x 2>&1 | tee scratch_yolov11x_100_epochs_resized.txt