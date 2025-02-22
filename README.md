## Ocularone Hazard Vest Dataset
### v1.0, 2024-03-15

This full dataset consists of 30,712 annotated images of a person wearing a green hazard vest in different outdoor scenarios of a university campus (IISc Bangalore). Our motivation for collecting this dataset is to improve the accuracy of detections of a person wearing a hazard vest and be able to track them autonomously using a drone. This is specifically designed to assist a Visually Impaired Person (VIP) navigate within urban spaces, as part of the Ocularone project described [here](https://dl.acm.org/doi/abs/10.1145/3544549.3585863). 

The images are taken by a [DJI Ryze Tello](https://www.ryzerobotics.com/tello) drone with an onboard camera. The camera has a Field of View (FoV) of 82.6 degrees and captures videos at a resolution of 720p at 30 frames per second (FPS).

### Setup
A person holding the drone at different heights and distances from the person wearing the hazard vest, captures the videos. The frames are extracted from the videos at 10 FPS. 

### Dataset
We have a total of 43 videos of duration between 1 minute to 2 minutes at different locations in the campus. The images have been manually annotated using [Roboflow](https://roboflow.com/). Overall, we have 30,712 images in different scenarios as explained in the following table. 

| Sl. No. | Scenarios | Number of annotated images | 
|  ---:  |  :---         |     ---:       |        
| 1 | VIP walks on a footpath with usual surroundings | 2115   | 
| 2 | VIP walks on a footpath with no pedestrians | 2294 | 
| 3 | VIP walks on a footpath with pedestrians in the FoV | 1371 | 
| 4 | VIP walks on a path with bicycles in the FoV | 901 | 
| 5 | VIP walks on a path with pedestrians in the FoV | 1658 | 
| 6 | VIP walks on a path with pedestrians and cycles in the FoV | 1057 | 
| 7 | VIP walks on the side of road with pedestrians in the FoV | 1326 | 
| 8 | VIP walks on the side of the road with usual surroundings | 1887 | 
| 9 | VIP walks on the side of the road with no pedestrians in the FoV | 2022 | 
| 10 | VIP walks on the side of the road with parked cars in the FoV | 2527 | 
| 11 | VIP walks in broad daylight in different scenarios apart from the above specific ones | 9169 | 
| 12 | Adversarial pictures of the VIP walking in dark backgrounds, blur pictures, bad orientations, evening time, etc. | 4384 |  

A subset of these samples images have been released at this time (2024-03-15). The entire dataset will be shared once the relevant paper, currently under review, is accepted.

### Note
In these datasets, the person wearing the hazard vest is/was a team member of the project who has consented to be part of this data collection effort. They are not visually impaired; they just serve as a proxy for one. To respect privacy, we have blurred the faces of the person wearing the hazard vest and any other bystander whose features are recognizable.

### Training of YOLOv8-nano
We use 10% of images randomly picked from each category to train the off-the-shelf [YOLOv8 nano](https://docs.ultralytics.com/) Deep Neural Network (DNN) model. So, a total of 3866 images were set aside from the dataset and split in the ratio of 80:20 for training and validation data. Rest of the images were kept aside for testing.

### Authors
* Suman Raj
* Bhavani A Madhabhavi
* Prince Modi
* Arnav Rajesh
* Pratham M
* Yogesh Simmhan

### Citation
You may cite this work as follows:

``Ocularone Hazard Vest Dataset, v1.0, Suman Raj, Bhavani A Madhabhavi, Kautuk Astu, Arnav Rajesh, Pratham M, Yogesh Simmhan, DREAM:Lab, Indian Institute of Science, Bangalore, 2024, https://github.com/dream-lab/ocularone-dataset``

### License

 <p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title" rel="cc:attributionURL" href="https://github.com/dream-lab/ocularone-dataset">Ocularone Hazard Vest Dataset</a> by <a rel="cc:attributionURL dct:creator" property="cc:attributionName" href="https://dream-lab.in/">DREAM:Lab, Indian Institute of Science, Bangalore</a> is licensed under <a href="http://creativecommons.org/licenses/by-nc/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">Attribution-NonCommercial 4.0 International<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1"></a></p> 
 
&copy; Indian Institute of Science, Bangalore, 2024
