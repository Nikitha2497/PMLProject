# PMLProject

This repo contains code for work done towards the course project for Predictive Machine Learning Course offered at the University of Texas at Austin.

In this project, we tackle the object detection problem in images under hazy or
foggy conditions. We propose to use a Reinforcement Learning based method
to perform the de-hazing of the image. We will use yolov5 to handle the object
detection. The novelty in our approach is to stack both RL dehazing method and
yolov5 detector together to gain performance improvement. We propose to train
both RL dehazing module and detector module together and setup the loss function
to promote joint learning of both the modules

The project is inspired by the work done in -
- Yu Zhang and Yunlong Dong. “Single Image Dehazing via Reinforcement Learning”. In: 2020
IEEE International Conference on Information Technology,Big Data and Artificial Intelligence
(ICIBA). Vol. 1. 2020, pp. 123–126. DOI: 10.1109/ICIBA50161.2020.9277382.

---
### Running the Code
Using the Cityscapes_dataset.yaml

``` python yolov5/train.py --img 640 --batch 16 --epochs 3 --data Cityscapes_dataset.yaml --weights yolov5s.pt```

RL training

``` python RL_learning.py```

Generate dehazed output for training detector

``` python RL_inference.py --mode yolo_train```

Using the Cityscapes_dataset_dehazed.yaml

``` python train.py ....```

Generate dehazed output for testing detector

``` python RL_inference.py --mode yolo_inference```

Testing detector

``` python python yolov5/detect.py --weights yolov5/runs/train/exp/weights/best.pt  --source yolov5/dataset/cityscapes/images/test_foggy```

### Installations
Packages required are available in the environment.yaml file

### References

This repository uses codes from the following repositories. We are grateful to the authors of these, who have made the code available :
- https://github.com/ultralytics/yolov5
- https://github.com/MayankSingal/PyTorch-Image-Dehazing
- https://github.com/jacob5412/Hazing-and-Dehazing-Project

The work was done jointly by [Nikitha Gollamudi](https://github.com/Nikitha2497) and [Devyani Maladkar](https://github.com/YANI-ALT) from the University of Texas at Austin.
