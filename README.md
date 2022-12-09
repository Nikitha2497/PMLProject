# PMLProject


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


packages required

pip install gymnasium
pip install matplotlib
pip install torch