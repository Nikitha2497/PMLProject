# PMLProject


Using the Cityscapes_dataset.yaml

``` python train.py ....```

RL training

``` python RL_learning.py```

Generate dehazed output for training detector

``` python RL_inference.py --mode yolo_train```

Using the Cityscapes_dataset_dehazed.yaml

``` python train.py ....```

Generate dehazed output for testing detector

``` python RL_inference.py --mode yolo_inference```

Testing detector

``` python detect.py ....```
