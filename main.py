import os
from typing import Optional

from ultralytics import YOLO

model = YOLO('yolov8s-cls.pt')

results = model.train(
     data = 'dataset',
     imgsz = 512,
     epochs = 3,
     batch = 10,
     name = 'yolov8s_custom'
 )

