import os
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model("14847453_1920_1080_30fps.mp4",show=True, save=True)
 