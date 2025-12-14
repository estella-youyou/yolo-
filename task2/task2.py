import os
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model(r"task2\task2原始视频.mp4",show=True, save=True)
 