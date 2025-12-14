import os
from ultralytics import YOLO
model = YOLO("yolov8s.pt")
model.train(
    data="coco128.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    device="cpu"
)