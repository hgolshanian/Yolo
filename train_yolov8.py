from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
model = YOLO('yolov8xl.pt')  
import roboflow


results = model.train(data="/home/admin1/training/root-stem.v6i.yolov8/data.yaml", epochs=100)
results = model.val()

success = model.export(format="onnx")
