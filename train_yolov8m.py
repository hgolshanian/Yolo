from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
#model = YOLO('best16.pt')  
model = YOLO('yolov8m.pt') 
import roboflow

#Device is determined automatically. If a GPU is available then it will be used, otherwise training will start on CPU
results = model.train(data="/home/riseholme/Yolo/yolo-config/tree_detector.v5i.yolov8/data.yaml", epochs=2, batch=-1, imgsz=512, conf=0.25, iou=0.8) #if GPU drivers are installed trainig automatically happens through GPU.
#batch=-1 means auto batch

#results = model.train(data="/home/riseholme/Yolo/yolo-config/tree_detector.v5i.yolov8/data.yaml", epochs=2, imgsz=512) 


# Train the model with 2 GPUs.
#results = model.train(data='coco128.yaml', epochs=100, imgsz=640, device=[0, 1])
#There is one GPU on my laptop. when I installed nividia drivers using sudo ubuntu-drivers autoinstall. training continued using GPU without  mentioning device=[0].It was the same progress when mentioning device=[0].

#predicting
#results = model('/home/riseholme/Yolo/yolo-config/tree_detector.v5i.yolov8/train/images20231103_113137_jpg.rf.e07a471ac39ad74ae9f54ef7f93eb956.jpg')

results = model.val()
#valid_results = model.val()
#print(valid_results)

success = model.export(format="onnx")
