from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os

model = YOLO('best12.pt')  # load a pretrained model (recommended for training)

image_directory = '/home/riseholme/Yolo/yolo8n_tree/tree_pic'  # Replace this with your image directory path

image_files = [os.path.join(image_directory, file) for file in os.listdir(image_directory) if file.endswith(('jpg', 'jpeg', 'png'))]

for img_path in image_files:
    frame = cv2.imread(img_path)

    ss = model(frame)[0]
       

    for i in range(ss.boxes.shape[0]):
      if (ss.boxes[i].cls.cpu().numpy()[0]==0):
        x1, y1, x2, y2 = ss.boxes[i].xyxy.cpu().numpy()[0]
        frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
