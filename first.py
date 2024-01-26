from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import roboflow

model = YOLO('yolov8m.pt')  # load a pretrained model (recommended for training)

import cv2

# Create a VideoCapture object to read from the default camera (0)
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Could not open camera")
    exit()

# Loop through frames and display them live
while True:
    # Read frame from camera
    ret, frame = cap.read()

    # If frame was not successfully captured, break out of loop
    if not ret:
        print("Error capturing frame")
        break

    # Display frame in a window called "Camera"
    cv2.imshow('Camera', model(frame)[0].plot(1))
   
    # Wait for 1 millisecond and check for key press to exit
    if cv2.waitKey(1) == ord('q'):
        break

# Release capture object and close all windows
cap.release()
cv2.destroyAllWindows()


ss = model(frame)[0]

#object_index = [0,59]
fruit_index = [46,47]
for j in fruit_index:    
    plt.figure(j)
    print('For fruit name : ' + ss.names[j])
    for i in range(ss.boxes.shape[0]):
        if (ss.boxes[i].cls.cpu().numpy()[0]==j):            
            print(ss.boxes[i].xywhn) #tensor            
            plt.figure(i)          

            x1,y1,x2,y2=ss.boxes[i].xyxy.cpu().numpy()[0]
            #plt.title(f"x*y is :{abs(x2-x1)*abs(y2-y1)}")
            plt.title(f"x is :{(x2+x1)/2}")
            
            plt.imshow(ss.orig_img[int(y1):int(y2),int(x1):int(x2),:])     
            plt.show()  #added myself to plot, in jupyter this is not needed

#x1,y1,x2,y2=ss.boxes[i].xyxy.cpu().numpy()[0]

#print(x1,y1,x2,y2)

#plt.imshow(ss.orig_img[int(x1):int(x2),int(y1):int(y2),:])            
#plt.show() 
