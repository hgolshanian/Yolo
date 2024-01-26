import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import time

# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Configure the color stream
#config.enable_device('238122070540')  # CAMERA IN home
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Enable depth stream

# Start the pipeline
pipeline.start(config)

# Load YOLO model
#model = YOLO('yolov8x.pt')  # Load a pretrained YOLO model
model = YOLO('bestb2.pt')

ii = [0,0,0,0,0,0,0,0,0,0]
k=1
xmin=0
xmax=0

while True:
    # Wait for the next set of frames from the RealSense camera
    frames = pipeline.wait_for_frames()

    # Get the color and depth frames
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    if not color_frame or not depth_frame:
        continue

    # Convert the color frame to a numpy array
    color_image = np.asanyarray(color_frame.get_data())

    # Perform object detection with YOLO
    detections = model(color_image)

    # Display the frame with YOLO object detection
    cv2.imshow('RealSense Camera', detections[0].plot(1))
    cv2.waitKey(1)

# Release the camera and close all OpenCV windows
pipeline.stop()
cv2.destroyAllWindows()

