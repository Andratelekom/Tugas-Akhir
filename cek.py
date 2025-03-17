from ultralytics import YOLO

model = YOLO('terbaru.pt') # yolov3-v7
print(model.names)