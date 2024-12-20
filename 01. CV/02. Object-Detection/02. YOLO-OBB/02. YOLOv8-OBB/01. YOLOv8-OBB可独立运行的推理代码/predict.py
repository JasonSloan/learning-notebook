from ultralytics import YOLO

model = YOLO(model="weights/yolov8s-obb.pt")
model.predict("images/P0006.png", imgsz=[1024, 1024], save=True)        