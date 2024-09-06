from ultralytics import YOLO

model = YOLO(model="weights/yolov8s-obb.pt")
model.export(format='onnx', imgsz=(640, 640), simplify=True)