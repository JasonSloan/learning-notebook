from ultralytics import YOLO

model = YOLO("yolov8s.yaml")

model.export(format="onnx")