from ultralytics import YOLO

# model = YOLO("yolov8s.yaml")
# model.export(format="onnx")

weight = "weights/pruned.pt"
model = YOLO(weight)
model.export(format="onnx")