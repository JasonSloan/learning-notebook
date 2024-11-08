from ultralytics import YOLO

# model = YOLO("yolov8s.yaml")
# model.export(format="onnx")

weight = "/root/study/yolov8_prune/runs/detect/train2/weights/last.pt"
model = YOLO(weight)
model.export(format="onnx")         # 无需手动eval