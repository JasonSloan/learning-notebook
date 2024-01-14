from ultralytics import YOLO

model = YOLO("weights/yolov8s.pt")
# L1正则的惩罚项系数sr=0.02
model.train(data="ultralytics/cfg/datasets/persons.yaml", epochs=200, sr=0)