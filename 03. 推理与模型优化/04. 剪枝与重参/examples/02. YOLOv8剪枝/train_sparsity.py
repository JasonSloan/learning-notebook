"""
修改的代码:
ultralytics/engine/trainer.py: 禁用amp, 增加梯度惩罚项系数
ultralytics/engine/model.py: 主要是将sr参数绑定到self.trainer上
"""
from ultralytics import YOLO

model = YOLO("weights/person_200eopch_best.pt")
# L1正则的惩罚项系数sr=0.02
model.train(data="ultralytics/cfg/datasets/persons.yaml", epochs=100, sr=0.02)