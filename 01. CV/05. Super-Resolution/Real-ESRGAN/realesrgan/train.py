# flake8: noqa
import os.path as osp
from basicsr.train import train_pipeline

# import realesrgan.archs
# import realesrgan.data
# import realesrgan.models
import basicsr.models.sr_model
import basicsr.losses.basic_loss

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
