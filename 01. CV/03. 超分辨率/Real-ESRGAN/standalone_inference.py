

import cv2
import numpy as np
import torch
from tqdm import tqdm

from standalone_model import RRDBNet
import contextlib
import time


class Profile(contextlib.ContextDecorator):
    """计时代码，可删"""
    def __init__(self, t=0.0):
        self.t = t
        self.cuda = torch.cuda.is_available()

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt

    def time(self):
        if self.cuda:
            torch.cuda.synchronize()
        return time.time()


class ESRGAN():
    def __init__(self, model_path):
        self.load_model(model_path)

    def load_model(self, model_path):
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        loadnet = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(loadnet['params_ema'], strict=True)
        model.eval()
        model = model.to("cuda")
        self.model = model.half()

    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = img / 255.
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        img = img.unsqueeze(0).to("cuda")
        img = img.half()
        return img


    def post_process(self, output_img):
        output_img = output_img.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))
        output = (output_img * 255.0).round().astype(np.uint8)
        return output


    def inference(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        with dt:
            img = self.preprocess(img)
            with torch.no_grad():
                output_img = self.model(img)
            output = self.post_process(output_img)
        return output


if __name__ == '__main__':
    # imgs_paths = ['inputs/00003.png', 'inputs/00017_gray.png', 'inputs/0014.jpg', 'inputs/0030.jpg',
    #               'inputs/ADE_val_00000114.jpg', 'inputs/OST_009.png', 'inputs/children-alpha.png',
    #               'inputs/tree_alpha_16bit.png', 'inputs/video', 'inputs/wolf_gray.jpg']

    # imgs_paths = ['inputs/OST_009.png' for i in range(10)]
    # model_path = "weights/RealESRGAN_x4plus.pth"
    # esrgan = ESRGAN(model_path)
    # dt = Profile()
    # seen = 0
    # output = esrgan.inference(imgs_paths[0])            # dry  run
    # pbar = tqdm(imgs_paths, desc="Inference......")
    # for img_path in pbar:
    #     if img_path.endswith("png") or img_path.endswith("jpg"):
    #         seen += 1
    #         output = esrgan.inference(img_path)
    #         # save_path = f"results/{seen}.png"
    #         # cv2.imwrite(save_path, output)
    # t = dt.t * 1E3 / seen  # speeds per image
    # print("Processed {} pitures, Average processing time: {:.2f} ms, FPS: {:.2f}".format(seen, t, 1 / (t / 1000)))

    import os
    images_root_dir = '/root/work/real-esrgan/train/datasets/landsea/raw_data'
    images_sub_dir = 'lr_sole_psnr15'
    lr_image_names = os.listdir(os.path.join(images_root_dir, images_sub_dir))
    # lr_image_names = ['ch02_007282_728_ch02_20230728111148.png', 'ch04_005727_725_ch04_20230725101335.png']
    lr_images_path = [os.path.join(images_root_dir, images_sub_dir, lr_image_name) for lr_image_name in lr_image_names]
    experiments_root_dir = '/root/work/real-esrgan/train/experiments/21_finetune_RealESRGANx4plusPairedData_psnr15Mixed_DupulicateNo_BaseOn20timeRegionOnly_iter10K.pth_10k_glr4e-5_dlr4e-5'
    model_dir = "models"
    model_name = "net_g_2000.pth"
    model_path = os.path.join(experiments_root_dir, model_dir, model_name)
    save_root_dir = os.path.join(experiments_root_dir, f'{images_sub_dir}_fake_hr'
                                                       f'_{model_name.split(".")[0]}')
    os.makedirs(save_root_dir, exist_ok=True)
    esrgan = ESRGAN(model_path)
    dt = Profile()
    seen = 0       # dry  run
    pbar = tqdm(lr_images_path, desc="Inference......")
    for img_path in pbar:
        if img_path.endswith("png") or img_path.endswith("jpg"):
            seen += 1
            output = esrgan.inference(img_path)
            save_path = os.path.join(save_root_dir, os.path.basename(img_path))
            cv2.imwrite(save_path, output)
    t = dt.t * 1E3 / seen  # speeds per image
    print("Processed {} pitures, Average processing time: {:.2f} ms, FPS: {:.2f}".format(seen, t, 1 / (t / 1000)))

    # import cv2
    # import os
    #
    # input_video_path = 'inputs/ocean/ocean_ship.mp4'
    #
    # output_frames_dir = 'inputs/ocean/output_frames'
    # os.makedirs(output_frames_dir, exist_ok=True)
    #
    # # Open the video file
    # cap = cv2.VideoCapture(input_video_path)
    #
    # # Get video properties
    # frame_width = int(cap.get(3))
    # frame_height = int(cap.get(4))
    # fps = int(cap.get(5))
    #
    # # Define the codec and create a VideoWriter object
    # output_path = 'outputs'
    # seen = 0
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     print(f"frame{seen}")
    #     seen+=1
    #     image_name = os.path.join(output_path, f"{seen}.png")
    #     cv2.imwrite(image_name, frame)
    #
    # cap.release()


