# 马给了一些阿里云对象存储的视频，本脚本目的是每个视频提取一张图片
import glob
import os
import cv2
from tqdm import tqdm


def exact_one_image_per_video(video_idx, video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    # Create the output directory to store the output frames
    os.makedirs(output_dir, exist_ok=True)
    ret = False
    while not ret:
        ret, frame = cap.read()
    if not ret:
        print(f"Could not read video: {video_path}.")
        return
    output_frame_path = os.path.join(output_dir, f"{video_idx:04d}.jpg")
    cv2.imwrite(output_frame_path, frame)
    cap.release()


if __name__ == '__main__':
    videos_dir = '/root/work/real-esrgan/train/datasets/landsea/raw_data/aliyunOSS/videos'
    videos_path = glob.glob(os.path.join(videos_dir, "**/**/**.mp4"))
    output_dir = "/root/work/real-esrgan/train/datasets/landsea/raw_data/aliyunOSS/pics"
    pbar = tqdm(videos_path, total=len(videos_path))
    video_idx = 0
    for video_path in pbar:
        exact_one_image_per_video(video_idx, video_path, output_dir)
        video_idx+=1
