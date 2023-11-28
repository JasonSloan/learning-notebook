# 按照mp4->yuv->hevc->yuv->png的顺序进行转换（其中landsea_x265Encoder命令来自马绍祎）
# 每个视频只取一帧，分别生成低分辨率图像和高分辨率图像
import os
import subprocess

import cv2
import numpy as np
from tqdm import tqdm
from  shutil import copy2

def rm_rf_yuv_hevc(path):
    rm_command_yuv = f"rm -rf {os.path.join(path, '*.yuv')}"
    rm_command_hevc = f"rm -rf {os.path.join(path, '*.hevc')}"
    subprocess.run(rm_command_yuv, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    subprocess.run(rm_command_hevc, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def convert_mp4_to_yuv(src, scale, fps, pix_fmt, reserved_time, dst):
    ffmpeg_mp4_to_yuv_command = [
        "ffmpeg",
        "-y",  # Add -y to automatically overwrite the output file if it exists
        "-i", src,  # Input mp4 file
        "-vf", "scale={}:{}".format(scale[0], scale[1]),
        "-r", str(fps),
        "-pix_fmt", pix_fmt,
        "-t", reserved_time,
        dst  # Output yuv file
    ]
    # Use subprocess.run to execute the ffmpeg command
    result_ffmpeg_mp4_to_yuv = subprocess.run(ffmpeg_mp4_to_yuv_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # Check the return code to see if the command was successful
    if result_ffmpeg_mp4_to_yuv.returncode == 0:
        pass
    else:
        print("ffmpeg command (mp4 to yuv) failed with error:")
        print(result_ffmpeg_mp4_to_yuv.stderr)

def convert_yuv_to_hevc(src, dst):
    landsea_x265Encoder_command = [
        "landsea_x265Encoder",
        src,   # Input file for landsea_x265Encoder
        dst  # Output file for landsea_x265Encoder
    ]
    # Use subprocess.run to execute the landsea_x265Encoder command
    result_landsea_x265Encoder = subprocess.run(landsea_x265Encoder_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # Check the return code to see if the landsea_x265Encoder command was successful
    if result_landsea_x265Encoder.returncode == 0:
        pass
    else:
        print("landsea_x265Encoder command failed with error:")
        print(result_landsea_x265Encoder.stderr)

def convert_hevc_to_yuv(src, dst):
    ffmpeg_hevc_to_yuv_command = [
        "ffmpeg",
        "-y",  # Add -y to automatically overwrite the output file if it exists
        "-i", src,  # Input hevc file
        dst  # Output yuv file
    ]

    # Use subprocess.run to execute the ffmpeg command
    result_ffmpeg_hevc_to_yuv = subprocess.run(ffmpeg_hevc_to_yuv_command, stdout=subprocess.PIPE,
                                               stderr=subprocess.PIPE, text=True)

    # Check the return code to see if the ffmpeg command was successful
    # if result_ffmpeg_hevc_to_yuv.returncode == 0:
    #     pass
    # else:
    #     print()
    #     print("ffmpeg command (hevc to yuv) failed with error:")
    #     print(result_ffmpeg_hevc_to_yuv.stderr)

def convert_yuv_to_png(src, dst, size=(428, 240)):
    w, h = size
    frame_size = int(w * h * 1.5)  # YUV420 has 1.5 bytes per pixel

    # Read the YUV file and convert it to PNG images
    with open(src, 'rb') as f:
        # Read one frame's worth of data
        frame_data = f.read(frame_size)

        # Convert YUV data to a numpy array
        yuv_frame = np.frombuffer(frame_data, dtype=np.uint8)

        # Reshape the YUV data to the image dimensions
        yuv_frame = yuv_frame.reshape((int(1.5 * h), w))

        height_to_use = int(1.5 * h)
        width_to_use = w - (w % 2)
        yuv_frame = yuv_frame[:height_to_use, :width_to_use]

        # Convert YUV to BGR
        bgr_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)

        # Save the frame as a PNG image
        output_filename = dst
        success = cv2.imwrite(output_filename, bgr_frame)
        if not success:
            print()
            print("Failed to save frame to {}".format(output_filename))

if __name__ == '__main__':
    root_path = "/root/work/mount"
    sub_directories = os.listdir(root_path)
    total_videos_number = 0
    for sub_directory in sub_directories:
        sub_videos = os.listdir(os.path.join(root_path, sub_directory))
        sub_videos = [sub_video for sub_video in sub_videos if sub_video.endswith(".mp4")]
        total_videos_number += len(sub_videos)
    print("Total videos number: {}".format(total_videos_number))
    image_idx = 0
    for sub_directory in sub_directories:
        rm_rf_yuv_hevc(os.path.join(root_path, sub_directory))
        sub_videos = os.listdir(os.path.join(root_path, sub_directory))
        sub_videos = [sub_video for sub_video in sub_videos if sub_video.endswith(".mp4") and sub_video.startswith('ch')]
        sub_videos_bar = tqdm(sub_videos, desc="Processing {}".format(sub_directory))
        for sub_video in sub_videos_bar:
            try:
                # 第一阶段：保存低分辨率图像
                convert_mp4_to_yuv(
                    src=os.path.join(root_path, sub_directory, sub_video),
                    scale=(428, 240),
                    fps=10,
                    pix_fmt="yuv420p",
                    reserved_time="00:00:01",
                    dst=os.path.join(root_path, sub_directory, sub_video.replace(".mp4", ".yuv"))
                )
                convert_yuv_to_hevc(
                    src=os.path.join(root_path, sub_directory, sub_video.replace(".mp4", ".yuv")),
                    dst=os.path.join(root_path, sub_directory, sub_video.replace(".mp4", ".hevc"))
                )
                convert_hevc_to_yuv(
                    src=os.path.join(root_path, sub_directory, sub_video.replace(".mp4", ".hevc")),
                    dst=os.path.join(root_path, sub_directory, sub_video.replace(".mp4", "_.yuv"))
                )
                convert_yuv_to_png(
                    src=os.path.join(root_path, sub_directory, sub_video.replace(".mp4", "_.yuv")),
                    dst=os.path.join(root_path.replace("mount", "data"), "lr", f"{image_idx:06d}_{sub_directory}_{sub_video.split('.')[0]}.png")
                )
                # 第二阶段：保存高分辨率图像
                convert_mp4_to_yuv(
                    src=os.path.join(root_path, sub_directory, sub_video),
                    scale=(428*4, 240*4),
                    fps=10,
                    pix_fmt="yuv420p",
                    reserved_time="00:00:01",
                    dst=os.path.join(root_path, sub_directory, sub_video.replace(".mp4", "_hr.yuv"))
                )
                convert_yuv_to_png(
                    src=os.path.join(root_path, sub_directory, sub_video.replace(".mp4", "_hr.yuv")),
                    dst=os.path.join(root_path.replace("mount", "data"), "hr", f"{image_idx:06d}_{sub_directory}_{sub_video.split('.')[0]}.png"),
                    size=(428*4, 240*4)
                )
                image_idx += 1
            except Exception as e:
                with open("error.txt", "a") as f:
                    f.write("Error convert video: {}\n".format(sub_video))
            finally:
                rm_rf_yuv_hevc(os.path.join(root_path, sub_directory))




