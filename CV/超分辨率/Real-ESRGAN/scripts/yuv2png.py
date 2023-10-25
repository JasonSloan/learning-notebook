# 将yuv转成png格式的图像
import numpy as np
import cv2

# Define the width and height of each frame
width = 428  # Replace with the actual width of your YUV frames
height = 240  # Replace with the actual height of your YUV frames

# Define the path to your YUV file
yuv_file = '/root/work/real-esrgan/mount/715/1.yuv'

# Define the format of the YUV data (e.g., YUV420, YUV422)
yuv_format = 'YUV420'  # Replace with your actual format


frame_size = int(width * height * 1.5)  # YUV420 has 1.5 bytes per pixel


# Read the YUV file and convert it to PNG images
with open(yuv_file, 'rb') as f:
    for i in range(10):  # Assuming there are 10 frames in the YUV file
        # Read one frame's worth of data
        frame_data = f.read(frame_size)

        # Convert YUV data to a numpy array
        yuv_frame = np.frombuffer(frame_data, dtype=np.uint8)

        # Reshape the YUV data to the image dimensions
        yuv_frame = yuv_frame.reshape((int(1.5 * height), width))

        height_to_use = int(1.5 * height)
        width_to_use = width - (width % 2)
        yuv_frame = yuv_frame[:height_to_use, :width_to_use]

        # Convert YUV to BGR
        bgr_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)

        # Save the frame as a PNG image
        output_filename = f'../tmp/output_frame_{i}.png'
        cv2.imwrite(output_filename, bgr_frame)

print("Conversion complete.")
