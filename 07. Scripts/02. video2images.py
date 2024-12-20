# 将视频转换为图像
import cv2
import os

def video_to_images(input_video_idx, input_video_path, output_directory):
    # Open the input video file
    cap = cv2.VideoCapture(input_video_path)

    # Check if the video file opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Create the output directory to store the output frames
    os.makedirs(output_directory, exist_ok=True)

    frame_count = 0  # Initialize frame count

    while True:
        # Read a frame from the input video
        ret, frame = cap.read()

        # Break the loop if we have reached the end of the video
        if not ret:
            break

        # Construct the output image file path
        output_frame_path = os.path.join(output_directory, f"{input_video_idx}_{frame_count:04d}.jpg")

        # Save the frame as an image
        cv2.imwrite(output_frame_path, frame)

        frame_count += 1

    # Release the video object
    cap.release()

    print(f"{frame_count} frames extracted and saved to '{output_directory}'")
if __name__ == '__main__':
    videos_dir = ''
    import glob
    videos_path = glob.glob(os.path.join(videos_dir, "**.mp4"))
    output_dir = "/root/work/real-esrgan/train/datasets/landsea/raw_data/aliyunOSS"
    for video_idx, video_path in enumerate(videos_path):
        video_to_images(video_idx, video_path, output_dir)
