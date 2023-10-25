# 将一个文件夹中的图像转成video
import cv2
import os


def images_to_video(input_directory, output_video_path, fps=10):
    # Get the list of image files in the input directory
    image_files = sorted([f for f in os.listdir(input_directory) if f.endswith(('.jpg', '.jpeg', '.png'))])

    if not image_files:
        print(f"No image files found in '{input_directory}'.")
        return

    # Get the dimensions of the first image to determine the video frame size
    first_image_path = os.path.join(input_directory, image_files[0])
    first_image = cv2.imread(first_image_path)
    frame_height, frame_width, _ = first_image.shape

    # Define the codec and create a VideoWriter object to save the video in MP4 format
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    for image_file in image_files:
        image_path = os.path.join(input_directory, image_file)
        frame = cv2.imread(image_path)
        out.write(frame)

    # Release the video writer object
    out.release()

    print(f"Images in '{input_directory}' converted to MP4 video: '{output_video_path}'")


if __name__ == '__main__':
    # Example usage:
    input_directory = '../outputs'
    output_video_path = '../outputs/output_video.mp4'
    images_to_video(input_directory, output_video_path, fps=10)
