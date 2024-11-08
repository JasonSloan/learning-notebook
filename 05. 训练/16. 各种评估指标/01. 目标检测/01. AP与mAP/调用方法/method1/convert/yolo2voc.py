import sys
import os
import os.path as osp
import glob
import cv2
import argparse
from tqdm import tqdm


def convert_yolo_coordinates_to_voc(
    x_c_n, y_c_n, width_n, height_n, img_width, img_height
):
    # remove normalization given the size of the image
    x_c = float(x_c_n) * img_width
    y_c = float(y_c_n) * img_height
    width = float(width_n) * img_width
    height = float(height_n) * img_height
    # compute half width and half height
    half_width = width / 2
    half_height = height / 2
    # compute left, top, right, bottom
    # in the official VOC challenge the top-left pixel in the image has coordinates (1;1)
    left = int(x_c - half_width) + 1
    top = int(y_c - half_height) + 1
    right = int(x_c + half_width) + 1
    bottom = int(y_c + half_height) + 1
    return left, top, right, bottom

def run(class_path, src_gtfolder, src_dtfolder, dst_gtfolder, dst_dtfolder, imgs_folder):
    os.makedirs(dst_gtfolder, exist_ok=True)
    os.makedirs(dst_dtfolder, exist_ok=True)
    # read the class_list.txt to a list
    with open(class_path) as f:
        obj_list = f.readlines()
        # remove whitespace characters like `\n` at the end of each line
        obj_list = [x.strip() for x in obj_list]

    for src, dst in zip([src_gtfolder, src_dtfolder], [dst_gtfolder, dst_dtfolder]):
        is_gt = src == src_gtfolder
        # create VOC format files
        txt_list = glob.glob(osp.join(src, "*.txt"))
        if len(txt_list) == 0:
            print(f"Error: no .txt files found in {src}")
            sys.exit()
        bar = tqdm(txt_list, desc=f"Dealing with {src}")
        for tmp_file in bar:
            # 1. check that there is an image with that name
            # get name before ".txt"
            image_name = osp.basename(tmp_file).split(".txt")[0]
            # check if image exists
            for fname in os.listdir(imgs_folder):
                if fname.startswith(image_name):
                    img = cv2.imread(osp.join(imgs_folder, fname))
                    ## get image width and height
                    img_height, img_width = img.shape[:2]
                    break
            else:
                # image not found
                print("Error: image not found, corresponding to " + tmp_file)
                sys.exit()
            # 2. open txt file lines to a list
            with open(tmp_file) as f:
                content = f.readlines()
            # remove whitespace characters like `\n` at the end of each line
            content = [x.strip() for x in content]
            # 3. create new file (VOC format)
            new_file = osp.join(dst, osp.basename(tmp_file))
            with open(new_file, "a") as new_f:
                for line in content:
                    # split a line by spaces.
                    # "c" stands for center and "n" stands for normalized
                    if is_gt:
                        obj_id, x_c_n, y_c_n, width_n, height_n = line.split()
                    else:
                        obj_id, x_c_n, y_c_n, width_n, height_n, conf = line.split()
                    obj_name = obj_list[int(obj_id)]
                    left, top, right, bottom = convert_yolo_coordinates_to_voc(
                        x_c_n, y_c_n, width_n, height_n, img_width, img_height
                    )
                    # add new line to file
                    if is_gt:
                        new_f.write(obj_name + " " + str(left) + " " + str(top) + " " + str(right) + " " + str(bottom) + "\n")
                    else:
                        new_f.write(obj_name + " " + conf + " " + str(left) + " " + str(top) + " " + str(right) + " " + str(bottom) + "\n")
    print("Done!")

def parse_args():
    parser = argparse.ArgumentParser(description="Convert YOLO to VOC format, xywh to xyxy")
    parser.add_argument("--class-path",     type=str, default="class_list.txt",  help="Path to the class list file")
    parser.add_argument("--src-gtfolder",   type=str, default="gt",              help="Path to the ground-truth folder, cls+xywh format")
    parser.add_argument("--src-dtfolder",   type=str, default="dt",              help="Path to the detection folder, cls+xywh+conf format")
    parser.add_argument("--dst-gtfolder",   type=str, default="gt-output",       help="Path to the output ground-truth folder")
    parser.add_argument("--dst-dtfolder",   type=str, default="dt-output",       help="Path to the output detection folder")
    parser.add_argument("--imgs-folder",    type=str, default="images",          help="Path to the image folder")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(**vars(args))
