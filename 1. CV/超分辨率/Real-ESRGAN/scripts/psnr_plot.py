# 根据log日志文件，提取出psnr值，然后绘制psnr随着iter次数变化的曲线
import re
from matplotlib import pyplot as plt
import os.path as osp

plt.rcParams['axes.unicode_minus'] = False  # 设置可以显示负号


def find_psnr(log_path):
    psnr_values = []
    with open(log_path) as fp:
        content_lines = fp.readlines()
    for line in content_lines:
        pattern = re.compile(r'# psnr: \d+\.\d+')
        extract_number_pattern = re.compile(r'-?\d+\.\d+|\d+')
        is_matched = pattern.findall(line)
        if is_matched:
            values = extract_number_pattern.findall(line)
            psnr_values.append([values[1], values[0]])
    return psnr_values


def plot_psnr(psnr_values, log_path):
    plt.figure(figsize=(8, 4))  # 设置画布大小
    x = [int(pair[0]) for pair in psnr_values]
    y = [float(pair[1]) for pair in psnr_values]
    plt.plot(x, y, "r-o")  # 绘制图像1,"r"代表红色"-"代表实线"o"代表每一个点都标注出来
    plt.xlabel("iter")  # 为x轴命名
    plt.ylabel("psnr")  # 为y轴命名
    plt.legend()  # 显示图标y1和y2
    plt.title("iter_psnr")  # 为整个图像命名
    save_path = osp.join(osp.dirname(log_path), "psnr.png")
    plt.savefig(save_path)  # 保存图片


if __name__ == '__main__':
    log_path = "/root/work/real-esrgan/train/experiments" \
               "/finetune_RealESRNetx4plus_TimeRegionOnly_DupulicateYes_BaseOnNone_RealESRGANx4plus_sigleGPU/train_finetune_RealESRNetx4plus_TimeRegionOnly_DupulicateYes_BaseOnNone_RealESRGANx4plus_sigleGPU_20231017_073344.log"
    psnr_values = find_psnr(log_path)
    plot_psnr(psnr_values, log_path)
