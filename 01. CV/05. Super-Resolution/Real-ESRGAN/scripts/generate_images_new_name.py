# 将图像按照715-730日期分组，升序赋予每张图像一个索引，最终形成图像名字为‘索引_日期_通道ID_图像名字’的新名字
# 与rename_images_with_new_name.py脚本配合使用
import os
def generate_images_new_name(path):
    images_names = os.listdir(path)
    images_dict = {}
    count = 0
    for i in range(715, 731):
        tmp = []
        for j in range(len(images_names)):
            if images_names[j].split('_')[1] != str(i):
                continue
            else:
                tmp.append(images_names[j])
        tmp.sort()
        for k in range(len(tmp)):
            images_dict[count] = tmp[k]
            count += 1
    assert len(images_dict) == len(images_names)
    with open("new_name.txt", 'w') as fp:
        for i in range(len(images_dict)):
            fp.write(images_dict[i] + '\n')


if __name__ == "__main__":
    path = '/root/work/data/hr'
    generate_images_new_name(path)
