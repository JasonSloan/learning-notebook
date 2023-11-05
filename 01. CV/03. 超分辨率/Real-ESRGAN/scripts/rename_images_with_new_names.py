# 按照new_name.txt中的图像名字，索引到对应的图像，然后按照new_name.txt为图像赋予新名字
# 与generate_images_new_name.py脚本配合使用
import os
with open('new_name.txt') as fp:
    images_list = fp.read()
images_list = images_list.split('\n')
images_indexes = ['{:06d}'.format(i) for i in range(len(images_list))]
hr_images = os.listdir('/root/work/data/hr')
for hr_image in hr_images:
    hr_image_index = images_list.index(hr_image)
    hr_image_new_name = images_indexes[hr_image_index] + '_' + "_".join(hr_image.split('_')[1:])
    os.rename(os.path.join('/root/work/data/hr', hr_image), os.path.join('/root/work/data/hr', hr_image_new_name))
lr_images = os.listdir('/root/work/data/lr')
for lr_image in lr_images:
    lr_image_index = images_list.index(lr_image)
    lr_image_new_name = images_indexes[lr_image_index] + '_' + "_".join(lr_image.split('_')[1:])
    os.rename(os.path.join('/root/work/data/lr', lr_image), os.path.join('/root/work/data/lr', lr_image_new_name))
