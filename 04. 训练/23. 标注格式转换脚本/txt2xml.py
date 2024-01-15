# -*- coding: utf-8 -*-
import os,shutil
import cv2
import numpy as np
from lxml.etree import Element, SubElement, tostring
def txt_xml(img_path,img_name,txt_path,img_txt,xml_path,img_xml):
    #读取txt的信息
    clas=[]
    img=cv2.imread(os.path.join(img_path,img_name))
    imh, imw = img.shape[0:2]
    txt_img=os.path.join(txt_path,img_txt)

    if (not os.path.exists(txt_img)) or (0 == os.path.getsize(txt_img)):
        print(txt_img)
        print("txt_img")
        return
    try:
        with open(txt_img,"r") as f:
            # try:
            #     next(f)
            # except Exception as e:
            #     print(txt_img)
            #     print(e)
            #     return
            for line in f.readlines():
                line = line.strip('\n')
                list = line.split(" ")
                # print(list)
                clas.append(list)
    except Exception as e:
        print(txt_img)
        print(e)
        return

    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = '1'
    node_filename = SubElement(node_root, 'filename')
    #图像名称
    node_filename.text = img_name
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(imw)
    node_height = SubElement(node_size, 'height')
    node_height.text = str(imh)
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'
    for i in range(len(clas)):
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = str(clas[i][0])
        node_pose=SubElement(node_object, 'pose')
        node_pose.text="Unspecified"
        node_truncated=SubElement(node_object, 'truncated')
        node_truncated.text="truncated"
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')

        x = int(float(clas[i][1])*imw)
        y = int(float(clas[i][2])*imh)
        w = int(float(clas[i][3])*imw)
        h = int(float(clas[i][4])*imh)

        x1 = np.clip(x - w//2, 0, imw)
        y1 = np.clip(y - h//2, 0, imh)
        x2 = np.clip(x + w//2, 0, imw)
        y2 = np.clip(y + h//2, 0, imh)

        node_xmin.text = str(x1)
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(y1)
        node_xmax = SubElement(node_bndbox, 'xmax')
        # print(str(int(clas[i][1])+int(clas[i][3])))
        node_xmax.text = str(x2)
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(y2)
    xml = tostring(node_root, pretty_print=True)  # 格式化显示，该换行的换行
    img_newxml = os.path.join(xml_path, img_xml)
    file_object = open(img_newxml, 'wb')
    file_object.write(xml)
    file_object.close()

if __name__ == "__main__":
    #图像文件夹所在位置
    img_path = "/home/huangzm/data/yitiji/person_car/images"
    #标注文件夹所在位置
    txt_path="/home/huangzm/data/yitiji/person_car/labels"
    #txt转化成xml格式后存放的文件夹
    xml_path="/home/huangzm/data/yitiji/person_car/xml"
    for img_name in os.listdir(img_path):
        
        img_xml=img_name.split(".")[0]+".xml"
        img_txt=img_name.split(".")[0]+".txt"
        txt_xml(img_path, img_name, txt_path, img_txt,xml_path, img_xml)
