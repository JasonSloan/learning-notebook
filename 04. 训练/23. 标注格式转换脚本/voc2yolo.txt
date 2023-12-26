import os
import os.path as osp
import glob
from tqdm import tqdm
import xmltodict


def xml2txt(xml_dir, save_dir, cls_mappings):
    os.makedirs(save_dir, exist_ok=True)
    xmls_path = glob.glob(xml_dir + '/*.xml')
    pbar = tqdm(xmls_path)
    for xml_path in pbar:
        xml_name = osp.basename(xml_path)
        txt_name = xml_name.replace('.xml', '.txt')
        save_path = os.path.join(save_dir, txt_name)
        with open(xml_path, 'r') as fp:
            xml_content = fp.read()
            json_content = xmltodict.parse(xml_content)
            objs = json_content['annotation']['object']
            objs = [objs] if isinstance(objs, dict) else objs
            img_w = int(json_content['annotation']['size']['width'])
            img_h = int(json_content['annotation']['size']['height'])
            ls = []
            for obj in objs:
                if obj['name'] not in cls_mappings.keys():
                    continue
                cls_idx = cls_mappings[obj['name']]
                boxes = obj['bndbox']
                x1 = int(boxes['xmin'])
                y1 = int(boxes['ymin'])
                x2 = int(boxes['xmax'])
                y2 = int(boxes['ymax'])
                xc = int((x1 + x2) / 2)
                yc = int((y1 + y2) / 2)
                w = x2 - x1
                h = y2 - y1
                xc, yc, w, h = xc / img_w, yc / img_h, w / img_w, h / img_h
                xc, yc, w, h = round(xc, 6), round(yc, 6), round(w, 6), round(h, 6)
                l = [str(cls_idx), str(xc), str(yc), str(w), str(h)]
                ls.append(l)
            with open(save_path, 'w') as fp:
                for l in ls:
                    fp.write(' '.join(l)+'\n')
    

if __name__ == '__main__':
    xml_dir = '../datasets/safety_hat_detection/annotations'
    save_dir = '../datasets/safety_hat_detection/raw_data/labels'
    cls_mappings = {'head' : '0' , 'helmet' : '1'}
    xml2txt(xml_dir, save_dir, cls_mappings)

        
            
    


