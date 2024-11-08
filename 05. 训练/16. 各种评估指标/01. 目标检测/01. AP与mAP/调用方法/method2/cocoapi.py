import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


annTypes = ['segm','bbox','keypoints']

def get_coco_imgs_anns(annFile):
    # 根据json标注文件创建coco示例
    coco=COCO(annFile)

    # 获得json标注文件中的所有类别的名称
    cats = coco.loadCats(coco.getCatIds())
    cats_names=[cat['name'] for cat in cats]
    print(f'COCO categories: \n{" ".join(cats_names)}\n')

    # 获得类别名为pedestrian的类别id
    catIds = coco.getCatIds(catNms=['person'])
    # 根据类别id获得所有该类别的图片id
    imgIds = coco.getImgIds(catIds=catIds)
    # 在图片id中随机选取一个加载出来图片相关信息
    img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
    print(f'COCO imgs: \n{img}\n')

    # 获得某图片ID某类别ID的所有标注值的IDs
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    # 获得某图片ID某类别ID的所有标注值
    anns = coco.loadAnns(annIds)
    print(f'COCO anns: \n{anns}\n')


def eval_coco(annFile, resFile):
    cocoGt = COCO(annFile)
    cocoDt = cocoGt.loadRes(resFile)
    imgIds = sorted(cocoGt.getImgIds())
    cocoEval = COCOeval(cocoGt, cocoDt, annTypes[1])
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    
    
if __name__ == "__main__":
    annFile = 'annotations.json'
    resFile = 'predictions.json'
    get_coco_imgs_anns(annFile)
    eval_coco(annFile, resFile)



    
