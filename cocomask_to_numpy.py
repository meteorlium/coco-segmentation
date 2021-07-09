"""
读取mask，转化为one hot形式的numpy矩阵并保存
"""
import os
import numpy as np
from pycocotools.coco import COCO

from config import Config


def cocomask_to_target(input_dir, split, num_classes=-1):
    # load coco
    # num_classes = -1 means all classes
    annotation_file = os.path.join(
        input_dir, "annotations", f"instances_{split}2017.json"
    )
    print(annotation_file)
    coco = COCO(annotation_file)
    # get cat ids
    catIds = coco.getCatIds()
    if num_classes > 0:
        catIds = catIds[:num_classes]
    print("class index:")
    print(catIds)
    # get img ids
    imgIds = []
    for index in catIds:
        imgId = coco.getImgIds(catIds=index)
        imgIds += imgId
    imgIds = list(set(imgIds))
    num_img = len(imgIds)
    print(f"images num: {num_img}")
    # generate mask
    for i_img, imgId in enumerate(imgIds):
        if (i_img + 1) % 1000 == 0:
            print(i_img, "/", num_img)
        img = coco.loadImgs(imgId)[0]
        annIds = coco.getAnnIds(imgIds=imgId, catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        mask = np.zeros([img["height"], img["width"]], dtype="uint8")
        for ann in anns:
            ann_mask = coco.annToMask(ann)
            mask[ann_mask > 0] = ann["category_id"]

        np.save(
            os.path.join(input_dir, split + "2017target", img["file_name"][:-4]), mask
        )


if __name__ == "__main__":
    config = Config()
    cocomask_to_target(config.dir_coco, "train", config.num_classes)
    cocomask_to_target(config.dir_coco, "val", config.num_classes)
