import torch
import numpy as np
import logging
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

from torch.utils.data import DataLoader

from config import Config
from model.unet import UNet
from utils.dataset import CocoDataset
from utils.metrics import Evaluator


def vision_confusion_matrix(matrix, log=True):
    if log:
        matrix = np.log(matrix + 1)
    plt.figure()
    plt.matshow(matrix, cmap="binary")
    plt.xlabel("classes (true)")
    plt.ylabel("classes (pred)")
    plt.title("confusion matrix (log)")
    plt.savefig("./save/vision_confusion_matrix.jpg")


if __name__ == "__main__":
    logging.basicConfig(
        filename="./save/log_test.log", level=logging.INFO, format="%(message)s"
    )
    config = Config()

    device = torch.device(config.device_test)

    net = UNet(config.num_channels, config.num_classes, config.is_bilinear)
    if config.parallel:
        net = torch.nn.DataParallel(net)
    if config.load_test:
        net.load_state_dict(torch.load(config.load_test))
        logging.info(f"model: {config.load_test}")
    net.to(device=device)

    dataset = CocoDataset(
        imgs_dir=config.dir_val_image,
        masks_dir=config.dir_val_mask,
        model_input_shape=config.crop_shape,
        is_train=False,
    )
    dataloader = DataLoader(
        dataset=dataset, batch_size=config.batch_size, shuffle=False
    )

    evaluator = Evaluator(config.num_classes)

    # test and compute metrics

    net.eval()

    finish_nums = 0

    for images, masks in dataloader:

        finish_nums += images.shape[0]
        print(f"{finish_nums} / {len(dataset)}")

        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)

        with torch.no_grad():
            pred_masks = net(images)

        pred_masks = pred_masks.cpu().numpy()
        masks = masks.cpu().numpy()
        pred_masks = np.argmax(pred_masks, axis=1)

        masks = np.int64(masks)
        evaluator.add_batch(masks, pred_masks)

    confusion_matrix = evaluator.confusion_matrix
    acc_each_class = evaluator.pixel_accuracy_each_class()
    acc_mean = evaluator.Pixel_Accuracy()
    iou_each_class = evaluator.intersection_over_union_each_class()
    iou_mean = evaluator.Mean_Intersection_over_Union()

    np.save("./save/confusion_matrix.npy", confusion_matrix)
    np.save("./save/acc_each_class.npy", acc_each_class)
    np.save("./save/iou_each_class.npy", iou_each_class)

    vision_confusion_matrix(confusion_matrix, log=True)
    logging.info(f"val mean IoU: {iou_mean}")
    logging.info(f"val mean acc: {acc_mean}")

    # coco classes names
    ann_file = "/data/liuming/coco-data/annotations/instances_train2017.json"
    coco = COCO(ann_file)
    cat_ids = coco.getCatIds()
    logging.info(
        f"id:  0, name:      background, IoU: {iou_each_class[0]:.6f}, acc: {acc_each_class[0]:.6f}"
    )
    for cat_id in cat_ids:
        cat = coco.loadCats(cat_id)[0]
        id = cat["id"]
        name = cat["name"]
        logging.info(
            f"id: {id:2d}, name: {name.rjust(15)}, IoU: {iou_each_class[id]:.6f}, acc: {acc_each_class[id]:.6f}"
        )
