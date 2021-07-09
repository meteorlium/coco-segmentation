import torch
import numpy as np
import os
import matplotlib.pyplot as plt

import time
import random


from config import Config
from model.unet import UNet


def pred(net, image, device):

    if len(image.shape) == 2:
        image = np.tile(image, [3, 1, 1])
    else:
        image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, 0)  # [1, 3, h, w]
    image = torch.tensor(image)
    image = image.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        pred_mask = net(image)  # [1, num_classes, h, w]

    pred_mask = pred_mask.cpu().numpy()
    pred_mask = np.argmax(pred_mask, axis=1).squeeze()

    return pred_mask


def create_visual_mask(mask):
    """set color to one hot mask

    Args:
        mask : numpy [h, w], one hot

    Returns:
        visual_mask : numpy [h, w, 3], value is color
    """
    label2color_dict = {0: [0, 0, 0]}
    for r in range(5):
        for g in range(5):
            for b in range(5):
                label2color_dict[r * 5 * 5 + g * 5 + b + 1] = [
                    (5 - r) * 50,
                    (5 - g) * 50,
                    (5 - b) * 50,
                ]
    assert_str = "classes num is too big, please add new color in label2color_dict"
    assert np.max(mask) <= len(label2color_dict), assert_str
    visual_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i in range(visual_mask.shape[0]):  # i for h
        for j in range(visual_mask.shape[1]):
            color = label2color_dict[mask[i, j]]
            visual_mask[i, j] = color
    return visual_mask


def draw(split):
    # image: [h, w, 3]
    # mask: [h, w]
    # pred: [h, w]
    config = Config()

    if split == "val":
        imgs_dir = config.dir_val_image
        masks_dir = config.dir_val_mask
    elif split == "train":
        imgs_dir = config.dir_train_image
        masks_dir = config.dir_train_mask
    else:
        raise RuntimeError("split must be train or val")
    mask_list = os.listdir(masks_dir)

    # random.shuffle(mask_list)

    device = torch.device(config.device_test)
    net = UNet(config.num_channels, config.num_classes, config.is_bilinear)
    net.load_state_dict(torch.load(config.load_test))
    net.to(device=device)
    net.eval()

    # draw
    for i_mask, mask_name in enumerate(mask_list):
        mask = np.load(os.path.join(masks_dir, mask_name))
        image_name = mask_name[:-4] + ".jpg"
        image = plt.imread(os.path.join(imgs_dir, image_name))

        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title("image")

        plt.subplot(1, 3, 2)
        mask_visual = create_visual_mask(mask)
        plt.imshow(mask_visual)
        plt.title("true mask")

        pred_mask = pred(net, image, device)
        pred_mask_visual = create_visual_mask(pred_mask)
        plt.subplot(1, 3, 3)
        plt.imshow(pred_mask_visual)
        plt.title("pred mask")

        plt.tight_layout()
        os.makedirs(f"./save/draw/{split}", exist_ok=True)
        plt.savefig(f"./save/draw/{split}/" + image_name)
        plt.close()

        if i_mask == 20:
            break


def compute_time():
    # image: [h, w, 3]
    # mask: [h, w]
    # pred: [h, w]
    config = Config()

    imgs_dir = config.dir_train_image
    masks_dir = config.dir_train_mask
    mask_list = os.listdir(masks_dir)

    device = torch.device(config.device_test)
    net = UNet(config.num_channels, config.num_classes, config.is_bilinear)
    net.load_state_dict(torch.load(config.load_test))
    net.to(device=device)
    net.eval()

    # compute time
    count_times = 100
    total_time = 0
    for i_mask, mask_name in enumerate(mask_list):
        mask = np.load(os.path.join(masks_dir, mask_name))
        image_name = mask_name[:-4] + ".jpg"
        image = plt.imread(os.path.join(imgs_dir, image_name))

        starttime = time.time()
        pred_mask = pred(net, image, device)
        endtime = time.time()
        total_time += endtime - starttime
        if i_mask == count_times:
            break
    print(f"time per image: {total_time / count_times:.4f}s")


if __name__ == "__main__":
    draw("train")
    draw("val")
    compute_time()
