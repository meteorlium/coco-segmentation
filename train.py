import logging
import os

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from config import Config
from utils.criterion import set_criterion
from utils.dataset import CocoDataset
from model.unet import UNet
from utils.optim import set_optim
from utils.scheduler import set_scheduler
from utils.draw_loss_list import draw_loss_list
from utils.metrics import Evaluator

if __name__ == "__main__":
    logging.basicConfig(
        filename="save/log_train.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    config = Config()
    logging.info(config.information())

    device = torch.device(config.device_train)

    net = UNet(config.num_channels, config.num_classes, config.is_bilinear)
    if config.load_train:
        net.load_state_dict(torch.load(config.load_train))
    net.to(device=device)
    if config.parallel:
        net = torch.nn.DataParallel(net)

    dataset_train = CocoDataset(
        imgs_dir=config.dir_train_image,
        masks_dir=config.dir_train_mask,
        model_input_shape=config.crop_shape,
        is_train=True,
    )
    dataloader_train = DataLoader(
        dataset=dataset_train, batch_size=config.batch_size, shuffle=True
    )
    dataset_val = CocoDataset(
        imgs_dir=config.dir_val_image,
        masks_dir=config.dir_val_mask,
        model_input_shape=config.crop_shape,
        is_train=False,
    )
    dataloader_val = DataLoader(
        dataset=dataset_val, batch_size=config.batch_size, shuffle=False
    )

    optimizer = set_optim(net, config.optimizer, config.lr)
    scheduler = set_scheduler(optimizer, config.scheduler)
    criterion = set_criterion(config.criterion)
    evaluator = Evaluator(config.num_classes)

    epochs = config.epochs
    list_loss_train = np.zeros(epochs)
    list_loss_val = np.zeros(epochs)
    best_val_miou = 0
    for epoch in range(epochs):
        logging.info(f"epoch: {epoch+1}")

        # train

        net.train()
        loss_train = 0
        loss_train_num = 0
        pbar = tqdm(desc=f"Epoch {epoch + 1}/{epochs}", unit="img")

        for images, masks in dataloader_train:

            images = images.to(device=device, dtype=torch.float32)
            masks = masks.to(device=device, dtype=torch.float32)

            pred_masks = net(images)
            loss = criterion(pred_masks, masks)
            loss_train += float(loss)
            loss_train_num += 1
            pbar.set_postfix(**{"loss (batch)": loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.update(images.shape[0])

        loss_train /= loss_train_num
        logging.info(f"loss train: {loss_train:.8f}")
        list_loss_train[epoch] = loss_train

        # eval

        net.eval()
        loss_val = 0
        loss_val_num = 0
        evaluator.reset()

        for images, masks in dataloader_val:

            images = images.to(device=device, dtype=torch.float32)
            masks = masks.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                pred_masks = net(images)
            loss = criterion(pred_masks, masks)
            loss_val += float(loss)
            loss_val_num += 1

            pred_masks = pred_masks.cpu().numpy()
            masks = masks.cpu().numpy()
            pred_masks = np.argmax(pred_masks, axis=1)
            evaluator.add_batch(masks, pred_masks)

        loss_val /= loss_val_num
        # scheduler.step(loss_val)
        logging.info(f"loss val: {loss_val:.8f}")
        list_loss_val[epoch] = loss_val

        confusion_matrix = evaluator.confusion_matrix
        acc = evaluator.Pixel_Accuracy()
        mean_iou = evaluator.Mean_Intersection_over_Union()

        # logging.info('confusion_matrix:')
        # logging.info(confusion_matrix)
        logging.info(f"val mIoU: {mean_iou}")
        logging.info(f"val acc: {acc}")

        if config.save:
            if (epoch + 1) % 5 == 0:
                torch.save(
                    net.state_dict(),
                    os.path.join(config.dir_checkpoints, f"{epoch + 1}.pth"),
                )
            if mean_iou > best_val_miou:
                best_val_miou = mean_iou
                torch.save(
                    net.state_dict(),
                    os.path.join(config.dir_checkpoints, f"best{epoch + 1}.pth"),
                )
            np.save(config.dir_loss_list + f"/train_loss.npy", list_loss_train)
            np.save(config.dir_loss_list + f"/val_loss.npy", list_loss_val)
            draw_loss_list(
                train_loss=list_loss_train, val_loss=list_loss_val, epoch=epoch
            )
