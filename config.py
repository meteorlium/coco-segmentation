import os


class Config(object):
    def __init__(self):
        # train
        self.device_train = "cuda"
        self.device_test = "cuda"
        self.parallel = False
        self.epochs = 30
        self.batch_size = 14

        # SGD, Adam, RMSprop
        self.optimizer = "Adam"
        self.lr = 1e-4
        # ce, dice, focal
        self.criterion = "focal"
        # CosineAnnealingLR, CosineAnnealingWarmRestarts
        self.scheduler = "CosineAnnealingLR"

        # input
        self.num_channels = 3
        self.num_classes = 91
        self.crop_shape = [512, 512]

        # load and save model
        self.load_train = "./save/model_final.pth"
        self.load_test = "./save/model_final.pth"

        self.save = True

        # model
        self.model = "unet"
        self.is_bilinear = True

        # data dir
        self.dir_coco = "/data/liuming/coco-data"
        self.dir_train_image = os.path.join(self.dir_coco, "train2017")
        self.dir_train_mask = os.path.join(self.dir_coco, "train2017target")
        self.dir_val_image = os.path.join(self.dir_coco, "val2017")
        self.dir_val_mask = os.path.join(self.dir_coco, "val2017target")

        # save and load dir
        self.dir_checkpoints = "./save/checkpoints"
        os.makedirs(self.dir_checkpoints, exist_ok=True)
        self.dir_loss_list = "./save/losslist"
        os.makedirs(self.dir_loss_list, exist_ok=True)

    def information(self):
        return f"""config:

device-train: {self.device_train}
device-test: {self.device_test}
parallel: {self.parallel}
epochs: {self.epochs}
batch size: {self.batch_size}

optimizer: {self.optimizer}
lr: {self.lr}
criterion: {self.criterion}

num-channels: {self.num_channels}
num-classes: {self.num_classes}
crop-shape: {self.crop_shape}

load-train: {self.load_train}
load-test: {self.load_test}

model: {self.model}
is-bilinear: {self.is_bilinear}

dir-coco: {self.dir_coco}
dir-train-image: {self.dir_train_image}
dir-train-mask: {self.dir_train_mask}
dir-val-image: {self.dir_val_image}
dir-val-mask: {self.dir_val_mask}

dir-checkpoints: {self.dir_checkpoints}
dir-loss-list: {self.dir_loss_list}
"""
