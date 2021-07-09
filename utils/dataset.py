import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from random import random
from PIL import Image, ImageEnhance


class CocoDataset(torch.utils.data.Dataset):
    """
    将已有的imgs_dir和masks_dir数据制作成pytorch使用的dataset
    """

    def __init__(
        self, imgs_dir, masks_dir, model_input_shape=[512, 512], is_train=True
    ):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.model_input_shape = model_input_shape
        self.name_list = os.listdir(self.masks_dir)
        self.is_train = is_train
        """
        mask dir由cocomask_to_numpy.py产生，只有包含指定class的图片。
        img dir是所有coco数据。
        因此，name list选取mask dir。
        """

    def __getitem__(self, index):
        name = self.name_list[index][:-4]  # remove '.jpg'
        image = plt.imread(os.path.join(self.imgs_dir, name + ".jpg"))
        if len(image.shape) == 2:
            image = np.transpose(np.tile(image, [3, 1, 1]), [1, 2, 0])
        # image: nparray, [h, w, 3]
        mask = np.load(os.path.join(self.masks_dir, name + ".npy"))
        # mask: nparray, [h, w]
        image, mask = self.transform(image, mask)
        return image, mask

    def __len__(self):
        return len(self.name_list)

    def transform(self, image, mask):
        """
        将data从numpy[h, w, c]经过重复、随即裁剪，最终变为tensor[c, h, w]
        """
        if self.is_train:
            # 预处理：使用PIL
            image = Image.fromarray(np.uint8(image))
            mask = Image.fromarray(np.uint8(mask))
            # 随机缩放
            image, mask = self.random_resize(image, mask)
            # 随机翻转
            image, mask = self.random_flip_leftright(image, mask)
            # image, mask = self.random_flip_upbottom(image, mask)
            # # 随机旋转90度
            # image, mask = self.random_rotate(image, mask)
            # # 随机颜色调整
            # image = self.random_color(image)
            # 预处理结束，转回numpy
            image = np.array(image)
            mask = np.array(mask)
        # 随机裁剪到model input shape（如果尺寸不够，重复扩大）
        image, mask = self.random_crop(image, mask)
        # 转换为tensor
        image, mask = self.to_tensor(image, mask)
        return image, mask

    def random_resize(self, image, mask):
        random_scale = random() * 0.5 + 0.75  # scale: [0.75, 1.25]
        target_size_0 = int(image.size[0] * random_scale)
        target_size_1 = int(image.size[1] * random_scale)
        image = image.resize((target_size_0, target_size_1))
        mask = mask.resize((target_size_0, target_size_1), resample=0)
        return image, mask

    def random_rotate(self, image, mask):
        rotate_times = int(random() * 4)
        if rotate_times == 1:
            image = image.transpose(Image.ROTATE_90)
            mask = mask.transpose(Image.ROTATE_90)
        elif rotate_times == 2:
            image = image.transpose(Image.ROTATE_180)
            mask = mask.transpose(Image.ROTATE_180)
        elif rotate_times == 3:
            image = image.transpose(Image.ROTATE_270)
            mask = mask.transpose(Image.ROTATE_270)
        return image, mask

    def random_color(self, image):
        """
        对图像进行颜色抖动
        """
        random_factor = np.random.randint(0, 31) / 10.0
        image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
        random_factor = np.random.randint(10, 21) / 10.0
        image = ImageEnhance.Brightness(image).enhance(random_factor)  # 调整图像的亮度
        random_factor = np.random.randint(10, 21) / 10.0
        image = ImageEnhance.Contrast(image).enhance(random_factor)  # 调整图像对比度
        random_factor = np.random.randint(0, 31) / 10.0
        image = ImageEnhance.Sharpness(image).enhance(random_factor)  # 调整图像锐度
        return image

    def random_flip_leftright(self, image, mask):
        is_flip_leftright = random()
        if is_flip_leftright > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return image, mask

    def random_flip_upbottom(self, image, mask):
        is_flip_upbottom = random()
        if is_flip_upbottom > 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        return image, mask

    def tile(self, image, mask):
        """若尺寸小于crop shape，重复扩大"""
        if image.shape[0] < self.model_input_shape[0]:
            image = np.tile(
                image, (self.model_input_shape[0] // image.shape[0] + 1, 1, 1)
            )
            mask = np.tile(mask, (self.model_input_shape[0] // mask.shape[0] + 1, 1))
        if image.shape[1] < self.model_input_shape[1]:
            image = np.tile(
                image, (1, self.model_input_shape[1] // image.shape[1] + 1, 1)
            )
            mask = np.tile(mask, (1, self.model_input_shape[1] // mask.shape[1] + 1))
        return image, mask

    def random_crop(self, image, mask):
        """
        将data裁剪成需要的大小model_input_shape
        若data shape小于model_input_shape，则先重复扩大
        """
        image, mask = self.tile(image, mask)
        row = int(random() * (image.shape[0] - self.model_input_shape[0]))
        cal = int(random() * (image.shape[1] - self.model_input_shape[1]))
        image = image[
            row : row + self.model_input_shape[0],
            cal : cal + self.model_input_shape[1],
            :,
        ]
        mask = mask[
            row : row + self.model_input_shape[0], cal : cal + self.model_input_shape[1]
        ]
        return image, mask

    def to_tensor(self, image, mask):
        image = np.transpose(image, [2, 0, 1])
        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        return image, mask


if __name__ == "__main__":
    batch_size = 1
    root = "/data/liuming/coco-data"
    imgs_dir = os.path.join(root, "train2017")
    masks_dir = os.path.join(root, "train2017target")
    dataset = CocoDataset(
        imgs_dir, masks_dir, model_input_shape=[512, 512], is_train=True
    )
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    i = 0
    for img, mask in data_loader:
        # for img, mask, origin_img in data_loader:  # !

        print(f"img.shape: {img.shape}")
        print(f"mask.shape: {mask.shape}")
        # save img demo
        img_numpy = img[0].numpy()
        img_numpy = np.transpose(img_numpy, [1, 2, 0])
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(img_numpy)
        mask_numpy = mask[0].numpy()
        plt.subplot(1, 3, 2)
        plt.imshow(mask_numpy)
        # plt.subplot(1, 3, 3)  # !
        # plt.imshow(origin_img[0])  # !
        os.makedirs("debug/test_dataset", exist_ok=True)
        plt.savefig(f"debug/test_dataset/{i}.jpg")
        print("save img and mask figure")

        # print('mask bincount:')
        # print(torch.bincount(mask[0].reshape(-1)))

        i += 1
        if i == 10:
            break
