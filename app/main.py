from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
import os
import uuid
from PIL import Image
import torch
import numpy as np
import json
import sys

APP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(APP_DIR, "..")
sys.path.append(BASE_DIR)
from model.unet import UNet
from app.coco_classes_name import coco_classes_name

app = Flask(__name__)


device = torch.device("cuda:2")


def load_model():
    net = UNet(num_channels=3, num_classes=91, is_bilinear=True)
    model_load_path = os.path.join(BASE_DIR, "save/model_final.pth")
    net.load_state_dict(torch.load(model_load_path))
    net.to(device=device)
    net.eval()
    return net


net = load_model()


def pred_to_prob_matrix(image):
    if len(image.shape) == 2:
        image = np.tile(image, [3, 1, 1])
    else:
        image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, 0)  # [1, 3, h, w]
    image = torch.tensor(image)
    image = image.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        pred_mask = net(image)  # [1, num_classes, h, w]

    pred_mask_cpu = pred_mask.cpu()
    prob = torch.nn.functional.softmax(pred_mask_cpu, dim=1)
    prob = prob.numpy()
    return prob


def prob_matrix_to_mask(prob):
    mask = np.argmax(prob, axis=1).squeeze()
    return mask


def prob_matrix_to_mean_prob_list(prob, mask):
    classes = np.unique(mask)
    mean_prob_of_classes = []
    for i_class in classes:
        prob_dict = {}
        prob_dict["class id"] = int(i_class)
        prob_dict["class name"] = coco_classes_name[i_class]
        prob_i_class = prob[0, i_class]
        prob_dict["mean prob"] = float(np.mean(prob_i_class[mask == i_class]))
        prob_dict["area (pixels)"] = int(prob_i_class[mask == i_class].shape[0])
        mean_prob_of_classes.append(prob_dict)

    return mean_prob_of_classes


def set_color():
    label2color_dict = {0: [0, 0, 0]}
    for r in range(5):
        for g in range(5):
            for b in range(5):
                label2color_dict[r * 5 * 5 + g * 5 + b + 1] = [
                    (5 - r) * 50,
                    (5 - g) * 50,
                    (5 - b) * 50,
                ]
    return label2color_dict


def create_visual_mask(mask):
    """set color to one hot mask

    Args:
        mask : numpy [h, w], one hot

    Returns:
        visual_mask : numpy [h, w, 3], value is color
    """
    label2color_dict = set_color()
    visual_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i in range(visual_mask.shape[0]):  # i for h
        for j in range(visual_mask.shape[1]):
            color = label2color_dict[mask[i, j]]
            visual_mask[i, j] = color
    return visual_mask


@app.route("/", methods=["POST", "GET"])
def home_page():
    if request.method == "POST":
        # if get image, upload and save
        f = request.files["file"]
        upload_dir = os.path.join(APP_DIR, "static")
        os.makedirs(upload_dir, exist_ok=True)
        # save name: uuid
        save_image_name = str(uuid.uuid1()) + ".jpg"
        upload_path = os.path.join(upload_dir, save_image_name)
        f.save(upload_path)

        # pred and save
        image = Image.open(upload_path)
        image = np.array(image)
        prob_matrix = pred_to_prob_matrix(image)
        pred_mask = prob_matrix_to_mask(prob_matrix)
        pred_mask_visual = create_visual_mask(pred_mask)
        result = image * 0.5 + pred_mask_visual * 0.5
        result = Image.fromarray(np.uint8(result))
        result.save(os.path.join(upload_dir, "result.jpg"))
        result.save(os.path.join(upload_dir, save_image_name))

        return render_template("upload_ok.html")

    # init page
    return render_template("upload.html")


@app.route("/json", methods=["POST", "GET"])
def json_page():
    if request.method == "POST":
        # if get image, upload and save
        f = request.files["file"]
        upload_dir = os.path.join(APP_DIR, "static")
        os.makedirs(upload_dir, exist_ok=True)
        # save name: uuid
        save_image_name = str(uuid.uuid1()) + ".jpg"
        upload_path = os.path.join(upload_dir, save_image_name)
        f.save(upload_path)

        # pred and save
        image = Image.open(upload_path)
        image = np.array(image)
        prob_matrix = pred_to_prob_matrix(image)
        pred_mask = prob_matrix_to_mask(prob_matrix)
        mean_prob_of_classes = prob_matrix_to_mean_prob_list(prob_matrix, pred_mask)
        pred_mask_visual = create_visual_mask(pred_mask)
        result = image * 0.5 + pred_mask_visual * 0.5
        result = Image.fromarray(np.uint8(result))
        result.save(os.path.join(upload_dir, "result.jpg"))
        result.save(os.path.join(upload_dir, save_image_name))

        return jsonify(
            {
                "classes prob": mean_prob_of_classes,
                "image url": "http://xxx.xx.xxx.xx:xxxxx/static/" + save_image_name,
            }  # ! 改成你的ip地址和端口
        )

    # init page
    return render_template("upload.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8015")  # ! 改成你的ip地址和端口
