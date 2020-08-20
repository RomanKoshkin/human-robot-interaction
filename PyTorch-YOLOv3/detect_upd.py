from __future__ import division


from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime

from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

class Recog:

    def __init__(self):
        prefix = "/home/torobo/catkin_ws/src/tutorial/PyTorch-YOLOv3/"
        # print(prefix)
        self.image_folder =     prefix + "data/samples"
        self.model_def =        prefix + "config/yolov3.cfg"
        self.weights_path =     prefix + "weights/yolov3.weights"
        self.class_path =       prefix + "data/coco.names"
        self.conf_thres =   0.8
        self.nms_thres =    0.4
        self.batch_size =   1
        self.n_cpu =        0
        self.img_size =     416
        self.org_img_size = (0, 0)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.exists("output"):
            os.makedirs("output")

        self.model = Darknet(self.model_def, self.img_size).to(self.device)

        if self.weights_path.endswith(".weights"):
            # Load darknet weights
            self.model.load_darknet_weights(self.weights_path)
        else:
            # Load checkpoint weights
            self.model.load_state_dict(torch.load(self.weights_path))

        self.model.eval()  # Set in evaluation mode

        self.classes = load_classes(self.class_path)  # Extracts class labels from file

        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        # Bounding-box colors
        cmap = plt.get_cmap("tab20b")
        self.colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    
    def pad_to_square(self, img, pad_value):
        c, h, w = img.shape
        dim_diff = np.abs(h - w)
        # (upper / left) padding and (lower / right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
        # Add padding   
        img = F.pad(img, pad, "constant", value=pad_value)
        return img, pad

    def resize(self, image, size):
        image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
        return image

    def detect(self, image):
        self.org_img_size = image.shape[:2]
        # preprocess
        img = transforms.ToTensor()(image)
        img, _ = self.pad_to_square(img, 0)
        img = self.resize(img, self.img_size)
        input_img = Variable(img.type(self.Tensor))

        # detect
        with torch.no_grad():
            detections = self.model(input_img.unsqueeze(0))
            detections = non_max_suppression(detections, self.conf_thres, self.nms_thres)[0]

        # Create plot
        # plt.figure()
        # fig = plt.figure()
        # ax = plt.gca()
        # X = ax.imshow(img.detach().numpy().transpose(1,2,0))

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, self.img_size, self.org_img_size)
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(self.colors, n_cls_preds)
            # for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

            #     # print("\t+ Label: %s, Conf: %.5f" % (self.classes[int(cls_pred)], cls_conf.item()))

            #     box_w = x2 - x1
            #     box_h = y2 - y1

            #     color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            #     # Create a Rectangle patch
            #     bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                
            #     # Add the bbox to the plot
            #     # ax.add_patch(bbox)
            #     # Add label
            #     # plt.text(
            #     #     x1,
            #     #     y1,
            #     #     s=self.classes[int(cls_pred)],
            #     #     color="white",
            #     #     verticalalignment="top",
            #     #     bbox={"color": color, "pad": 0})
            # # plt.close()
            # return bbox
            return detections.detach().numpy()

# recog = Recog()
# im = np.array(Image.open('data/samples/dog.jpg'))
# array = recog.detect(im)
# print('hey')

class Recog2:

    def __init__(self, chkpnt):
        prefix = "/home/torobo/catkin_ws/src/tutorial/PyTorch-YOLOv3/"
        # print(prefix)
        self.model_def =        "/home/torobo/PyTorch-YOLOv3/config/yolov3-custom.cfg"
        self.class_path =       "/home/torobo/PyTorch-YOLOv3/data/custom/classes.names"
        self.weights_path =     "/home/torobo/PyTorch-YOLOv3/checkpoints/yolov3_ckpt_{}.pth".format(chkpnt)
        self.conf_thres =   0.8
        self.nms_thres =    0.4
        self.batch_size =   1
        self.n_cpu =        0
        self.img_size =     416
        self.org_img_size = (0, 0)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.exists("output"):
            os.makedirs("output")

        self.model = Darknet(self.model_def, self.img_size).to(self.device)

        if self.weights_path.endswith(".weights"):
            # Load darknet weights
            self.model.load_darknet_weights(self.weights_path)
        else:
            # Load checkpoint weights
            self.model.load_state_dict(torch.load(self.weights_path))

        self.model.eval()  # Set in evaluation mode

        self.classes = load_classes(self.class_path)  # Extracts class labels from file

        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        # Bounding-box colors
        cmap = plt.get_cmap("tab20b")
        self.colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    
    def pad_to_square(self, img, pad_value):
        c, h, w = img.shape
        dim_diff = np.abs(h - w)
        # (upper / left) padding and (lower / right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
        # Add padding   
        img = F.pad(img, pad, "constant", value=pad_value)
        return img, pad

    def resize(self, image, size):
        image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
        return image

    def detect(self, image):
        self.org_img_size = image.shape[:2]
        # preprocess
        img = transforms.ToTensor()(image)
        img, _ = self.pad_to_square(img, 0)
        img = self.resize(img, self.img_size)
        input_img = Variable(img.type(self.Tensor))

        # detect
        with torch.no_grad():
            detections = self.model(input_img.unsqueeze(0))
            detections = non_max_suppression(detections, self.conf_thres, self.nms_thres)[0]

        # Create plot
        # plt.figure()
        # fig = plt.figure()
        # ax = plt.gca()
        # X = ax.imshow(img.detach().numpy().transpose(1,2,0))

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, self.img_size, self.org_img_size)
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(self.colors, n_cls_preds)
            # for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

            #     # print("\t+ Label: %s, Conf: %.5f" % (self.classes[int(cls_pred)], cls_conf.item()))

            #     box_w = x2 - x1
            #     box_h = y2 - y1

            #     color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            #     # Create a Rectangle patch
            #     bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                
            #     # Add the bbox to the plot
            #     # ax.add_patch(bbox)
            #     # Add label
            #     # plt.text(
            #     #     x1,
            #     #     y1,
            #     #     s=self.classes[int(cls_pred)],
            #     #     color="white",
            #     #     verticalalignment="top",
            #     #     bbox={"color": color, "pad": 0})
            # # plt.close()
            # return bbox
            return detections.detach().numpy()