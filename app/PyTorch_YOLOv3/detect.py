from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os 
import sys
import time
import datetime
import argparse

from os.path import splitext, exists, join
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

def preprocess(img):
        img = rgb2gray(img)
        bound = img.shape[0] // 3
        up = exposure.equalize_adapthist(img[:bound, :])
        down = exposure.equalize_adapthist(img[bound:, :])
        enhance = np.append(up, down, axis=0)
        edge = sobel(gaussian(enhance, 2))
        enhance = enhance + edge * 3
        return np.where(enhance > 1, 1, enhance)

def clahe_hist(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
    return cl1

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    # parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    # parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    # parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    # parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    # parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    # parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    # parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    # parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    # parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/custom/images/valid/", help="path to dataset")
    # parser.add_argument("--image_folder", type=str, default="pre_img/", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_1000.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/custom/classes.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold") # 0.8
    parser.add_argument("--nms_thres", type=float, default=0.3, help="iou thresshold for non-maximum suppression") # 0.25
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, default="checkpoints/yolov3_ckpt_best_f01.pth", help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("pre_img", exist_ok=True)
    os.makedirs("coordinate", exist_ok=True)

    fname_list = []

    for file in os.listdir(opt.image_folder):
        
        # img_name = "data/custom/images/valid/" + file
        # img = cv2.imread(img_name, 0)
        # img = clahe_hist(img)
        # img = preprocess(img/255) * 255
        # cv2.imwrite(f"pre_img/{file}", img)
        file_name = splitext(file)[0]
        fname_list.append(f"{file_name}.txt")

    fname_list = sorted(fname_list)
    
    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    
    # Load checkpoint weights
    model.load_state_dict(torch.load(opt.checkpoint_model))
    
    # if opt.weights_path.endswith(".weights"):
    #     # Load darknet weights
    #     model.load_darknet_weights(opt.checkpoint_model)
    # else:
    #     # Load checkpoint weights
    #     model.load_state_dict(torch.load(opt.checkpoint_model))

    model.eval()  # Set in evaluation mode

    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        
        input_imgs = Variable(input_imgs.type(Tensor))
        
        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    # Bounding-box colors
    # cmap = plt.get_cmap("tab20b")
    plt.set_cmap('gray')
    # colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    
    rewrite = True

    print("\nSaving images:")
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        print("(%d) Image: '%s'" % (img_i, path))

        # Create plot
        
        img = np.array(Image.open(path).convert('L'))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
            # unique_labels = detections[:, -1].cpu().unique()
            # n_cls_preds = len(unique_labels)
            # bbox_colors = random.sample(colors, n_cls_preds)
            rewrite = True
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
    
                # y1 = y1 - 25
                # y2 = y2 + 25
                box_w = x2 - x1 
                box_h = y2 - y1 
                x1, y1, x2, y2 = math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)
                box_w, box_h = x2-x1, y2-y1
                if rewrite:
                    f1 = open(f"../VertebraSegmentation/coordinate/{fname_list[img_i]}", 'w')
                    f2 = open(f"coordinate/{fname_list[img_i]}", 'w') 
                    # f.write(f"{x1} {y1} {x2} {y2} {box_w} {box_h}\n")
                    f1.write("{:d} {:d} {:d} {:d} {:d} {:d}\n".format(x1, y1, x2, y2, box_w, box_h) )
                    f2.write("{:d} {:d} {:d} {:d} {:d} {:d}\n".format(x1, y1, x2, y2, box_w, box_h) )
                    rewrite = False
                else:
                    f1 = open(f"../VertebraSegmentation/coordinate/{fname_list[img_i]}", 'a')
                    f2 = open(f"coordinate/{fname_list[img_i]}", 'a') 
                    # f.write(f"{x1} {y1} {x2} {y2} {box_w} {box_h}\n")
                    f1.write("{:d} {:d} {:d} {:d} {:d} {:d}\n".format(x1, y1, x2, y2, box_w, box_h) )
                    f2.write("{:d} {:d} {:d} {:d} {:d} {:d}\n".format(x1, y1, x2, y2, box_w, box_h) )
                # color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=0.5, edgecolor='red', facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                # plt.text(
                #     x1,
                #     y1,
                #     s=classes[int(cls_pred)],
                #     color="white",
                #     verticalalignment="top",
                #     bbox={"color": color, "pad": 0},
                # )

        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        # plt.set_cmap('gray')
        filename = path.split("/")[-1].split(".")[0]
        plt.savefig(f"output/{filename}.png", bbox_inches="tight", pad_inches=0.0, facecolor="none")
        plt.close()
