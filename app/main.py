from __future__ import division


from PyTorch_YOLOv3.utils.utils import *
from PyTorch_YOLOv3.utils.datasets import *
from PyTorch_YOLOv3.models import *
from PyTorch_YOLOv3.utils.parse_config import *

from VertebraSegmentation.coordinate import *
from VertebraSegmentation.filp_and_rotate import *
from VertebraSegmentation.net.data import VertebraDataset
from VertebraSegmentation.net.model.unet import Unet
from VertebraSegmentation.net.model.resunet import ResidualUNet

import cv2
import math
import os 
import sys
import time
import datetime
import argparse
import shutil

from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, binary_erosion, square
from skimage.filters import sobel
from skimage.color import gray2rgb
from skimage import morphology

from os.path import splitext, exists, join
from PIL import Image

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from tqdm import tqdm
from skimage import io
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

import os

Max_H = 110
Max_W = 190

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

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

def delete_gap_dir(dir):
    if os.path.isdir(dir):
        for d in os.listdir(dir):
            delete_gap_dir(os.path.join(dir, d))
    if not os.listdir(dir):
        os.rmdir(dir)
    delete_gap_dir(os.getcwd())

def detect_one(img_path, modelSelcet="PyTorch_YOLOv3/checkpoints/yolov3_ckpt_best_f01.pth"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default=f"{img_path}", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="PyTorch_YOLOv3/config/yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--class_path", type=str, default="PyTorch_YOLOv3/data/custom/classes.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold") # 0.8
    parser.add_argument("--nms_thres", type=float, default=0.3, help="iou thresshold for non-maximum suppression") # 0.25
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, default=f"{modelSelcet}", help="path to checkpoint model")
    opt = parser.parse_args()

    print(opt.checkpoint_model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("./output", exist_ok=True)
    os.makedirs("./coordinate", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
    # Load checkpoint weights
    model.load_state_dict(torch.load(opt.checkpoint_model))
    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    print(opt.image_path + ".png")

    img = cv2.imread(opt.image_path + ".png", 0)
    
    img = clahe_hist(img)
    img = preprocess(img/255)
    img = transforms.ToTensor()(img).float()
    img, _ = pad_to_square(img, 0)
    img = resize(img, opt.img_size)

    print("\nPerforming object detection:")

    input_imgs = Variable(img.type(Tensor)).unsqueeze(0)
    
    detections = None
    with torch.no_grad():
        detections = model(input_imgs)
        detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

    plt.set_cmap('gray')
    rewrite = True
    
    img = np.array(Image.open(img_path + ".png").convert('L'))
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    # print(img.shape)
    filename = img_path[-4:]
    
    if detections is not None:
        # Rescale boxes to original image
        detections = rescale_boxes(detections[0], opt.img_size, img.shape[:2])
              
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            box_w = x2 - x1 
            box_h = y2 - y1 
            x1, y1, x2, y2 = math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)
            box_w, box_h = x2-x1, y2-y1

            if rewrite:
                f1 = open(f"./coordinate/{filename}.txt", 'w')
                f1.write("{:d} {:d} {:d} {:d} {:d} {:d}\n".format(x1, y1, x2, y2, box_w, box_h) )
                rewrite = False
            else:
                f1 = open(f"./coordinate/{filename}.txt", 'a')
                f1.write("{:d} {:d} {:d} {:d} {:d} {:d}\n".format(x1, y1, x2, y2, box_w, box_h) )

            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=0.5, edgecolor='red', facecolor="none")
            ax.add_patch(bbox)

    plt.axis("off")
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    plt.savefig(f"./output/{filename}.png", bbox_inches="tight", pad_inches=0.0, facecolor="none")
    plt.close()
    print("\nImage has saved")

    f1.close()
    path1 = join("./coordinate", filename)
    path2 = join("./GT_coordinate", filename)
    
    Sort_coordinate(f"{path1}.txt", flag=True)
    Sort_coordinate(f"{path2}.txt", flag=False)
def detect():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="PyTorch_YOLOv3/data/custom/images/valid/", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="PyTorch_YOLOv3/config/yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--class_path", type=str, default="PyTorch_YOLOv3/data/custom/classes.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold") # 0.8
    parser.add_argument("--nms_thres", type=float, default=0.3, help="iou thresshold for non-maximum suppression") # 0.25
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, default="PyTorch_YOLOv3/checkpoints/yolov3_ckpt_best_f01.pth", help="path to checkpoint model")
    opt = parser.parse_args()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("PyTorch_YOLOv3/output", exist_ok=True)
    os.makedirs("PyTorch_YOLOv3/pre_img", exist_ok=True)
    os.makedirs("PyTorch_YOLOv3/coordinate", exist_ok=True)

    fname_list = []
    for file in os.listdir(opt.image_folder):
        
        file_name = splitext(file)[0]
        fname_list.append(f"{file_name}.txt")

    fname_list = sorted(fname_list)
    
    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
    # Load checkpoint weights
    model.load_state_dict(torch.load(opt.checkpoint_model))
    
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
    
    for batch_i, (img_paths, input_imgs) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Batch Inference Time"):
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
        # print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)
    


    plt.set_cmap('gray')

    rewrite = True
    print("\nSaving images:")


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

            rewrite = True
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                # print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
                # x1 = x1 - 10 
                y1 = y1 - 5
                y2 = y2 + 5 
                x1 = x1 - 50
                x2 = x2 + 50
                box_w = x2 - x1 
                box_h = y2 - y1 
                x1, y1, x2, y2 = math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)
                box_w, box_h = x2-x1, y2-y1
                
                
                if rewrite:
                    f1 = open(f"VertebraSegmentation/coordinate/{fname_list[img_i]}", 'w')
                    f2 = open(f"PyTorch_YOLOv3/coordinate/{fname_list[img_i]}", 'w') 
                    f1.write("{:d} {:d} {:d} {:d} {:d} {:d}\n".format(x1, y1, x2, y2, box_w, box_h) )
                    f2.write("{:d} {:d} {:d} {:d} {:d} {:d}\n".format(x1, y1, x2, y2, box_w, box_h) )
                    rewrite = False
                else:
                    f1 = open(f"VertebraSegmentation/coordinate/{fname_list[img_i]}", 'a')
                    f2 = open(f"PyTorch_YOLOv3/coordinate/{fname_list[img_i]}", 'a') 
                    f1.write("{:d} {:d} {:d} {:d} {:d} {:d}\n".format(x1, y1, x2, y2, box_w, box_h) )
                    f2.write("{:d} {:d} {:d} {:d} {:d} {:d}\n".format(x1, y1, x2, y2, box_w, box_h) )
                # color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=0.5, edgecolor='red', facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        # plt.set_cmap('gray')
        filename = path.split("/")[-1].split(".")[0]
        plt.savefig(f"PyTorch_YOLOv3/output/{filename}.png", bbox_inches="tight", pad_inches=0.0, facecolor="none")
        plt.close()

    # print(f"Max_Height={Max_H}", f"Max_Width={Max_W}")

def create_valid_return_len(coordinate_path, save_path, source_path):

    os.makedirs(save_path, exist_ok=True)
    os.makedirs("VertebraSegmentation/test/valid", exist_ok=True)

    box_num = []
    
    for file in tqdm(sorted(listdir(source_path)), desc=f"{save_path}"): 
        img = cv2.imread(join(source_path, file), 0)
        boxes = get_information(file, coordinate_path)
        
        box_num.append(len(boxes))
        
        for id, box in enumerate(boxes):
            box = list(map(int, box))
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            width, height = box[4], box[5]
            detect_region = np.zeros((height, width))
            detect_region = img[y1:y2+1, x1:x2+1]
            
            cv2.imwrite(join("VertebraSegmentation", "valid_data", f"{splitext(file)[0]}_{id}.png"), detect_region)

    return box_num

def normalize_data(output, threshold=0.6):
    
    return np.where(output > threshold, 1, 0)


def predict(model, loader, numOfEachImg, save_path="VertebraSegmentation//test//valid"):

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    result = np.zeros((1, 1200, 500), dtype=int)
    
    count = 0
    with torch.no_grad():
        for _, (img, filename) in tqdm(enumerate(loader), total=len(loader), desc="Predict"):
            
            index = int((splitext(filename[0])[0])[:4]) - 1 # 0, 1, 2, 3, ... , 19
            id = int((splitext(filename[0])[0])[5:])
            count += 1
                       
            img = img.to(device)
            output = model(img)    
            output = torch.sigmoid(output)[0, :]
            output = (normalize_data(output.cpu().numpy())*255).astype(np.uint8)    

            boxes = get_information(f"{(splitext(filename[0])[0])[:4]}.png", "VertebraSegmentation/coordinate")          

            box = boxes[id]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            
            result[:, y1:y2+1, x1:x2+1] = output

            if count == numOfEachImg[index]:
                result = result.astype(np.uint8)
                for dim in range(result.shape[0]):
                    io.imsave(f"{save_path}//p{(splitext(filename[0])[0])[:4]}.png", result[dim])   

                result = np.zeros((1, 1200, 500), dtype=int)
                count = 0

def dice_coef(target, truth, empty=False, smooth=1.0):
    if not empty:
        target = target.flatten()
        truth = truth.flatten()
        union = np.sum(target) + np.sum(truth)
        intersection = np.sum(target * truth)      
        dice = (2 * intersection + smooth) / (union + smooth)
        return dice
    else:
        print("aaaa")
        return 0


def create_valid_return_len_one(coordinate_path, save_path, source_path, target):

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(source_path, exist_ok=True)
    
    img = cv2.imread(join(source_path, target), 0)
    boxes = get_information(target, coordinate_path)
    
    for id, box in enumerate(boxes):
        box = list(map(int, box))
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        width, height = box[4], box[5]
        detect_region = np.zeros((height, width))
        detect_region = img[y1:y2+1, x1:x2+1]

        cv2.imwrite(join(save_path, f"{splitext(target)[0]}_{id}.png"), detect_region)

    return len(boxes)    
    
def predict_one(model, loader, numOfImg, img, save_path=".//result"):

    os.makedirs(save_path, exist_ok=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = np.zeros((1, 1200, 500), dtype=int)

    GT_path = join("./source/label", img)
    label_img = cv2.imread(GT_path, 0) # 2-dimension
    Dice_list = [None for _ in range(20)]
    bone_num = 0
    
    GT_id = 0
    id = 0
    with torch.no_grad():
        for _, (img, filename) in tqdm(enumerate(loader), total=len(loader), desc="Predict"):
            id = int((splitext(filename[0])[0])[5:])  
            img = img.to(device) # img is 4-dimension
            output = model(img)    
            output = torch.sigmoid(output)[0, :]
            output = (normalize_data(output.cpu().numpy())*255).astype(np.uint8)    
            boxes = get_information(f"{(splitext(filename[0])[0])[:4]}.png", "./coordinate")    

            output = (output==255).astype(bool)
            output = morphology.remove_small_objects(output, min_size=2000, connectivity=2)
            output = output.astype(int)
            output = np.where(output==1, 255, 0)

            box = boxes[id]

            # print(id)

            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])       
            result[:, y1:y2+1, x1:x2+1] = output
            
            GT_boxes = get_information(f"{(splitext(filename[0])[0])[:4]}.png", "./GT_coordinate")

            flag = True
        
            while flag:
                GT_box = GT_boxes[GT_id]
                GT_x1, GT_y1, GT_x2, GT_y2 = int(GT_box[0]), int(GT_box[1]), int(GT_box[2]), int(GT_box[3])  
                
                result_label = label_img[GT_y1: GT_y2+1, GT_x1: GT_x2+1]
                Dice = 0
                if (abs(y1 - GT_y1) > 35 and len(boxes) != len(GT_boxes)):
                    Dice = dice_coef(None, None, empty=True)
                else:
                    # intersection_x1 = min(x1, GT_x1)
                    # intersection_y1 = min(y1, GT_y1)
                    # intersection_x2 = max(x2, GT_x2)
                    # intersection_y2 = max(y2, GT_y2)
                    # intersection_h = intersection_y2-intersection_y1+1
                    # intersection_w = intersection_x2-intersection_x1+1

                    intersection_result = np.zeros((1200, 500), dtype=int)
                    intersection_result_label = np.zeros((1200, 500), dtype=int)
                    
                    intersection_result[y1:y2+1, x1:x2+1] = result[:, y1:y2+1, x1:x2+1]
                    intersection_result_label[GT_y1:GT_y2+1, GT_x1:GT_x2+1] = result_label

                    Dice = dice_coef(intersection_result/255, intersection_result_label/255)
                    Dice = round(float(Dice), 2)
                    id += 1
                    flag = False
                
                Dice_list[GT_id] = Dice
                GT_id += 1
                
            
            
            # result_label = np.zeros((y2-y1, x2-x1), dtype=int)
            # result_label = label_img[y1:y2+1, x1:x2+1]

            # Dice = dice_coef(result[0, y1:y2+1, x1:x2+1]/255, result_label/255)
            # Dice = round(float(Dice), 2)
            # Dice_list[id] = Dice
            
            bone_num += 1

            if _+1 == numOfImg:
                result = result.astype(np.uint8)
                for dim in range(result.shape[0]):
                    io.imsave(f"{save_path}//p{(splitext(filename[0])[0])[:4]}.png", result[dim])   


    return Dice_list, bone_num
    
def Segmentation_one(target):

    # create splitting dataset
    numOfImg = create_valid_return_len_one("./coordinate", 
                                           "./valid_data",
                                           "./source/image",
                                           target) # target must be xxxx.png
    # recombine each sepertated img 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = VertebraDataset(".//valid_data//")
    model = ResidualUNet(n_channels=1, n_classes=1)
    checkpoint = torch.load("VertebraSegmentation/net/save/best_detect_f03.pt")
    model.load_state_dict(checkpoint["state_dict"])
    loader = DataLoader(dataset, batch_size=1)
    model = model.to(device)
    return predict_one(model, loader, numOfImg, target) # return Dice_list and bone_num
    print("Done.")

def Segmentation():

    # create splitting dataset
    numOfEachImg = create_valid_return_len("VertebraSegmentation/coordinate", 
                                           "VertebraSegmentation/valid_data",
                                           "VertebraSegmentation/original_data/f01/image")
    
    # recombine each sepertated img 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = VertebraDataset("VertebraSegmentation//valid_data//")
    model = ResidualUNet(n_channels=1, n_classes=1)
    checkpoint = torch.load("VertebraSegmentation//net//save//best_test_f01.pt")
    # checkpoint = torch.load("save//last_detect.pt")

    model.load_state_dict(checkpoint["state_dict"])
    loader = DataLoader(dataset, batch_size=1)
    model = model.to(device)
    
    predict(model, loader, numOfEachImg)
    print("Done.")


def delete_valid_data(path):
    
    try:
        shutil.rmtree(path)
    except OSError as e:
        print(e)
    else:
        print("The directory is deleted successfully")

def Sort_coordinate(path, flag):
    # path = join("./coordinate", filename)
        
    f = open(path, 'r')
    lines = f.readlines()
    f.close()
    list_of_list = []
    
    for i, line in enumerate(lines):
        lines[i] = line.replace("\n", "")
        L = list(map(lambda x: int(x), lines[i].split(' ')))
        list_of_list.append(L)

    list_of_list.sort(key=lambda x: x[1])
    
    f1 = open(path, 'w')
    f2 = open(path, 'a')

    for i, line in enumerate(list_of_list):
        
        if flag:
            if i == 0:
                f1.write("{:d} {:d} {:d} {:d} {:d} {:d}\n".format(line[0], line[1], line[2], line[3], line[4], line[5]) )
                f1.close()
            else:
                f2.write("{:d} {:d} {:d} {:d} {:d} {:d}\n".format(line[0], line[1], line[2], line[3], line[4], line[5]) )
        else:
            if i == 0:
                f1.write("{:d} {:d} {:d} {:d}\n".format(line[0], line[1], line[2], line[3]) )
                f1.close()
            else:
                f2.write("{:d} {:d} {:d} {:d}\n".format(line[0], line[1], line[2], line[3]) )
    f2.close()
    
    return len(list_of_list)

def main():
    delete_valid_data(r"VertebraSegmentation/valid_data")
    # create coordinate of each boxes
    detect()
    Segmentation()

if __name__ == "__main__":

    main()
    