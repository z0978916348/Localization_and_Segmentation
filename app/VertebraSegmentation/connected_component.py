
from torch.nn import Module, CrossEntropyLoss
import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor
from skimage.filters import gaussian, sobel
from skimage.color import rgb2gray
from skimage import exposure, io
from os import listdir, path
import torch
import numpy as np
from skimage.transform import rescale, resize, downscale_local_mean
import math
from os.path import splitext, exists, join
import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings
from os import mkdir, listdir
from os.path import splitext, exists, join
from tqdm import tqdm
warnings.filterwarnings('ignore')


PATH_LABEL = [join("label_data", "f01"), join("label_data", "f02"), join("label_data", "f03")]
SOURCE_DIR = "original_data"
SOURCE_SUB_DIR = ["f01", "f02", "f03"]
SUB_DIR = ["image", "label"]
TARGET_DIR = "label_data"

codebook = {0:(0, 0, 0), 1:(0, 8, 255), 2:(0, 93, 255), 3:(0, 178, 255), 4:(0, 255, 77), 5:(0, 255, 162), 
6:(0, 255, 247), 7:(8, 255, 0), 8:(77, 0, 255), 9:(93, 255, 0), 10:(162, 0, 255), 11:(178, 255, 0), 
12:(247, 0, 255), 13:(255, 0, 8), 14:(255, 0, 93), 15:(255, 0, 178), 16:(255, 76, 0), 
17:(255, 162, 0), 18:(255, 247, 0)}

def connected_component_label(path):
    
    # Getting the input image
    img = cv2.imread(path, 0)
    # Converting those pixels with values 1-127 to 0 and others to 1
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
    # Applying cv2.connectedComponents() 
    num_labels, labels = cv2.connectedComponents(img)
    

    test = np.zeros((labels.shape[0], labels.shape[1], 3))
    
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            test[i][j] = codebook[labels[i][j]]
    return cv2.cvtColor(test.astype('float32'), cv2.COLOR_BGR2RGB)

    # # Map component labels to hue val, 0-179 is the hue range in OpenCV
    # label_hue = np.uint8(179*labels/np.max(labels))
    # blank_ch = 255*np.ones_like(label_hue)
    # labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # # Converting cvt to BGR
    # labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # # set bg label to black
    # labeled_img[label_hue==0] = 0
        
    # # Showing Original Image
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # # plt.axis("off")
    # # plt.title("Orginal Image")
    # # plt.show()
    
    # #Showing Image after Component Labeling
    # # plt.imshow(cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB))
    # # plt.axis('off')
    # # plt.title("Image after Component Labeling")
    # # plt.show()

    # return cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB)

def rgb2label(img, color_codes = None, one_hot_encode=False):
    if color_codes is None:
        color_codes = {val:i for i,val in enumerate(set( tuple(v) for m2d in img for v in m2d ))}
    n_labels = len(color_codes)
    result = np.ndarray(shape=img.shape[:2], dtype=int)
    result[:,:] = -1
    
    color_codes = sorted(color_codes)
    sort_color_codes = dict()
    for idx, rgb in enumerate(color_codes):
        result[(img==rgb).all(2)] = idx
        sort_color_codes[rgb] = idx
    
    # for rgb, idx in color_codes.items():
    #     result[(img==rgb).all(2)] = idx

    if one_hot_encode:
        one_hot_labels = np.zeros((img.shape[0],img.shape[1],n_labels))
        # one-hot encoding
        for c in range(n_labels):
            one_hot_labels[: , : , c ] = (result == c ).astype(int)
        result = one_hot_labels

    # return result, sort_color_codes
    return result

# def Labeling(img_path):
    
#     img = connected_component_label(img_path)
#     img_labels, color_codes = rgb2label(img, one_hot_encode=True)

#     return img_labels


if __name__ == '__main__':

    for path in PATH_LABEL:
        if not exists(join(path, SUB_DIR[1])):
            mkdir(join(path, SUB_DIR[1]))
    
    
    for src_subdir in SOURCE_SUB_DIR:
        for file in tqdm(listdir(join(SOURCE_DIR, src_subdir, SUB_DIR[1])), desc=f"{SOURCE_DIR}/{src_subdir}/{SUB_DIR[1]}"):     
            path = join(SOURCE_DIR, src_subdir, SUB_DIR[1], f"{splitext(file)[0]}.png")
            colorful_img = connected_component_label(path)
            cv2.imwrite(join(TARGET_DIR, src_subdir, SUB_DIR[1], f"{splitext(file)[0]}.png"), colorful_img)

    print("Generate labeling data !")
