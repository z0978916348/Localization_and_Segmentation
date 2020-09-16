from os import mkdir, listdir
from os.path import splitext, exists, join
from PIL import Image, ImageOps
from tqdm import tqdm
import torchvision.transforms.functional as F
import cv2
import numpy as np
import os 
import math
import shutil
from .coordinate import get_information

SOURCE_DIR = [join("original_data", "f01"), join("original_data", "f02")]
LABELING_SOURCE_DIR = [join("label_data", "f02"), join("label_data", "f03")]
OBJECT_SOURCE_DIR = [join("detect_data", "f02"), join("detect_data", "f03")]

TARGET_DIR = "extend_dataset"
LABELING_TARGET_DIR = "label_extend_dataset"
OBJECT_DIR = "extend_detect_data"

SUB_DIR = ["image", "label"]

SPLIT_DIR = 'splitting_dataset'

ORIGINAL_SPLIT = "original_split_data"
ORIGINAL_SPLIT_DATA = [join("original_split_data", "f01"), join("original_split_data", "f02"), join("original_split_data", "f03")]

ORIGINAL_SRC = "original_data"

OBJECT_DETECTION_LABEL = "object_detect_label"
DETECTION_DATA = "detect_data"
F_SUBDIR = ["f01", "f02", "f03"]
# ORIGINAL_SOURCE_DIR = [join("original_data", "f01"), join("original_data", "f02"), join("original_data", "f03")]



ROTATION_ANGLE = [180]


Max_H = 110
Max_W = 190
def delete_dir(path):
    
    try:
        shutil.rmtree(path)
    except OSError as e:
        print(e)
    else:
        print("The directory is deleted successfully")

def contrast_img(img1, c, b):  
    rows, cols, channels = img1.shape

    blank = np.zeros([rows, cols, channels], img1.dtype)
    dst = cv2.addWeighted(img1, c, blank, 1-c, b)
    # cv2.imshow('original_img', img1)
    # cv2.imshow("contrast_img", dst)
    return dst

def create_valid_return_len(dir, save_path, source_path):

    os.makedirs(save_path, exist_ok=True)
    os.makedirs("test/valid", exist_ok=True)

    box_num = []
    
    for file in tqdm(sorted(listdir(join(ORIGINAL_SRC, source_path))), desc=f"{ORIGINAL_SRC}//{source_path}"): # .png
        img = cv2.imread(join(ORIGINAL_SRC, source_path, file), 0)
        boxes = get_information(file, dir)
        
        box_num.append(len(boxes))
        
        for id, box in enumerate(boxes):
            box = list(map(int, box))
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            width, height = box[4], box[5]
            detect_region = np.zeros((height, width))
            detect_region = img[y1:y2+1, x1:x2+1]
                    
            print(detect_region)

            cv2.imwrite(join(save_path, f"{splitext(file)[0]}_{id}.png"), detect_region)

    return box_num


if __name__ == '__main__':

    # if not exists(TARGET_DIR):
    #     mkdir(TARGET_DIR)
    # for sub_dir in SUB_DIR:
    #     dir = join(TARGET_DIR, sub_dir)
    #     if not exists(dir):
    #         mkdir(dir)

    
    # for source in SOURCE_DIR:
    #     for sub_dir in SUB_DIR:
    #         for file in tqdm(listdir(join(source, sub_dir)), desc=f"{source}//{sub_dir}"):
    #             img = Image.open(join(source, sub_dir, file))
    #             img.save(join(TARGET_DIR, sub_dir, f"{splitext(file)[0]}_0.png"))
    #             for angle in ROTATION_ANGLE:
    #                 img.rotate(angle, expand=True).save(join(TARGET_DIR, sub_dir, f"{splitext(file)[0]}_{angle}.png"))
    #             img = ImageOps.mirror(img)
    #             img.save(join(TARGET_DIR, sub_dir, f"{splitext(file)[0]}_f0.png"))
    #             for angle in ROTATION_ANGLE:
    #                 img.rotate(angle, expand=True).save(join(TARGET_DIR, sub_dir, f"{splitext(file)[0]}_f{angle}.png"))

    #######################################################################################################################
    # # for labeling dataset
    # if not exists(LABELING_TARGET_DIR):
    #     mkdir(LABELING_TARGET_DIR)
    # for sub_dir in SUB_DIR:
    #     dir = join(LABELING_TARGET_DIR, sub_dir)
    #     if not exists(dir):
    #         mkdir(dir)

    # for source in LABELING_SOURCE_DIR:
    #     for sub_dir in SUB_DIR:
    #         for file in tqdm(listdir(join(source, sub_dir)), desc=f"{source}//{sub_dir}"):
    #             img = Image.open(join(source, sub_dir, file))
    #             img.save(join(LABELING_TARGET_DIR, sub_dir, f"{splitext(file)[0]}_0.png"))
    #             for angle in ROTATION_ANGLE:
    #                 img.rotate(angle, expand=True).save(join(LABELING_TARGET_DIR, sub_dir, f"{splitext(file)[0]}_{angle}.png"))
    #             img = ImageOps.mirror(img)
    #             img.save(join(LABELING_TARGET_DIR, sub_dir, f"{splitext(file)[0]}_f0.png"))
    #             for angle in ROTATION_ANGLE:
    #                 img.rotate(angle, expand=True).save(join(LABELING_TARGET_DIR, sub_dir, f"{splitext(file)[0]}_f{angle}.png"))

    ########################################################################################################################
    # for object detect splitting dataset
    os.makedirs(OBJECT_DETECTION_LABEL, exist_ok=True)
    os.makedirs(DETECTION_DATA, exist_ok=True)


    for sub in F_SUBDIR:
        # os.makedirs(join(OBJECT_DETECTION_LABEL, sub), exist_ok=True)
        os.makedirs(join(DETECTION_DATA, sub), exist_ok=True)
        for sub_dir in SUB_DIR:
            os.makedirs(join(DETECTION_DATA, sub, sub_dir), exist_ok=True)

    for source in F_SUBDIR: # f01, f02, f03
        for sub_dir in SUB_DIR: # image, label
            for file in tqdm(listdir(join(ORIGINAL_SRC, source, sub_dir)), desc=f"{ORIGINAL_SRC}//{source}//{sub_dir}"): # .png
                img = cv2.imread(join(ORIGINAL_SRC, source, sub_dir, file), 0)
                boxes = get_information(file, "object_detect_label")
                
                for id, box in enumerate(boxes):
                    box = list(map(float, box))
                    x1, y1, x2, y2 = math.floor(box[0]), math.floor(box[1]), math.ceil(box[2]), math.ceil(box[3])
                    x1 = x1-50
                    x2 = x2+50
                    width, height = x2-x1, y2-y1
                    
                    ##############################################################################
                    # if width < Max_W:
                    #     remain = Max_W - width
                    #     halfWidth, theOtherHalfWidth = remain//2, remain - remain//2
                    #     width = Max_W

                    # if height < Max_H:   
                    #     remain = Max_H - height               
                    #     halfHeight, theOtherHalfHeight = remain//2, remain - remain//2
                    #     box_h = Max_H
                    
                    # x1 = x1 - halfWidth
                    # x2 = x2 + theOtherHalfWidth
                    # y1 = y1 - halfHeight
                    # y2 = y2 + theOtherHalfHeight

                    # x1 = 0 if x1 < 0 else x1 
                    # y1 = 0 if y1 < 0 else y1 
                    # x2 = 499 if x2 > 499 else x2 
                    # y2 = 1199 if y2 > 1199 else y2 
                    ################################################################################

                    detect_region = np.zeros((height, width))
                    detect_region = img[y1:y2+1, x1:x2+1]
                    
                    
                    cv2.imwrite(join(DETECTION_DATA, source, sub_dir, f"{splitext(file)[0]}_{id}.png"), detect_region)

    
    # extend object detect dataset

    

    OBJ_EXTEND = "extend_detect_data"

    os.makedirs(OBJ_EXTEND, exist_ok=True)

    for sub_dir in SUB_DIR:
        dir = join(OBJ_EXTEND, sub_dir)
        delete_dir(dir)
        os.makedirs(dir, exist_ok=True)

    for source in OBJECT_SOURCE_DIR:
        for sub_dir in SUB_DIR:
            for file in tqdm(listdir(join(source, sub_dir)), desc=f"{source}//{sub_dir}"):
                img = Image.open(join(source, sub_dir, file))
                img.save(join(OBJECT_DIR, sub_dir, f"{splitext(file)[0]}_0.png"))
                for angle in ROTATION_ANGLE:
                    img.rotate(angle, expand=True).save(join(OBJECT_DIR, sub_dir, f"{splitext(file)[0]}_{angle}.png"))
                img = ImageOps.mirror(img)
                img.save(join(OBJECT_DIR, sub_dir, f"{splitext(file)[0]}_f0.png"))
                for angle in ROTATION_ANGLE:
                    img.rotate(angle, expand=True).save(join(OBJECT_DIR, sub_dir, f"{splitext(file)[0]}_f{angle}.png"))         


    

    # VALID_SOURCE = "coordinate"
    # VALID_DATA = "valid_data"
    # VALID_DIR = "f01/image"
    # os.makedirs(VALID_DATA, exist_ok=True)
    
    
    # for file in tqdm(listdir(join(ORIGINAL_SRC, VALID_DIR)), desc=f"{VALID_DATA}"): # .png
    #     img = cv2.imread(join(ORIGINAL_SRC, VALID_DIR, file), 0)
    #     boxes = get_information(file, OBJECT_DETECTION_LABEL)
        
    #     num = len(boxes)

    #     for id, box in enumerate(boxes):
    #         box = list(map(int, box))
    #         x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    #         width, height = box[4], box[5]
    #         detect_region = np.zeros((height, width))
    #         detect_region = img[y1:y2+1, x1:x2+1]
                    
                    
    #         cv2.imwrite(join(VALID_DATA, f"{splitext(file)[0]}_{id}.png"), detect_region)

    ######################################################################################################################

    # exist_split_extend = True

    # if not exists(SPLIT_DIR):
    #     mkdir(SPLIT_DIR)
    # for sub_dir in SUB_DIR:
    #     dir = join(SPLIT_DIR, sub_dir)
    #     if not exists(dir):
    #         mkdir(dir)
    #         exist_split_extend = False

    # if not exist_split_extend:
    #     for sub_dir in SUB_DIR:
    #         for file in tqdm(listdir(join(TARGET_DIR, sub_dir)), desc=f"{SPLIT_DIR}//{sub_dir}"):
                
    #             img = cv2.imread(join(TARGET_DIR, sub_dir, file), 0)
    #             Height, Width = img.shape
    #             gap = Height // 3
    #             truncated = 0

    #             for _ in range(3):
                    
    #                 split_img = img[truncated:truncated+gap, :]
    #                 truncated += gap
                    
    #                 if _ == 0:
    #                     cv2.imwrite(join(SPLIT_DIR, sub_dir, f"{splitext(file)[0]}_top.png"), split_img)
    #                 elif _ == 1:
    #                     cv2.imwrite(join(SPLIT_DIR, sub_dir, f"{splitext(file)[0]}_mid.png"), split_img)
    #                 else:
    #                     cv2.imwrite(join(SPLIT_DIR, sub_dir, f"{splitext(file)[0]}_bot.png"), split_img)

    # exist_split_original = True

    # if not exists(ORIGINAL_SPLIT):
    #     mkdir(ORIGINAL_SPLIT)

    # for sub_org in ORIGINAL_SPLIT_DATA:
    #     if not exists(sub_org):
    #         mkdir(sub_org)
    #     for sub_dir in SUB_DIR:
    #         dir = join(sub_org, sub_dir)
    #         if not exists(dir):
    #             mkdir(dir)
    #             exist_split_original = False
                
    # if not exist_split_original:
    #     for dir in ["f01", "f02", "f03"]:
    #         for sub_dir in SUB_DIR:    
    #             for file in tqdm(listdir(join(ORIGINAL_SRC, dir, sub_dir)), desc=f"{ORIGINAL_SPLIT}//{sub_dir}"):
    #                 img = cv2.imread(join(ORIGINAL_SRC, dir, sub_dir, file), 0)
            
    #                 Height, Width = img.shape
    #                 gap = Height // 3
    #                 truncated = 0

    #                 for _ in range(3):
                        
    #                     split_img = img[truncated:truncated+gap, :]
    #                     truncated += gap
                        
    #                     if _ == 0:
    #                         cv2.imwrite(join(ORIGINAL_SPLIT, dir, sub_dir, f"{splitext(file)[0]}_top.png"), split_img)
    #                     elif _ == 1:
    #                         cv2.imwrite(join(ORIGINAL_SPLIT, dir, sub_dir, f"{splitext(file)[0]}_mid.png"), split_img)
    #                     else:
    #                         cv2.imwrite(join(ORIGINAL_SPLIT, dir, sub_dir, f"{splitext(file)[0]}_bot.png"), split_img)

  