from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor
from skimage.filters import gaussian, sobel
from skimage.color import rgb2gray
from skimage import exposure, io
from os import listdir, path
import torch
import numpy as np
import cv2
from os.path import splitext, exists, join

codebook = {(0, 0, 0): 0, (0, 8, 255): 1, (0, 93, 255): 2, (0, 178, 255): 3, (0, 255, 77): 4, (0, 255, 162): 5, 
(0, 255, 247): 6, (8, 255, 0): 7, (77, 0, 255): 8, (93, 255, 0): 9, (162, 0, 255): 10, (178, 255, 0): 11, 
(247, 0, 255): 12, (255, 0, 8): 13, (255, 0, 93): 14, (255, 0, 178): 15, (255, 76, 0): 16, 
(255, 162, 0): 17, (255, 247, 0): 18}

class VertebraDataset(Dataset):
    def __init__(self, dataset_path, train=False, image_folder="image", mask_folder="label"):
        self.train = train
        self.dataset_path = dataset_path
        if self.train:
            self.image_folder = image_folder
            self.images = sorted(listdir(path.join(dataset_path, image_folder)))
            self.mask_folder = mask_folder
            self.masks = sorted(listdir(path.join(dataset_path, mask_folder)))
        else:
            self.images = sorted(listdir(path.join(dataset_path)), key=lambda x: int((splitext(x)[0])[5:]))
        self.transform = Compose([ToTensor()])
        
    def __getitem__(self, idx):
        if self.train:
            img_path = path.join(self.dataset_path, self.image_folder, self.images[idx])
            # img = io.imread(img_path)
            img = cv2.imread(img_path, 0)
            img = self.clahe_hist(img)
            img = img / 255
            # img = self.preprocess(img)

            out_img = np.zeros((1,) + img.shape, dtype=np.float)
            out_img[:, ] = img
            mask_path = path.join(self.dataset_path, self.mask_folder, self.masks[idx])
            
            ######################### original #########################

            mask = np.array(rgb2gray(io.imread(mask_path))) / 255  

            ######################### original #########################

            ######################### color #########################

            # mask = cv2.imread(mask_path, 1)
            # mask = self.rgb2label(mask, color_codes=codebook, one_hot_encode=True)

            ######################### color #########################
            
            return torch.as_tensor(out_img, dtype=torch.float), torch.as_tensor(mask, dtype=torch.float)
        else:
            img_path = path.join(self.dataset_path, self.images[idx])
            # img = io.imread(img_path)
            img = cv2.imread(img_path, 0)
            img = self.clahe_hist(img)
            img = img / 255
            # img = self.preprocess(img) 
            out_img = np.zeros((1,) + img.shape, dtype=np.float)
            out_img[:, ] = img
            return torch.as_tensor(out_img, dtype=torch.float), self.images[idx]

    def __len__(self):
        return len(self.images)

    @classmethod
    def preprocess(cls, img):
        img = rgb2gray(img)
        bound = img.shape[0] // 3
        up = exposure.equalize_adapthist(img[:bound, :])
        down = exposure.equalize_adapthist(img[bound:, :])
        enhance = np.append(up, down, axis=0)
        edge = sobel(gaussian(enhance, 2))
        enhance = enhance + edge * 3
        return np.where(enhance > 1, 1, enhance)
    @classmethod
    def clahe_hist(cls, img):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl1 = clahe.apply(img)
        return cl1
    @classmethod
    def rgb2label(cls, img, color_codes = None, one_hot_encode=False):
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
            # one_hot_labels = np.zeros((img.shape[0],img.shape[1],n_labels))
            one_hot_labels = np.zeros((n_labels, img.shape[0],img.shape[1]))
            # one-hot encoding
            for c in range(n_labels):
                # one_hot_labels[ :, : , c ] = (result == c ).astype(int)
                one_hot_labels[ c , : , : ] = (result == c ).astype(int)
            result = one_hot_labels

        # return result, sort_color_codes
        return result

if __name__ == '__main__':
    
    dataset = VertebraDataset("..//..//extend_dataset", train=True)
    
    a, b = dataset[0]
    print(a.shape)
    print(b.shape)
