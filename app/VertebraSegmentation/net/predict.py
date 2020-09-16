from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage import io
from data import VertebraDataset
from model.unet import Unet
import torch
import torch.nn as nn
import numpy as np
import warnings
from model.unet import UNet
from model.resunet import ResidualUNet
import cv2
import math



warnings.filterwarnings('ignore')



codebook = {0:(0, 0, 0), 1:(0, 8, 255), 2:(0, 93, 255), 3:(0, 178, 255), 4:(0, 255, 77), 5:(0, 255, 162), 
6:(0, 255, 247), 7:(8, 255, 0), 8:(77, 0, 255), 9:(93, 255, 0), 10:(162, 0, 255), 11:(178, 255, 0), 
12:(247, 0, 255), 13:(255, 0, 8), 14:(255, 0, 93), 15:(255, 0, 178), 16:(255, 76, 0), 
17:(255, 162, 0), 18:(255, 247, 0)}


def normalize_data(output, threshold=0.6):
    
    return np.where(output > threshold, 1, 0)


############################# color ###################################
# def predict(model, loader, save_path="..//test//predict_color"):
############################# color ###################################

############################# original ###################################
# def predict(model, loader, save_path="..//test//predict"):
############################# original ###################################

############################# detect ###################################
def predict(model, loader, save_path="..//test//detect"):
# def predict(model, loader, save_path="..//test//valid"):
############################# detect ###################################
    model.eval()
    with torch.no_grad():
        for _, (img, filename) in tqdm(enumerate(loader), total=len(loader), desc="Predict"):
            img = img.to(device)
            output = model(img)            
            
            # output = (torch.softmax(output, dim=1)[:, 1])
            
            ############################# original ###################################
            
            output = torch.sigmoid(output)[0, :]
            output = (normalize_data(output.cpu().numpy())*255).astype(np.uint8)
            
            ############################# original ###################################

            ############################# color ###################################

            # # (19, 1200, 500)
            
            # # output = torch.sigmoid(output)
            # # output = output[0][0]
            # # output = (normalize_data(output.unsqueeze(0).cpu().numpy())*255).astype(np.uint8)
            
            # output = output.permute(0, 2, 3, 1) # (1200, 500, 19)
            # output = torch.sigmoid(output) 
            # output = torch.argmax(output, dim=3)
            # output = image_decode(output[0]).cpu().numpy().astype(np.uint8)

            ############################# color ###################################
            
            
            for dim in range(output.shape[0]):
                io.imsave(f"{save_path}//p{filename[dim]}", output[dim])


# def predict(model, loader, save_path="..//test//valid"):
#     model.eval()
#     with torch.no_grad():
#         for _, (img, filename) in tqdm(enumerate(loader), total=len(loader), desc="Predict"):
#             img = img.to(device)
#             output = model(img)            
                        
#             output = torch.sigmoid(output)[0, :]
#             output = (normalize_data(output.cpu().numpy())*255).astype(np.uint8)          
            
#             for dim in range(output.shape[0]):
#                 io.imsave(f"{save_path}//p{filename[dim]}", output[dim])


# def predict_split(model, loader, save_path="..//test//predict_split"):
#     model.eval()
#     with torch.no_grad():
#         for _, (img, filename) in tqdm(enumerate(loader), total=len(loader), desc="Predict"):
#             img = img.to(device)
#             output = model(img)            
#             # output = (torch.softmax(output, dim=1)[:, 1])
#             output = torch.sigmoid(output)[0, :]
#             output = (normalize_data(output.cpu().numpy())*255).astype(np.uint8)
            
#             for dim in range(output.shape[0]):
#                 io.imsave(f"{save_path}//p{filename[dim]}", output[dim])


# def predict_one(img):
#     img = VertebraDataset.preprocess(img)
#     format_img = np.zeros([1, 1, img.shape[0], img.shape[1]])
#     format_img[0, 0] = img
#     format_img = torch.tensor(format_img, dtype=torch.float)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = Unet(in_channels=1, out_channels=2)
#     model.load_state_dict(torch.load(model_path)["state_dict"])
#     model = model.to(device)
#     model.eval()
#     with torch.no_grad():
#         format_img = format_img.to(device)
#         output = model(format_img)
#         output = (torch.softmax(output, dim=1)[:, 1]) * 255
#         output = output.cpu().numpy().astype(np.uint8)
#     return output[0]


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    
    ########################### color ###############################

    # dataset = VertebraDataset("..//original_data//f01//image")
    # model = ResidualUNet(n_channels=1, n_classes=19)
    # # checkpoint = torch.load("save//best_color.pt")
    # checkpoint = torch.load("save//last_color.pt")

    ########################### color ###############################

    ########################### original ###############################

    # dataset = VertebraDataset("..//original_data//f01//image")
    # model = ResidualUNet(n_channels=1, n_classes=1)
    # # model = UNet(n_channels=1, n_classes=2)
    # checkpoint = torch.load("save//best.pt")
    # # checkpoint = torch.load("save//last.pt")

    ########################### original ###############################

    ########################### detect ###############################
        
    # dataset = VertebraDataset("..//detect_data//f01//image")
    dataset = VertebraDataset("..//valid_data//")
    model = ResidualUNet(n_channels=1, n_classes=1)
    # model = UNet(n_channels=1, n_classes=2)
    checkpoint = torch.load("save//best_detect.pt")
    # checkpoint = torch.load("save//last_detect.pt")

    ########################### detect ###############################
    
    


    # torch.save({"state_dict": model.state_dict(), "loss": loss_mean, "batchsize": batch_size, "Epoch": ep + 1}, path.join(save_dir, "last.pt"))
    model.load_state_dict(checkpoint["state_dict"])
    loader = DataLoader(dataset, batch_size=1)
    model = model.to(device)
    
    predict(model, loader)
    print("Done.")
