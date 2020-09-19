from torch.nn import Module, CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
from os import path
from model.unet import Unet
from model.unet import UNet # new 
from model.resunet import ResidualUNet
from data.dataset import VertebraDataset
import matplotlib.pyplot as plt
import time
import warnings
import numpy as np
import cv2
import torch.nn.functional as F
warnings.filterwarnings('ignore')

def dice_coef(target, truth, smooth=1.0):
    target = target.contiguous().view(-1)
    truth = truth.contiguous().view(-1)
    union = target.sum() + truth.sum()
    intersection = torch.sum(target * truth)
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice



def Multiclass_dice_coef(input, target, weights=None):
    num_labels = target.shape[1]    # one-hot encoding length

    if weights is None:
        weights = torch.ones(num_labels) # uniform weights for all classes

    # weights[0] = 0

    totalLoss = 0
 
    for idx in range(num_labels):
	    diceLoss = dice_coef(input[:, idx], target[:, idx])
	    if weights is not None:
		    diceLoss *= weights[idx]
	    totalLoss += diceLoss
 
    return totalLoss/num_labels

def normalize_data(output, threshold=0.7):
    output = output.cpu().detach().numpy()
    output = np.where(output > threshold, 1.0, 0)
    output = torch.from_numpy(output).to("cuda")
    
    return output

# def save_fig(epoch, loss, trainscore, testscore, save_dir=path.join("save")):
#     plt.plot(epoch, loss, label="Loss")
#     plt.title("Loss")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.legend()
#     plt.savefig(path.join(save_dir, "loss.png"))
#     plt.clf()

#     plt.plot(epoch, trainscore, label="Train")
#     plt.plot(epoch, testscore, label="Test")
#     plt.title("Score")
#     plt.xlabel("Epoch")
#     plt.ylabel("Score")
#     plt.legend()
#     plt.savefig(path.join(save_dir, "score.png"))
#     plt.clf()


def eval(model, loader, device):
    scores = list()
    model.eval()
    with torch.no_grad():
        for _, (img, mask) in tqdm(enumerate(loader), total=len(loader), desc="Evaluate"):
            img = img.to(device)
            mask = mask.to(device)
            output = model(img)
            
            # output = torch.softmax(output, dim=1)

            ######################### original #########################

            output = torch.sigmoid(output)
            output =  normalize_data(output)# OUTPUT NEED TO NORMALIZE TO 0 OR 1 FOR A THRESHOLD
            score = dice_coef(output[:, :], mask[:, :output.shape[2], :output.shape[3]])

            ######################### original #########################

            ######################### color #########################

            # output = torch.sigmoid(output)
            # output = normalize_data(output, threshold=0.5)

            # # output = torch.softmax(output, dim=1)
            # # output = torch.argmax(output, dim=1)
            
            # score = Multiclass_dice_coef(output, mask)
            # # score = dice_coef(output[0][0], mask[0][0])

            ######################### color #########################

            scores.append(score)
    return torch.mean(torch.stack(scores, dim=0))


def run_one_epoch(model, loader, device, criterion, optimizer):
    total_loss = 0
    model.train()
    for _, (img, mask) in tqdm(enumerate(loader), total=len(loader), desc="Train"):
        img = img.to(device)
        mask = mask.to(device)
        optimizer.zero_grad()
        output = model(img)
        
        
        ######################### color #########################
        
        # # output = F.sigmoid(output)
        
        # # output = normalize_data(output, threshold=0.7)
        
        # # loss = criterion(output[0], mask[0])
        # loss = criterion(output, mask)

        ######################### color #########################

        ######################### original #########################

        loss = criterion(output[0,0,:,:], mask[0, :, :])

        ######################### original #########################
        
        loss.backward()
        optimizer.step()
        total_loss += loss
    return total_loss / len(loader)

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
    
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class MulticlassDiceLoss(nn.Module):
	"""
	requires one hot encoded target. Applies DiceLoss on each class iteratively.
	requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
	  batch size and C is number of classes
	"""
	def __init__(self):
		super(MulticlassDiceLoss, self).__init__()
 
	def forward(self, input, target, weights=None):
 
		num_labels = target.shape[1]    # one-hot encoding length

		if weights is None:
		    weights = torch.ones(num_labels) #uniform weights for all classes


		dice = DiceLoss()
		totalLoss = 0
 
		for idx in range(num_labels):
			diceLoss = dice(input[:, idx], target[:, idx])
			if weights is not None:
				diceLoss *= weights[idx]
			totalLoss += diceLoss
 
		return totalLoss/num_labels


def train(model, traindataset, testdataset, device, epochs, criterion, optimizer, batch_size=1, save_dir=path.join("save")):
    fig_epoch = list()
    fig_loss = list()
    fig_train_score = list()
    fig_test_score = list()

    highest_epoch = highest_score = 0

    trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True, num_workers=10, pin_memory=True)
    testloader = DataLoader(testdataset, batch_size=batch_size, shuffle=True, num_workers=10, pin_memory=True)

    model = model.to(device)

    for ep in range(epochs):
        timer = time.clock()

        learning_rate = 0
        
        for param_group in optimizer.param_groups:
            learning_rate = param_group['lr']
        
        adjust_learning_rate(learning_rate, optimizer, ep)

        print(f"[ Epoch {ep + 1}/{epochs} ]")
        loss_mean = run_one_epoch(model, trainloader, device, criterion, optimizer)
        train_score = eval(model, trainloader, device)
        test_score = eval(model, testloader, device)

        # fig_epoch.append(ep + 1)
        # fig_loss.append(loss_mean)
        # fig_train_score.append(train_score)
        # fig_test_score.append(test_score)
        # save_fig(fig_epoch, fig_loss, fig_train_score, fig_test_score, save_dir=save_dir)

        if test_score > highest_score:
            highest_score = test_score
            highest_epoch = ep + 1
            ######################### original #########################
            # torch.save({"state_dict": model.state_dict(), "loss": loss_mean, "batchsize": batch_size, "Epoch": ep + 1}, path.join(save_dir, "best.pt"))
            ######################### original #########################

            ######################### color #########################
            # torch.save({"state_dict": model.state_dict(), "loss": loss_mean, "batchsize": batch_size, "Epoch": ep + 1}, path.join(save_dir, "best_color.pt"))
            ######################### color #########################

            ######################### detect #########################
            torch.save({"state_dict": model.state_dict(), "loss": loss_mean, "batchsize": batch_size, "Epoch": ep + 1}, path.join(save_dir, "best_test_f01.pt"))
            ######################### detect #########################

        ######################### original #########################
        # torch.save({"state_dict": model.state_dict(), "loss": loss_mean, "batchsize": batch_size, "Epoch": ep + 1}, path.join(save_dir, "last.pt"))
        ######################### original #########################

        ######################### color #########################
        # torch.save({"state_dict": model.state_dict(), "loss": loss_mean, "batchsize": batch_size, "Epoch": ep + 1}, path.join(save_dir, "last_color.pt"))
        ######################### color #########################

        ######################### detect #########################
        torch.save({"state_dict": model.state_dict(), "loss": loss_mean, "batchsize": batch_size, "Epoch": ep + 1}, path.join(save_dir, "last_test_f01.pt"))
        ######################### detect #########################
        print(f"""
Best Score {highest_score} @ Epoch {highest_epoch}
Learning Rate: {learning_rate}
Loss: {loss_mean}
Train Dice: {train_score}
Test Dice: {test_score}
Time passed: {round(time.clock() - timer)} seconds.
""")

def adjust_learning_rate(LEARNING_RATE, optimizer, epoch):
    lr = LEARNING_RATE * (0.8 ** (epoch // 70))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def clahe_hist(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
    return cl1

if __name__ == '__main__':
    EPOCH = 120
    BATCHSIZE = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    ######################### color #########################

    # traindataset = VertebraDataset("..//label_extend_dataset", train=True)
    # testdataset = VertebraDataset("..//label_data//f01", train=True)
    # model = ResidualUNet(n_channels=1, n_classes=19)
    # criterion = BCEWithLogitsLoss()
    # # criterion = MulticlassDiceLoss()
    
    ######################### color #########################
    
    ######################### original #########################

    # traindataset = VertebraDataset("..//extend_dataset", train=True)
    # testdataset = VertebraDataset("..//original_data//f01", train=True)
    # model = ResidualUNet(n_channels=1, n_classes=1)
    # criterion = BCEWithLogitsLoss()

    ######################### original #########################

    ######################### split #########################

    # traindataset = VertebraDataset("..//splitting_dataset", train=True)
    # testdataset = VertebraDataset("..//original_split_data//f01", train=True)

    ######################### split #########################

    ######################### detect #########################

    traindataset = VertebraDataset("..//train", train=True)
    testdataset = VertebraDataset("..//test", train=True)
    model = ResidualUNet(n_channels=1, n_classes=1)
    criterion = BCEWithLogitsLoss()

    ######################### detect #########################

    # model = ResidualUNet(n_channels=1, n_classes=1)
    # model = UNet(n_channels=1, n_classes=2)
    # model = Unet(in_channels=1, out_channels=2)
    # model = ResUnet(in_channels=1, out_channels=2)
    
    # criterion = CrossEntropyLoss()
    # criterion = BCEWithLogitsLoss()
    # criterion = DiceLoss()
    
    optimizer = Adam(model.parameters(), lr=1e-4)

    train(model, traindataset, testdataset, device, EPOCH, criterion, optimizer, batch_size=BATCHSIZE)
    print("Done.")
