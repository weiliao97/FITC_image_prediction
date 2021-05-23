from __future__ import print_function
from torchsummary import summary
import torch.utils.data as data
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import sys
import cv2 

from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi']= 150

from train import *
from test import *

from models import Discriminator
from models import UNet_Attention
from prepare_data import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parser for FITC image prediction")

    parser.add_argument("--dataset_path", type=str, help="path to the training dataset")

    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--checkpoint_model", type=str, help=" path to checkpoint model")


    parser.add_argument('--resume_from', type=bool, required=False, help = "whether retrain from a certain old model")
    parser.add_argument('--old_dict', required='--resume_from' in sys.argv)



    parser.add_argument("--original_target", type=bool, help="whether use the original FITC as the target")
    parser.add_argument("--phase_norm", type=bool, help="whether perform normalization on the phase data")


    parser.add_argument("--attention", type=bool, help="whether use attentionin the mdoel or not")



    parser.add_argument("--img_size", type=int, default=64, help="image size")


    # Parse and return arguments
    args = parser.parse_args()


    best_loss = 1e3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    # creat discinminator
    discriminator = Discriminator()
    discriminator.to(device)
    summary(discriminator,[(1, args.img_size, args.img_size), (1, args.img_size, args.img_size)])

    #unet or attention 
    if args.attention:
        unet = UNet_Attention()
        unet.to(device)
    else: 
        unet = UNET(in_channels=1, out_channels=6)
        unet.to(device)




    # transformation 
    img_trans = torch.nn.Sequential(
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomRotation(45, interpolation=transforms.InterpolationMode.NEAREST, expand=False, center=None, fill=0, resample=None),
                                    transforms.RandomVerticalFlip(p=0.5),
    )
    scripted_transforms = torch.jit.script(img_trans)

    #data_loader
    train_loader, test_loader, train_dataset, test_dataset = split_data(args.dataset_path, args.batch_size, args.phase_norm, ros_norm = False, trans=scripted_transforms, maskeddata = args.original_target)


    if args.resume_from is not None:
        print("Loading weights from %s" % args.old_dict)
        unet.load_state_dict(torch.load(old_dict + ".pt"))
        discriminator.load_state_dict(torch.load(old_dict + + "_d.pt"))
        zip_p = test_rg(unet, discriminator, test_loader, weights, gradedata = False, maskdata=args.original_target)
        best_loss = zip_p[1]
        


    optimizer_g = torch.optim.Adam(unet.parameters(), betas=(0.5, 0.999), lr=args.lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), betas=(0.5, 0.999), lr=args.lr)

    loss_te = []
    loss_tr = []
    loss_gan= []
    loss_d = []

    for epoch in tqdm(range(args.epochs)):
        loss_tr_g, loss_tr_p, loss_tr_r, loss_tr_f  = train(unet, discriminator, train_loader, optimizer_g, optimizer_d, epoch, gradedata = False, maskdata=args.original_target)
        print('Train loss : total: {:.4f}, Gan loss: {:.4f} + pixel loss: {:.4f} + true loss: {:.4f} + fake loss: {:.4f}'.format(loss_tr_g+loss_tr_p, loss_tr_g, loss_tr_p, loss_tr_r, loss_tr_f))


        zip = test_rg(unet, discriminator, test_loader, gradedata = False, maskdata=args.original_target)

        loss_g = zip[0]
        loss_p = zip[1]
        loss_r = zip[2]
        loss_f = zip[3]
        phase_te = zip[4]
        fitc_te = zip[5]
        fitc_tep = zip[6]
        ros_te = zip[7]
        mask = zip[9]
        loss_l1 = zip[10]

        print('Val loss : total: {:.4f}, Gan loss: {:.4f} + pixel loss: {:.4f} + true l1: {:.4f} + true loss: {:.4f} + fake loss: {:.4f}'.format(loss_g+ loss_p, loss_g, loss_p, loss_l1, loss_r, loss_f))

        loss_te.append(loss_p)
        loss_tr.append(loss_tr_p)
        loss_gan.append(loss_g)
        loss_d.append(0.5*(loss_r+loss_f))

        if  loss_p < best_loss:
            torch.save(unet.state_dict(), args.checkpoint_model + ".pt")
            torch.save(discriminator.state_dict(), args.checkpoint_model + "_d.pt" )
            best_loss = loss_p


        #visuaize images

        a = np.random.choice(range(len(ros_te)), 6)
        fig, axs = plt.subplots(2, 9, figsize=(12, 5))
        ax = axs.ravel()

        for r, k in enumerate(a):
            final = fitc_tep[k, 0]
            ax[r*3].imshow(final, vmin=0, vmax=5)
            ax[r*3].axis('off')

            ax[r*3+1].imshow(fitc_te[k, 0], vmin=0, vmax=5)
            ax[r*3+1].axis('off')

            ax[r*3+2].imshow(phase_te[k, 0])
            ax[r*3+2].axis('off')
        plt.show()


        #visulize statistics 
        signal = [cv2.mean(fitc_tep[i,0], mask[i,0].astype(np.uint8))[0] for i in range(len(ros_te))]
        signal_or= [cv2.mean(fitc_te[i,0], mask[i,0].astype(np.uint8))[0] for i in range(len(ros_te))]
        in_lier_20 = [e for e in range(len(ros_te)) if abs(signal[e] - signal_or[e])/signal_or[e] <0.2]
        ratio_20 = len(in_lier_20)/len(ros_te)

        plt.figure(figsize=(5, 5))
        plt.scatter(signal_or, signal, s=1)
        plt.plot(np.linspace(0, 6, 100), np.linspace(0, 6, 100), c= "red", linestyle=':')
        plt.text(5, 5, '%.2f'%ratio_20, fontsize = 14)
        plt.ylim((1, 6))
        plt.xlim((1, 6))
        plt.show()