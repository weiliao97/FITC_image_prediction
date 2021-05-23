
from torch.autograd import Variable
import torch 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np 
from losses import *
import torch.nn.functional as F

def train(unet, discriminator, train_loader, optimizer_g, optimizer_d, epoch, gradedata = False, maskdata= False):
    unet.train()
    discriminator.train()

    loss_g = []
    loss_p = []
    loss_r = []
    loss_f = []
    for pack in train_loader:
        # print(label.shape)

        data = pack[0]
        stain = pack[1]
        label = pack[2]
        class_label = pack[3]

        data = Variable(data.float().to(device))
        stain = Variable(stain.float().to(device))

        # Adversarial ground truths
        valid = Variable(torch.Tensor(np.ones((data.size(0), 1))).to(device), requires_grad=False)
        fake = Variable(torch.Tensor(np.zeros((data.size(0), 1))).to(device), requires_grad=False)


        # ------------------
        #  Train Generators
        # ------------------

        optimizer_g.zero_grad()
        # GAN loss
        # pred = unet(data)
        fake_B = unet(data)


        # Pixel-wise loss

        if gradedata == True: 
            loss_pixel = 0.1*F.cross_entropy(fake_B, stain.squeeze(1).type(torch.cuda.LongTensor), weight= weights) + 0.9* tversky_loss(fake_B, stain.squeeze(1).type(torch.cuda.LongTensor), 0.5, 0.5)  
            #  
            # +
        else: 
            loss_pixel = weighted_loss(fake_B, stain)

        if gradedata == True: 
            fake_B = torch.argmax(fake_B, axis=1)
            fake_B = torch.unsqueeze(fake_B, dim=1).type(torch.cuda.FloatTensor)



        pred_fake = discriminator(fake_B, data)
        loss_GAN = F.mse_loss(pred_fake, valid)
       
        # Total loss
        loss_G = loss_GAN + 10* loss_pixel

        loss_G.backward()

        optimizer_g.step()
        

        # ---------------------
        #  Train Discriminator
        # ---------------------


        # Real loss
        optimizer_d.zero_grad()
        pred_real = discriminator(stain, data)
        loss_real = F.mse_loss(pred_real, valid)

        # Fake loss
        
        # if gradedata == True: 
        #     fake_B = torch.argmax(fake_B, axis=1)
        #     fake_B = torch.unsqueeze(fake_B, dim=1).type(torch.FloatTensor).to(device)

        pred_fake = discriminator(fake_B.detach(), data)
        loss_fake = F.mse_loss(pred_fake, fake)

        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)

        loss_D.backward()
        optimizer_d.step()
        

        loss_g.append(loss_GAN)
        loss_p.append(loss_pixel)
        loss_r.append(loss_real)
        loss_f.append(loss_fake)

    loss_g =  torch.stack(loss_g, dim=0)
    loss_g = torch.mean(loss_g).cpu().detach().numpy()
    loss_p =  torch.stack(loss_p, dim=0)
    loss_p = torch.mean(loss_p).cpu().detach().numpy()
    loss_r =  torch.stack(loss_r, dim=0)
    loss_r = torch.mean(loss_r).cpu().detach().numpy()
    loss_f =  torch.stack(loss_f, dim=0)
    loss_f = torch.mean(loss_f).cpu().detach().numpy()
    
    return loss_g, loss_p, loss_r, loss_f
    # return loss_p