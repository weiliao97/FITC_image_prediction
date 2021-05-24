import torch.nn.functional as F
from torch.autograd import Variable
import torch 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np 
from losses import * 

def test_dict(model, discriminator, dict_files, test_loader, gradedata = False, maskdata= False):
    zip_t = []
    for modeldict in dict_files: 
        model.load_state_dict(torch.load(modeldict + ".pt"))
        model.eval()

        discriminator.load_state_dict(torch.load(modeldict + "_d.pt"))
        discriminator.eval()
        
        loss_g = []
        loss_p = []
        loss_r = []
        loss_f = []
        loss_l1 = []

        y_list = []
        y_pred_list = []
        ros_list = []
        image_list = []
        condition_list = []

        if maskdata == True:
            mask_list = []
            # mask_back_list = []


        with torch.no_grad():  # validation does not require gradient
            for pack in test_loader:
    
                X = pack[0]
                stain = pack[1]
                ros = pack[2]
                class_label = pack[3]
                condition = pack [4]
                if maskdata == True:
                    mask = pack[5]
                    # mask_back = pack [6]

                X = Variable(torch.FloatTensor(X)).to(device)
                y = Variable(torch.FloatTensor(stain)).to(device)
                y_pred = model(X)

                valid = Variable(torch.Tensor(np.ones((X.size(0), 1))).to(device), requires_grad=False)
                fake = Variable(torch.Tensor(np.zeros((X.size(0), 1))).to(device), requires_grad=False)
            # Pixel-wise loss
                if gradedata == True: 
                    loss_pixel = 0.9*tversky_loss(y_pred, y.squeeze(1).type(torch.cuda.LongTensor), 0.5, 0.5) + 0.1*F.cross_entropy(y_pred, y.squeeze(1).type(torch.cuda.LongTensor), weight=weights) 
                    #  0.75*tversky_loss(y_pred, y.squeeze(1).type(torch.cuda.LongTensor), 0.5, 0.5) 
                    # 0.95* torch.pow(tversky_loss(y_pred, y.squeeze(1).type(torch.cuda.LongTensor), alpha=0.7, beta=0.3), 0.75)
                else: 
                    loss_pixel = weighted_loss(y_pred, y)
                    loss_simple = F.l1_loss(y_pred, y)

                if gradedata == True: 
                    y_pred = torch.argmax(y_pred, axis=1)
                    y_pred = torch.unsqueeze(y_pred, dim=1).type(torch.cuda.FloatTensor)

                pred_fake = discriminator(y_pred, X)
                loss_GAN = F.mse_loss(pred_fake, valid)

                
                
                # Total loss
                loss_G = loss_GAN + 10* loss_pixel

                #discrinminator
                pred_real = discriminator(y, X)

                loss_real = F.mse_loss(pred_real, valid)

                # Fake loss
                pred_fake = discriminator(y_pred, X)
                loss_fake = F.mse_loss(pred_fake, fake)

                # Total loss
                loss_D = 0.5 * (loss_real + loss_fake)

                loss_g.append(loss_GAN)
                loss_p.append(loss_pixel)
                loss_r.append(loss_real)
                loss_f.append(loss_fake)
                loss_l1.append(loss_simple)


                y_list.append(y)
                y_pred_list.append(y_pred)
                ros_list.append(ros)
                if maskdata == True:
                    mask_list.append(mask)
                    # mask_back_list.append(mask_back)
                # class_label_list.append(class_label)
                condition_list.append (condition)
                image_list.append(X)

        X = torch.cat(image_list, dim=0).cpu().detach().numpy()
        y = torch.cat(y_list, dim=0).cpu().detach().numpy()
        y_pred = torch.cat(y_pred_list, dim=0).cpu().detach().numpy()
        ros= torch.cat(ros_list, dim=0).cpu().detach().numpy()
        condition = torch.cat(condition_list, dim=0).cpu().detach().numpy()
        if maskdata == True:
            mask = torch.cat(mask_list, dim=0).cpu().detach().numpy()
            # mask_back = torch.cat(mask_back_list, dim=0).cpu().detach().numpy()
        
                
        loss_g =  torch.stack(loss_g, dim=0)
        loss_g = torch.mean(loss_g).cpu().detach().numpy()
        loss_l1 =  torch.stack(loss_l1, dim=0)
        loss_l1 = torch.mean(loss_l1).cpu().detach().numpy()
        loss_p =  torch.stack(loss_p, dim=0)
        loss_p = torch.mean(loss_p).cpu().detach().numpy()
        loss_r =  torch.stack(loss_r, dim=0)
        loss_r = torch.mean(loss_r).cpu().detach().numpy()
        loss_f =  torch.stack(loss_f, dim=0)
        loss_f = torch.mean(loss_f).cpu().detach().numpy()

        if maskdata == True:
            zip = [loss_g, loss_p, loss_r, loss_f, X, y, y_pred, ros, condition, mask, loss_l1]
            print 
        else: 
            zip = [loss_g, loss_p, loss_r, loss_f, X, y, y_pred, ros, condition]

        zip_t.append(zip)

    return zip_t