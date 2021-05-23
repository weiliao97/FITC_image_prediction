import torch.utils.data as data
import numpy as np 
import torch 
import torch.nn.functional as F

class Dataset(data.Dataset):
    """Args:
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    def __init__(self, data, stain, label, classes, condition, mask = None, mask_back = None,
                 transform=None, target_transform=None, mask_return=False):
        self.transform = transform
        self.target_transform = target_transform
        self.data = data
        self.stain = stain
        self.labels = label
        self.classes = classes
        self.condition = condition 
        self.mask = mask 
        self.mask_back = mask_back 
        self.mask_return = mask_return


    def __getitem__(self, index):
        """
         Args:
             index (int): Index
         Returns:
             tuple: (image, target) where target is index of the target class.
         """
        if self.mask_return == True: 
            img, stain, ros_target, class_target, condition, mask = self.data[index], self.stain[index], self.labels[index], self.classes[index], self.condition[index], self.mask[index]
        else:
            img, stain, ros_target, class_target, condition = self.data[index], self.stain[index], self.labels[index], self.classes[index], self.condition[index]

        img = np.float32(img)
        img = torch.from_numpy(img)
        stain = torch.from_numpy(stain)

        p2d = (2, 2, 2, 2)

        if self.mask_return == True: 
            mask = torch.from_numpy(mask)
            # mask_back = torch.from_numpy(mask_back/255)
            mask= F.pad(mask, p2d, "constant", 0)
            # mask_back= F.pad(mask_back, p2d, "constant", 0)
        

        img= F.pad(img, p2d, "constant", 0)
        stain= F.pad(stain, p2d, "constant", 0)
    

        # fitc = F.one_hot(stain.to(torch.int64), num_classes=10)

        if self.transform is not None: 
            # stain = torch.unsqueeze(stain, 1)
            if self.mask_return == True: 
                stacked = torch.cat([img, stain, mask], dim=0)  # shape=(2xHxW)
                stacked = self.transform(stacked)
                img, stain, mask  = torch.chunk(stacked, chunks=3, dim=0)
                mask = mask.type(torch.LongTensor)
                # mask_back = mask_back.type(torch.LongTensor)
            else: 
                stacked = torch.cat([img, stain], dim=0)  # shape=(2xHxW)
                stacked = self.transform(stacked)
                img, stain = torch.chunk(stacked, chunks=2, dim=0)
        
        img = img.type(torch.FloatTensor)
        stain = stain.type(torch.FloatTensor)

        # stain = torch.squeeze(stain, 0).type(torch.LongTensor)

        ros_target = np.float32(ros_target)
        class_target = np.long(class_target)

        if self.mask_return == True: 
            pack = [img, stain, ros_target, class_target, condition, mask]
        else:
            pack = [img, stain, ros_target, class_target, condition]

        return pack

    def __len__(self):
        return len(self.data)


def split_data(datafile, batch_size, phase_norm, ros_norm, trans=None, maskeddata=False):

    img_data = np.load(datafile, allow_pickle=True).item()
    phase_img = img_data['Images']
    fitc_img = img_data['labels']
    ros_level = img_data['ros_level']
    classes = img_data['classes']
    condition =img_data['condition']
    if maskeddata == True: 
        mask = img_data['mask']
        # mask_back = data['mask_back']
        mask = np.expand_dims(mask, axis = 1) 
        # mask_back = np.expand_dims(mask_back, axis = 1) 

    if phase_norm == True:
        print(np.amin(phase_img), np.amax(phase_img))
        phase_img = (phase_img - np.amin(phase_img)) / (np.amax(phase_img)- np.amin(phase_img))
        # fitc_img = (fitc_img - np.amin(fitc_img)) / (np.amax(fitc_img)- np.amin(fitc_img))

    if ros_norm == True:
        rmin = np.amin(ros_level)
        rmax = np.amax(ros_level)
        r_mean = np.mean(ros_level)
        r_std = np.std(ros_level)
        ros_level = (ros_level - rmin) / (rmax - rmin)
        # ros_level= np.tanh((0.1 * (ros_level - r_mean) / r_std ) - (0.1 * (rmin - r_mean) / r_std))
        print(rmin, rmax)

    ros_level = np.expand_dims(ros_level, axis=-1)
    phase_img = np.expand_dims(phase_img, axis = 1) 
    fitc_img = np.expand_dims(fitc_img, axis = 1) 
    

    np.random.seed(0)
    index_tr = np.random.choice(len(condition), int(0.8 * len(condition)), replace=False)
    index_te = [e for e in range(len(condition)) if e not in index_tr]

    if maskeddata == True: 
        train_dataset = Dataset(phase_img[index_tr], fitc_img[index_tr], ros_level[index_tr], classes[index_tr], condition[index_tr], mask[index_tr], transform = trans, mask_return = True)
        test_dataset = Dataset(phase_img[index_te], fitc_img[index_te], ros_level[index_te], classes[index_te], condition[index_te], mask[index_te], transform = None, mask_return = True)
    else: 
        train_dataset = Dataset(phase_img[index_tr], fitc_img[index_tr], ros_level[index_tr], classes[index_tr], condition[index_tr], transform = trans, mask_return = False)
        test_dataset = Dataset(phase_img[index_te], fitc_img[index_te], ros_level[index_te], classes[index_te], condition[index_te], transform = None,  mask_return = False)


    print('Train size: {:d}% Test size: {:d}%'.format(len(index_tr), len(index_te)))
    data_loader_train = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    data_loader_test = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return data_loader_train, data_loader_test, train_dataset, test_dataset