import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms
import constants
import utilities

class TrainImgsDataset(Dataset):
    def __init__(
            self, 
            path_imgs, 
            path_gts, 
            imgs_idx,
    ):
       super().__init__()
       self.patches = None 
       self.labels = None
       self.path_imgs = path_imgs
       self.path_gts = path_gts
       self.imgs_idx = imgs_idx

    def get_patches(self):
        self.patches = utilities.extract_data(self.path_imgs, self.imgs_idx, gt = False) 

    def get_labels(self):
        self.labels = utilities.extract_data(self.path_gts, self.imgs_idx, gt = True)  
    
    def augment_data(self):
        patch_size = constants.UNet_PATCH_SIZE + 2 * constants.UNet_OVERLAP

        horizontal_flip = torchvision.transforms.RandomHorizontalFlip(p=1) 
        vertical_flip = torchvision.transforms.RandomVerticalFlip(p=1) 
        pad_transform = torchvision.transforms.Pad(int(0.2*patch_size), padding_mode = "reflect")
        crop_transform = torchvision.transforms.CenterCrop((patch_size, patch_size))

        padded_imgs = pad_transform(self.patches)
        padded_labels = pad_transform(self.labels)

        angles = np.linspace(0, 90, 25)
        for angle in angles:
            rot_imgs = torchvision.transforms.functional.rotate(padded_imgs, angle)
            rot_labels = torchvision.transforms.functional.rotate(padded_labels, angle)

            rot_imgs = crop_transform(rot_imgs)
            rot_labels = crop_transform(rot_labels)

            hor_flip_imgs = horizontal_flip(rot_imgs)
            hor_flip_labels = horizontal_flip(rot_labels)
            ver_flip_imgs = vertical_flip(rot_imgs)
            ver_flip_labels = vertical_flip(rot_labels)

            if angle == 0:
                self.patches = torch.cat((self.patches, hor_flip_imgs, ver_flip_imgs), dim = 0)
                self.labels = torch.cat((self.labels, hor_flip_labels, ver_flip_labels), dim = 0)
            else:
                self.patches = torch.cat((self.patches, rot_imgs, hor_flip_imgs, ver_flip_imgs), dim = 0)
                self.labels = torch.cat((self.labels, rot_labels, hor_flip_labels, ver_flip_labels), dim = 0)

    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, index):
        img = self.patches[index]
        label = self.labels[index]
        return img, label
    
    def excecute(self):
        self.get_patches()  
        self.get_labels()
        self.augment_data()


class TestImgsDataset(Dataset):
    def __init__(
            self, 
            path_test_imgs, 
    ):
       super().__init__()
       self.patches = None 
       self.path_test_imgs = path_test_imgs

    def get_data(self):
        self.patches = utilities.load_test_data(self.path_test_imgs) 

    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, index):
        img = self.patches[index]
        return img
    
    def excecute(self):
        self.get_data()  