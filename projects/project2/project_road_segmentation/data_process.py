import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms
import constants
import utilities

class CutoutTransform:
    def __init__(self, cutout_size, fill_val):
        self.cutout_size = cutout_size
        self.fill_val = fill_val
    
    def __call__(self, imgs):
        h, w = imgs.shape[2], imgs.shape[3]
        cutout_size = min(self.cutout_size, w, h)

        left_x = np.random.randint(0, w - self.cutout_size + 1)
        top_y = np.random.randint(0, h - self.cutout_size + 1)

        imgs[:, :, left_x : left_x + cutout_size, top_y : top_y + cutout_size] = self.fill_val

        return imgs

class TrainImgsDataset(Dataset):
    def __init__(
            self, 
            path_imgs, 
            path_gts, 
            train,
            num_images,
            transform,
            aug,
            aug_factor,
    ):
       super().__init__()
       self.patches = None 
       self.labels = None
       self.weights = None
       self.path_imgs = path_imgs
       self.path_gts = path_gts
       self.train = train
       self.num_images = num_images
       self.transform = transform
       self.aug = aug
       self.aug_factor = aug_factor

    def get_data(self):
        self.patches = utilities.extract_data(self.path_imgs, self.num_images, self.train) 

    def get_labels(self):
        self.labels = utilities.extract_labels(self.path_gts, self.num_images, self.train)  

    def get_weights(self): 
        class_counts = torch.bincount(self.labels).float()
        self.weights = 1.0 / class_counts
        self.weights /= self.weights.min()
    
    def augment_data(self):
        if self.aug_factor <= 1 or not self.aug:
            return
        
        nbr_aug_imgs = int(len(self.patches) * (self.aug_factor - 1)) 
        select_idx = np.random.choice(len(self.patches), nbr_aug_imgs) # select patches to which the transforms will be applied to
        
        # Blurring
        blur_transform = torchvision.transforms.GaussianBlur(kernel_size = 5, sigma = 0.5)
        self.apply_transform(blur_transform, select_idx)
        
        # Rotate
        pad_transform = torchvision.transforms.Pad(int(0.2*constants.IMG_PATCH_SIZE), padding_mode = "reflect")
        rot_transfom = torchvision.transforms.RandomRotation(degrees = (-90, 90))
        crop_transform = torchvision.transforms.CenterCrop((constants.IMG_PATCH_SIZE, constants.IMG_PATCH_SIZE))
        combined_transform = torchvision.transforms.Compose([
            pad_transform,
            rot_transfom,
            crop_transform
        ])
        self.apply_transform(combined_transform, select_idx)
        
        # Color distort
        jitter_transform = torchvision.transforms.ColorJitter(brightness = 0.35, contrast = 0.35, saturation = 0.35, hue = 0.12)
        self.apply_transform(jitter_transform, select_idx)

        # Cutout
        mean = self.patches[select_idx].mean().item()
        cutout_transform = CutoutTransform(cutout_size = int(0.3 * constants.IMG_PATCH_SIZE), fill_val = mean)
        self.apply_transform(cutout_transform, select_idx)

    
    def apply_transform(self, transform, select_idx):
        select_patches = self.patches[select_idx]
        select_labels = self.labels[select_idx]
        aug_patches = transform(select_patches)
        self.patches = torch.cat((self.patches, aug_patches), dim = 0)
        self.labels = torch.cat((self.labels, select_labels), dim = 0)

    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, index):
        img = self.patches[index]
        label = self.labels[index]
        if self.transform:
            img = self.transform(img)
        return img, label
    
    def excecute(self):
        self.get_data()  
        self.get_labels()
        self.augment_data()
        self.get_weights()


class TestImgsDataset(Dataset):
    def __init__(
            self, 
            path_test_imgs, 
            transform,
    ):
       super().__init__()
       self.patches = None 
       self.path_test_imgs = path_test_imgs
       self.transform = transform

    def get_data(self):
        self.patches = utilities.load_test_data(self.path_test_imgs) 

    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, index):
        img = self.patches[index]
        if self.transform:
            img = self.transform(img)
        return img
    
    def excecute(self):
        self.get_data()  