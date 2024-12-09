import torch
from torchvision import transforms
import os
import numpy as np
import matplotlib.image as mpimg
import skimage.transform
import constants
import data_process
from skimage.util import view_as_windows

def img_crop(im, w, h, gt):
    padding = constants.UNet_OVERLAP
    if not isinstance(im, torch.Tensor):
        if gt:
            im = torch.from_numpy(im).unsqueeze(0).float()
        else:
            im = torch.from_numpy(im).permute(2, 0, 1).float()
    pad_transform = transforms.Pad(padding, padding_mode="reflect")
    im_padded = pad_transform(im).permute(1, 2, 0).numpy()
    
    patch_shape = (w + 2 * constants.UNet_OVERLAP, h + 2 * constants.UNet_OVERLAP, im_padded.shape[2]) 
    patches = view_as_windows(im_padded, patch_shape, step=constants.UNet_PATCH_SIZE) 

    patches = patches.reshape(-1, *patch_shape)

    return patches

###################################################################################################################
###################################################################################################################

def extract_data(filename, imgs_idx, gt):
    imgs = []

    for i in imgs_idx:
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            img = mpimg.imread(image_filename)
            imgs.append(img)  
        else:
            print("File " + image_filename + " does not exist")

    num_images = len(imgs)

    img_patches = [
        img_crop(imgs[i], constants.UNet_PATCH_SIZE, constants.UNet_PATCH_SIZE, gt) for i in range(num_images)
    ]
    
    data = [
        patch
        for i in range(len(img_patches))
        for patch in img_patches[i]
    ]

    if gt:
        data = np.asarray([(data[i] > constants.FOREGROUND_THRESHOLD).astype(float) for i in range(len(data))])   

    return torch.tensor(np.transpose(np.array(data), (0, 3, 1, 2)))

###################################################################################################################
###################################################################################################################

def split_data(num_images):
    np.random.seed(12)
    tot_imgs = 100
    all_idx = np.arange(1, tot_imgs + 1)
    nbr_train_imgs = int(0.8 * num_images)
    nbr_valid_imgs = int(0.2 * num_images)

    train_idx = np.random.choice(all_idx, size = nbr_train_imgs, replace=False)
    remaining_indices = np.setdiff1d(all_idx, train_idx)
    valid_idx = np.random.choice(remaining_indices, size = nbr_valid_imgs, replace=False)

    return train_idx, valid_idx

###################################################################################################################
###################################################################################################################

def get_dataloaders(num_images, batch_size):

    train_idx, valid_idx = split_data(num_images)

    train_data = data_process.TrainImgsDataset(path_imgs = constants.PATH_TRAIN_IMGS, 
                                   path_gts = constants.PATH_TRAIN_GT,
                                   imgs_idx = train_idx, 
                                   )
    train_data.excecute()

    train_loader = torch.utils.data.DataLoader(
        dataset = train_data,
        batch_size = batch_size,
        shuffle = True,
        drop_last = True,
        num_workers = constants.NUM_WORKERS,
        pin_memory = torch.cuda.is_available()
    )

    valid_data = data_process.TrainImgsDataset(path_imgs = constants.PATH_TRAIN_IMGS, 
                                  path_gts = constants.PATH_TRAIN_GT, 
                                  imgs_idx = valid_idx, 
                                  )
    valid_data.excecute()

    valid_loader = torch.utils.data.DataLoader(
        dataset = valid_data,
        batch_size = batch_size,
        shuffle = False,
        drop_last = True,
        num_workers = constants.NUM_WORKERS,
        pin_memory = torch.cuda.is_available()
    )

    return train_loader, valid_loader

###################################################################################################################
###################################################################################################################

def load_test_data(path):
    imgs = []
    num_imgs = constants.NBR_TEST_IMAGES
    for i in range(1, num_imgs + 1):
        imageid = f"test_{i}"
        image_filename = path + imageid + "/" + imageid + ".png"
        if os.path.isfile(image_filename):
            img = mpimg.imread(image_filename)
            resized_img = skimage.transform.resize(img, (600, 600), anti_aliasing=True)  # Resize to 600x600
            imgs.append(resized_img)
        else:
            print("File " + image_filename + " does not exist")

    num_images = len(imgs)

    img_patches = [
        img_crop(imgs[i], constants.UNet_PATCH_SIZE, constants.UNet_PATCH_SIZE, gt = False) for i in range(num_images)
    ]
    data = [
        img_patches[i][j]
        for i in range(len(img_patches))
        for j in range(len(img_patches[i]))
    ]
   
    return torch.tensor(np.transpose(np.array(data), (0, 3, 1, 2)))

###################################################################################################################
###################################################################################################################

def label_to_img(imgwidth, imgheight, w, h, labels):
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            im[i : i + w, j : j + h] = labels[idx]
            idx = idx + 1
    return im

###################################################################################################################
###################################################################################################################

def patch_to_label(patch):
    df = np.mean(patch)
    if df > constants.FOREGROUND_THRESHOLD:
        return 1
    else:
        return 0
    
###################################################################################################################
###################################################################################################################

def mask_to_submission_strings(img, idx):
    """Reads a single image and outputs the strings that should go into the submission file"""
    patch_size = 16
    for j in range(0, img.shape[1], patch_size):
        for i in range(0, img.shape[0], patch_size):
            patch = img[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(idx, j, i, label))

###################################################################################################################
###################################################################################################################

def masks_to_submission(submission_filename, imgs):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for i in range(len(imgs)):
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(imgs[i], i+1))






   