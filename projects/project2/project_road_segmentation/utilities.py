import torch
from torchvision import transforms
import os
import numpy as np
import matplotlib.image as mpimg
import constants
import data_process

# Helper functions

def load_image(infilename):
    data = mpimg.imread(infilename)
    return data

def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

# Concatenate an image and its groundtruth
def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)
        gt_img_3c[:, :, 0] = gt_img8
        gt_img_3c[:, :, 1] = gt_img8
        gt_img_3c[:, :, 2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

# Crop images into smaller patches of specified width w and height h
def img_crop(im, w, h, gt = False):
    if not gt:
        padding = (constants.IMG_PATCH_SIZE - constants.STANDARD_PATCH_SIZE) // 2
        if not isinstance(im, torch.Tensor):
            im = torch.from_numpy(im).permute(2, 0, 1).float()
        pad_transform = transforms.Pad(padding, padding_mode = "reflect")
        im_padded = pad_transform(im)
        _, imgheight, imgwidth = im_padded.shape
    else:
        im_padded = im
        imgheight, imgwidth = im_padded.shape
    
    list_patches = []
    is_2d = len(im_padded.shape) < 3
    step = constants.STANDARD_PATCH_SIZE
    for i in range(0, imgheight - h + 1, step):
        for j in range(0, imgwidth - w + 1, step):
            if is_2d:
                im_patch = im_padded[j : j + w, i : i + h]
            else:
                im_patch = im_padded[:, j : j + w, i : i + h]
            list_patches.append(im_patch)
    return list_patches

def extract_data(filename, num_images, train):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []

    start_img = 1 if train else 101 - num_images
    stop_img = num_images + 1 if train else 101

    for i in range(start_img, stop_img):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            img = mpimg.imread(image_filename)
            imgs.append(img) # img is a numpy array of shape (400x400x3)
        else:
            print("File " + image_filename + " does not exist")

    num_images = len(imgs)

    img_patches = [
        img_crop(imgs[i], constants.IMG_PATCH_SIZE, constants.IMG_PATCH_SIZE) for i in range(num_images)
    ]
    
    data = [
        img_patches[i][j]
        for i in range(len(img_patches))
        for j in range(len(img_patches[i]))
    ]

    return torch.tensor(np.array(data))

# Assign a label to a patch v. Returns 1 for road and 0 for background    
def patch_to_label(patch):
    foreground_threshold = 0.25
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0
    
# Extract label images
def extract_labels(filename, num_images, train):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []

    start_img = 1 if train else 101 - num_images
    stop_img = num_images + 1 if train else 101

    for i in range(start_img, stop_img):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print("File " + image_filename + " does not exist")

    num_images = len(gt_imgs)
    gt_patches = [
        img_crop(gt_imgs[i], constants.STANDARD_PATCH_SIZE, constants.STANDARD_PATCH_SIZE, gt = True) for i in range(num_images)
    ]
    data = np.asarray(
        [
            gt_patches[i][j]
            for i in range(len(gt_patches))
            for j in range(len(gt_patches[i]))
        ]
    )
    labels = np.asarray(
        [patch_to_label(data[i]) for i in range(len(data))]
    )

    return torch.tensor(labels).long()

def compute_mean_std(loader):
    mean = 0
    var = 0

    for _, (imgs, _) in enumerate(loader):
        mean += imgs.mean(dim = [0, 2, 3])
        var += (imgs - imgs.mean(dim=[0, 2, 3], keepdim=True)) ** 2.0
    
    mean /= len(loader)
    var /= len(loader)
    std = torch.sqrt(var.mean(dim=[0, 2, 3]))
    
    return mean, std


def get_dataloaders(num_images, batch_size, data_aug):

    train_data = data_process.TrainImgsDataset(path_imgs = 'training/images/', 
                                   path_gts = 'training/groundtruth/', 
                                   train = True,
                                   num_images = num_images, 
                                   transform = False,
                                   aug = data_aug,
                                   aug_factor = 1.1)
    train_data.excecute()

    mean_std_loader = torch.utils.data.DataLoader(
        dataset = train_data,
        batch_size = 128,
        shuffle = False,
        drop_last = True,
        num_workers = constants.NUM_WORKERS,
        pin_memory = torch.cuda.is_available()
    )

    mean, std = compute_mean_std(mean_std_loader)

    transform = transforms.Compose([transforms.Normalize(mean.tolist(), std.tolist())])

    train_data.transform = transform

    train_loader = torch.utils.data.DataLoader(
        dataset = train_data,
        batch_size = batch_size,
        shuffle = True,
        drop_last = True,
        num_workers = constants.NUM_WORKERS,
        pin_memory = torch.cuda.is_available()
    )

    valid_data = data_process.TrainImgsDataset(path_imgs = 'training/images/', 
                                  path_gts = 'training/groundtruth/', 
                                  train = False,
                                  num_images = max(1, int(np.round(num_images/4, 0))), 
                                  transform = transform,
                                  aug = False,
                                  aug_factor = 1)
    valid_data.excecute()

    valid_loader = torch.utils.data.DataLoader(
        dataset = valid_data,
        batch_size = batch_size,
        shuffle = False,
        drop_last = True,
        num_workers = constants.NUM_WORKERS,
        pin_memory = torch.cuda.is_available()
    )

    return train_loader, valid_loader, train_data.weights

def load_test_data(path):
    imgs = []
    num_imgs = 50

    for i in range(1, num_imgs + 1):
        imageid = f"test_{i}"
        image_filename = path + imageid + "/" + imageid + ".png"
        if os.path.isfile(image_filename):
            img = mpimg.imread(image_filename)
            imgs.append(img) # each image has shape 608x608x3
        else:
            print("File " + image_filename + " does not exist")

    num_images = len(imgs)

    img_patches = [
        img_crop(imgs[i], constants.IMG_PATCH_SIZE, constants.IMG_PATCH_SIZE) for i in range(num_images)
    ]
    data = [
        img_patches[i][j]
        for i in range(len(img_patches))
        for j in range(len(img_patches[i]))
    ]
   
    return torch.tensor(np.array(data))

def label_to_img(imgwidth, imgheight, w, h, labels):
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            im[j : j + w, i : i + h] = labels[idx]
            idx = idx + 1
    return im

def mask_to_submission_strings(img, idx):
    """Reads a single image and outputs the strings that should go into the submission file"""
    patch_size = 16
    for j in range(0, img.shape[1], patch_size):
        for i in range(0, img.shape[0], patch_size):
            patch = img[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(idx, j, i, label))

def masks_to_submission(submission_filename, imgs):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for i in range(len(imgs)):
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(imgs[i], i+1))