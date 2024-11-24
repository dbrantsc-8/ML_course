import torch
import os, sys
import numpy as np
import matplotlib.image as mpimg

IMG_PATCH_SIZE = 16
NUM_WORKERS = 2

# Helper functions

# Crop images into smaller patches of specified width w and height h
def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j : j + w, i : i + h]
            else:
                im_patch = im[j : j + w, i : i + h, :]
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
        img_crop(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)
    ]
    data = [
        img_patches[i][j]
        for i in range(len(img_patches))
        for j in range(len(img_patches[i]))
    ]

    return torch.tensor(data) # numpy array of shape (6250x16x16x3), (6250 = 10 images * 25*25 patches per image)

# Assign a label to a patch v. Returns 1 for road and 0 for background
def value_to_class(v):
    foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch
    #df = np.sum(v)
    if v > foreground_threshold:  # road
        return 1
    else:  # bgrd
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
        img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)
    ]
    data = np.asarray(
        [
            gt_patches[i][j]
            for i in range(len(gt_patches))
            for j in range(len(gt_patches[i]))
        ]
    )
    labels = np.asarray(
        [value_to_class(np.mean(data[i])) for i in range(len(data))]
    )

    return torch.tensor(labels).long()
    #return labels.astype(np.float32) # shape 6250x2 for 10 images (6250 patches and [0,1] od [1,0] for each patch)

def compute_mean_std(train_data):
    loader = torch.utils.data.DataLoader(
        dataset = train_data,
        batch_size = 128,
        shuffle = False,
        drop_last = True,
        num_workers = 0,
        pin_memory = torch.cuda.is_available()
    )

    mean = 0
    var = 0

    for _, (imgs, _) in enumerate(loader):
        mean += imgs.mean(dim = [0, 2, 3])
        var += (imgs - imgs.mean(dim=[0, 2, 3], keepdim=True)) ** 2.0
    
    mean /= len(loader)
    var /= len(loader)
    std = torch.sqrt(var.mean(dim=[0, 2, 3]))
    
    return mean, std