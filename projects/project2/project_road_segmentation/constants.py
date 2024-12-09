import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATH_TRAIN_IMGS = 'training/images/'
PATH_TRAIN_GT = 'training/groundtruth/'
PATH_TEST_IMGS = 'testing/'

IMG_PATCH_SIZE = 64
STANDARD_PATCH_SIZE = 16

UNet_OVERLAP = 28
UNet_PATCH_SIZE = 200

TEST_IMG_SIZE = 608
TEST_PATCHES_PER_IMG = 9
NBR_TEST_IMAGES = 50

FOREGROUND_THRESHOLD = 0.25
P_THRESHOLD = 0.5

NUM_WORKERS = 2 if torch.cuda.is_available() else 0
