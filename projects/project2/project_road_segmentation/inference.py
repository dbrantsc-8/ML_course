import torch
from torchvision import transforms
import numpy as np
import skimage.transform
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import models
import utilities
import constants
import data_process

class TestModel:
    def __init__(self, model, path_test_imgs, mean, std, device):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.path_test_imgs = path_test_imgs
    
    def predict(self):
        test_data = data_process.TestImgsDataset(self.path_test_imgs)
        test_data.excecute()

        test_loader = torch.utils.data.DataLoader(
            dataset = test_data,
            batch_size = 16,
            shuffle = False,
            drop_last = False,
            num_workers = constants.NUM_WORKERS,
            pin_memory = torch.cuda.is_available()
        )

        with torch.no_grad():
            preds = []
            for data in test_loader:
                data = data.to(self.device)
                output = self.model(data).squeeze(1)
                pred = (output > constants.P_THRESHOLD).float().to(self.device)
                preds.extend(pred.cpu().numpy())
        
        crop_transform = transforms.CenterCrop((constants.UNet_PATCH_SIZE, constants.UNet_PATCH_SIZE))
        preds = crop_transform(torch.Tensor(np.array(preds)))

        pred_imgs = [
            utilities.label_to_img(
                imgwidth = 600,
                imgheight = 600,
                w = constants.UNet_PATCH_SIZE,
                h = constants.UNet_PATCH_SIZE,
                labels = preds[i * constants.TEST_PATCHES_PER_IMG : (i + 1) * constants.TEST_PATCHES_PER_IMG],
            )
            for i in range(constants.NBR_TEST_IMAGES)
        ]
        return pred_imgs
    
    def create_submission(self, path_submission, pred_imgs):
        pred_imgs = skimage.transform.resize(np.array(pred_imgs), (len(pred_imgs), constants.TEST_IMG_SIZE, constants.TEST_IMG_SIZE))
        plt.figure()
        img = mpimg.imread('testing/test_10/test_10.png')
        plt.imshow(img)  # Display the original image
        plt.imshow(pred_imgs[9], cmap="Reds", alpha=0.3)  # Overlay ground truth with transparency
        plt.show()
        utilities.masks_to_submission(path_submission, pred_imgs)

def main():
    device = constants.DEVICE
    path_test_imgs = constants.PATH_TEST_IMGS
    model = models.UNet()
    path_model = 'models/UNet_200_best.pth'
    path_submission = 'submissions/delete_me.csv'

    model.load_state_dict(torch.load(path_model, map_location=torch.device(device)))
    predictor = TestModel(model, path_test_imgs, device)
    pred_imgs = predictor.predict()
    predictor.create_submission(path_submission, pred_imgs)

if __name__ == "__main__":
    main()

