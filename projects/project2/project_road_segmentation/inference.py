import torch
from torchvision import transforms
import numpy as np
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
        self.mean = mean
        self.std = std
    
    def predict(self):
        transform = transforms.Compose([transforms.Normalize(self.mean.tolist(), self.std.tolist())])
        test_data = data_process.TestImgsDataset(self.path_test_imgs, transform)
        test_data.excecute()

        test_loader = torch.utils.data.DataLoader(
            dataset = test_data,
            batch_size = 128,
            shuffle = False,
            drop_last = False,
            num_workers = constants.NUM_WORKERS,
            pin_memory = torch.cuda.is_available()
        )

        with torch.no_grad():
            preds = []
            for data in test_loader:
                data = data.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True) 
                preds.extend(pred.cpu().numpy())
        
        pred_imgs = [
            utilities.label_to_img(
                imgwidth = constants.TEST_IMG_SIZE,
                imgheight = constants.TEST_IMG_SIZE,
                w = constants.IMG_PATCH_SIZE,
                h = constants.IMG_PATCH_SIZE,
                labels = preds[i * constants.TEST_PATCHES_PER_IMG**2 : (i + 1) * constants.TEST_PATCHES_PER_IMG**2],
            )
            for i in range(constants.NBR_TEST_IMAGES)
        ]

        return pred_imgs
    
    def create_submission(self, path_submission, pred_imgs):
        utilities.masks_to_submission(path_submission, np.array(pred_imgs))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.basic_cnn()
    model.load_state_dict(torch.load("models/basic_cnn_with_augmentation.pth", map_location=torch.device(device)))
    path_test_imgs = 'testing/'
    path_submission = 'submissions/augmented.csv'

    # Compute mean and standard deviation over whole training set, only needs to be done once
    """train_data = data_process.TrainImgsDataset(path_imgs = 'training/images/', 
                                    path_gts = 'training/groundtruth/', 
                                    train = True,
                                    num_images = 100, 
                                    transform = False)
    train_data.excecute()

    mean_std_loader = torch.utils.data.DataLoader(
        dataset = train_data,
        batch_size = 128,
        shuffle = False,
        drop_last = True,
        num_workers = constants.NUM_WORKERS,
        pin_memory = torch.cuda.is_available()
    )

    mean, std = utilities.compute_mean_std(mean_std_loader)"""

    # Mean and sandard deviation over whole training set
    mean = torch.tensor([0.3330, 0.3301, 0.2958])
    std = torch.tensor([0.1834, 0.1784, 0.1769])

    predictor = TestModel(model, path_test_imgs, mean, std, device)

    pred_imgs = predictor.predict()

    predictor.create_submission(path_submission, pred_imgs)

if __name__ == "__main__":
    main()

