import matplotlib.pyplot as plt
import numpy as np
import torch
import utilities
import models
import constants

class TrainModel:
    def __init__(self, model, params, device):
        self.model = model.to(device)
        self.device = device
        self.params = params
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr = params['lr'], 
            weight_decay = params['weight_decay']
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max = params['T_max'],
        )
        self.criterion = torch.nn.functional.cross_entropy
    
    def evaluate_pred(self, output, target):
        pred = output.argmax(dim = 1, keepdim = True).to(self.device)
        pred = pred.view(-1)
        target = target.view(-1)

        tp = ((pred == 1) & (target == 1)).sum().item() # True positives
        tn = ((pred == 0) & (target == 0)).sum().item() # True negatives
        fp = ((pred == 1) & (target == 0)).sum().item() # False positives
        fn = ((pred == 0) & (target == 1)).sum().item() # False negatives

        return tp, tn, fp, fn
    
    def compute_scores(self, tp, tn, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        return f1, acc
    
    def train_epoch(self, train_loader, class_weights, epoch):
        self.model.train()
        loss_hist, acc_hist, f1_hist, lr_hist = [], [], [], []

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            class_weights = class_weights.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target, class_weights)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            tp, tn, fp, fn = self.evaluate_pred(output, target)
            f1, acc = self.compute_scores(tp, tn, fp, fn)

            loss_hist.append(loss.item())
            f1_hist.append(f1)
            acc_hist.append(acc)
            lr_hist.append(self.scheduler.get_last_lr()[0])

            if batch_idx % (len(train_loader.dataset) // len(data) // 10) == 0:
                print(
                    f"Train Epoch: {epoch}-{batch_idx:03d} "
                    f"batch_loss={loss.item():0.2e} "
                    f"batch_acc={acc:0.3f} "
                    f"batch_f1={f1:0.3f} "
                    f"lr={self.scheduler.get_last_lr()[0]:0.3e} "
                )

        return loss_hist, f1_hist, acc_hist, lr_hist

    @torch.no_grad()
    def evaluate(self, valid_loader):
        self.model.eval() 
        valid_loss = 0
        tot_tp, tot_tn, tot_fp, tot_fn = 0, 0, 0, 0

        for data, target in valid_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            valid_loss += self.criterion(output, target).item() * len(data)

            tp, tn, fp, fn = self.evaluate_pred(output, target)
            tot_tp += tp
            tot_tn += tn
            tot_fp += fp
            tot_fn += fn

        valid_loss /= len(valid_loader.dataset)
        valid_f1, valid_acc = self.compute_scores(tot_tp, tot_tn, tot_fp, tot_fn)

        print(
            "Validation set: Average loss: {:.4f}, Accuracy: {:.0f} %, F1-score: {:.4f}".format(
                valid_loss,
                100.0 * valid_acc,
                valid_f1,
            )
        )

        return valid_loss, valid_f1, valid_acc
    
    def plots(self, tr_f1, tr_acc, tr_loss, valid_f1, valid_acc, valid_loss, lr_hist, num_epochs):
        n_train = len(tr_acc)
        t_train = num_epochs * np.arange(n_train) / n_train
        t_val = np.arange(1, num_epochs + 1)

        plt.figure(figsize=(6.4 * 4, 4.8))
        plt.subplots_adjust(wspace=0.4)
        plt.subplot(1, 4, 1)
        plt.plot(t_train, tr_loss, label="Train")
        plt.plot(t_val, valid_loss, label="Validation")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.subplot(1, 4, 2)
        plt.plot(t_train, tr_f1, label="Train")
        plt.plot(t_val, valid_f1, label="Validation")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("F1-score")

        plt.subplot(1, 4, 3)
        plt.plot(t_train, tr_acc, label="Train")
        plt.plot(t_val, valid_acc, label="Validation")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")

        plt.subplot(1, 4, 4)
        plt.plot(t_train, lr_hist)
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")

        plt.show()
        

def main():
    params = {
        'lr': 1e-3,
        'weight_decay': 5e-1,
        'num_epochs': 5,
        'num_images': 80,
        'data_augmentation': False,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.ResNet()
    train_loader, valid_loader, class_weights = utilities.get_dataloaders(
        num_images = params['num_images'], 
        batch_size = 64,
        data_aug = params['data_augmentation']
    )

    params['T_max'] = 2*(len(train_loader.dataset) * params['num_epochs']) // train_loader.batch_size

    trainer = TrainModel(model, params, device)

    tr_loss_hist, tr_f1_hist, tr_acc_hist, lr_hist = [], [], [], []
    valid_loss_hist, valid_f1_hist, valid_acc_hist = [], [], []

    for epoch in range(1, params['num_epochs'] + 1):
        tr_loss, tr_f1, tr_acc, lr = trainer.train_epoch(train_loader, class_weights, epoch)
        tr_loss_hist.extend(tr_loss)
        tr_f1_hist.extend(tr_f1)
        tr_acc_hist.extend(tr_acc)
        lr_hist.extend(lr)

        valid_loss, valid_f1, valid_acc = trainer.evaluate(valid_loader)
        valid_loss_hist.append(valid_loss)
        valid_f1_hist.append(valid_f1)
        valid_acc_hist.append(valid_acc)
    
    trainer.plots(
        tr_f1_hist, tr_acc_hist, tr_loss_hist, valid_f1_hist, valid_acc_hist, valid_loss_hist, lr_hist, params['num_epochs']
    )

    torch.save(model.state_dict(), "/content/drive/MyDrive/ML_project2/models/advanced_model.pth")

if __name__ == "__main__":
    #print(f'patch size: {constants.IMG_PATCH_SIZE}')
    main()