import torch
import constants
from torchvision import transforms

def basic_cnn():
    model = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 5, padding = 2),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1),

        torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, padding = 2, stride = 2),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size = 5, stride = 4, padding = 2),

        torch.nn.Flatten(),
        torch.nn.Linear(
            in_features=(constants.IMG_PATCH_SIZE // 16)**2 * 64, 
            out_features=2
            ),
    )
    return model

def adv_cnn():
    class adv_cnn(torch.nn.Module):
        def __init__(self):
            super(adv_cnn, self).__init__()
            self.dropout_percentage = 0.02
            self.relu = torch.nn.LeakyReLU(negative_slope = 0.01)

            # BLOCK-1
            self.conv1_1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            self.batchnorm1_1 = torch.nn.BatchNorm2d(16)
            self.dropout1_1 = torch.nn.Dropout(p=self.dropout_percentage)
            self.maxpool1_1 = torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

            self.conv1_2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            self.batchnorm1_2 = torch.nn.BatchNorm2d(32)
            self.dropout1_2 = torch.nn.Dropout(p=self.dropout_percentage)
            self.maxpool1_2 = torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

            # BLOCK-2
            self.conv2_1 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            self.batchnorm2_1 = torch.nn.BatchNorm2d(32)
            self.dropout2_1 = torch.nn.Dropout(p=self.dropout_percentage)
            self.maxpool2_1 = torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

            self.conv2_2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            self.batchnorm2_2 = torch.nn.BatchNorm2d(64)
            self.dropout2_2 = torch.nn.Dropout(p=self.dropout_percentage)
            self.maxpool2_2 = torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
            
            # BLOCK-3
            self.conv3_1 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            self.batchnorm3_1 = torch.nn.BatchNorm2d(64)
            self.dropout3_1 = torch.nn.Dropout(p=self.dropout_percentage)
            
            self.conv3_2 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            self.batchnorm3_2 = torch.nn.BatchNorm2d(128)
            self.adjust_3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1,1), stride=(1,1))
            self.dropout3_2 = torch.nn.Dropout(p=self.dropout_percentage)
            self.maxpool3 = torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

            # BLOCK-4
            self.conv4_1 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            self.batchnorm4_1 = torch.nn.BatchNorm2d(128)
            self.dropout4_1 = torch.nn.Dropout(p=self.dropout_percentage)

            self.conv4_2 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            self.batchnorm4_2 = torch.nn.BatchNorm2d(256)
            self.adjust_4 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1,1), stride=(1,1))
            self.dropout4_2 = torch.nn.Dropout(p=self.dropout_percentage)
            self.maxpool4 = torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

            # BLOCK-5
            self.conv5 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            self.batchnorm5 = torch.nn.BatchNorm2d(256)
 
            
            # FINAL BLOCK (FC)
            self.k_size = constants.IMG_PATCH_SIZE // 64
            self.fc_1 = torch.nn.Linear(in_features=256*self.k_size*self.k_size, out_features=256)
            self.dropout_fc = torch.nn.Dropout(p=self.dropout_percentage)
            self.fc_2 = torch.nn.Linear(in_features=256, out_features=2)


        def forward(self, x):
            #print(f'Dropout: {self.dropout_percentage}')
            # BLOCK-1 
            #print("Shape before block 1:", x.shape)
            x = self.relu(self.batchnorm1_1(self.conv1_1(x)))
            x = self.maxpool1_1(self.dropout1_1(x))
            x = self.relu(self.batchnorm1_2(self.conv1_2(x)))
            x = self.maxpool1_2(self.dropout1_2(x))

            # BLOCK-2
            #print("Shape before block 2:", x.shape)
            x = self.relu(self.batchnorm2_1(self.conv2_1(x)))
            x = self.maxpool2_1(self.dropout2_1(x))
            x = self.relu(self.batchnorm2_2(self.conv2_2(x)))
            op2 = self.maxpool2_2(self.dropout2_2(x))

            # BLOCK-3
            #print("Shape before block 3:", x.shape)
            x = self.relu(self.batchnorm3_1(self.conv3_1(op2)))
            x = self.dropout3_1(x)
            x = self.relu(self.batchnorm3_2(self.conv3_2(x)))
            x = self.dropout3_2(x)

            op2 = self.adjust_3(op2)
            x = self.relu(x + op2)
            op3 = self.maxpool3(x)

            # BLOCK-4
            x = self.relu(self.batchnorm4_1(self.conv4_1(op3)))
            x = self.dropout4_1(x)
            x = self.relu(self.batchnorm4_2(self.conv4_2(x)))
            x = self.dropout4_2(x)

            op3 = self.adjust_4(op3)
            x = self.relu(x + op3)
            x = self.maxpool4(x)

            # BLOCK-5
            x = self.batchnorm5(self.conv5(x))

            # FINAL BLOCK (FC)
            #print("Shape before flattening:", x.shape)
            x = torch.flatten(x, start_dim = 1)
            #print("Shape before FC:", x.shape)
            x = self.relu(self.fc_1(x))
            x = self.fc_2(self.dropout_fc(x))
            return x
        
    return adv_cnn()

def UNet():
    class UNet(torch.nn.Module):
        def __init__(self):
            super(UNet, self).__init__()
            self.enc1 = Encoder(in_channels = 3, out_channels = 64)
            self.enc2 = Encoder(in_channels = 64, out_channels = 128)
            self.enc3 = Encoder(in_channels = 128, out_channels = 256)
            self.enc4 = Encoder(in_channels = 256, out_channels = 512)
            #self.enc5 = Encoder(in_channels = 128, out_channels = 256)

            self.bottleneck_conv = torch.nn.Conv2d(in_channels = 512, out_channels = 1024, kernel_size = 3, padding = 1)
            self.bottleneck_batch = torch.nn.BatchNorm2d(1024)
            self.bottleneck_relu = torch.nn.ReLU()

            self.dec1 = Decoder(in_channels = 1024, out_channels = 512)
            self.dec2 = Decoder(in_channels = 512, out_channels = 256)
            self.dec3 = Decoder(in_channels = 256, out_channels = 128)
            self.dec4 = Decoder(in_channels = 128, out_channels = 64)
            
            self.final_conv = torch.nn.Conv2d(in_channels = 64, out_channels = 1, kernel_size = 1)
            self.crop = transforms.CenterCrop((constants.STANDARD_PATCH_SIZE, constants.STANDARD_PATCH_SIZE))
        
        def forward(self, x):
            enc1, skip1 = self.enc1(x)
            enc2, skip2 = self.enc2(enc1)
            enc3, skip3 = self.enc3(enc2)
            enc4, skip4 = self.enc4(enc3)

            b = self.bottleneck_relu(self.bottleneck_batch(self.bottleneck_conv(enc4)))

            dec1 = self.dec1(b, skip4)
            dec2 = self.dec2(dec1, skip3)
            dec3 = self.dec3(dec2, skip2)
            dec4 = self.dec4(dec3, skip1)
            crop = self.crop(self.final_conv(dec4))
            return crop

    class Encoder(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super(Encoder, self).__init__()
            self.dropout_percentage = 0.2
            self.relu = torch.nn.ReLU()
            self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
            self.batchnorm1 = torch.nn.BatchNorm2d(out_channels)
            self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
            self.batchnorm2 = torch.nn.BatchNorm2d(out_channels)
            self.dropout = torch.nn.Dropout(p=self.dropout_percentage)
            self.maxpool = torch.nn.MaxPool2d(kernel_size = 2, stride = 2)

        def forward(self, x):
            x = self.relu(self.batchnorm1(self.conv1(x)))
            x = self.relu(self.batchnorm2(self.conv2(x)))
            out = self.maxpool(self.dropout(x))
            return out, x

    class Decoder(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super(Decoder, self).__init__()
            self.relu = torch.nn.ReLU()
            self.deconv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 2, stride = 2) 
            self.conv1 = torch.nn.Conv2d(2*out_channels, out_channels, kernel_size = 3, padding = 1)
            self.batchnorm1 = torch.nn.BatchNorm2d(out_channels)
            self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
            self.batchnorm2 = torch.nn.BatchNorm2d(out_channels)

        def forward(self, x, skip):
            x = self.deconv(x)
            x = torch.cat([x, skip], dim = 1)
            x = self.relu(self.batchnorm1(self.conv1(x)))
            x = self.relu(self.batchnorm2(self.conv2(x)))
            return x

    return UNet()