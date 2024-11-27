import torch
import constants

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
            self.dropout_percentage = 0.15
            self.relu = torch.nn.ReLU()

            # BLOCK-1
            self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            self.batchnorm1 = torch.nn.BatchNorm2d(32)
            self.maxpool1 = torch.nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1))

            # BLOCK-2
            self.conv2_1 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            self.batchnorm2_1 = torch.nn.BatchNorm2d(32)
            self.maxpool2_1 = torch.nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1))

            self.conv2_2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            self.batchnorm2_2 = torch.nn.BatchNorm2d(64)
            self.dropout2 = torch.nn.Dropout(p=self.dropout_percentage)
            self.maxpool2_2 = torch.nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1))
            
            # BLOCK-3
            self.conv3_1 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            self.batchnorm3_1 = torch.nn.BatchNorm2d(64)
            self.maxpool3_1 = torch.nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1))

            self.conv3_2 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            self.batchnorm3_2 = torch.nn.BatchNorm2d(128)
            self.dropout3 = torch.nn.Dropout(p=self.dropout_percentage)
            self.maxpool3_2 = torch.nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1))

            # BLOCK-4
            self.conv4 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            self.batchnorm4 = torch.nn.BatchNorm2d(128)
            self.dropout4 = torch.nn.Dropout(p=self.dropout_percentage)
            
            # FINAL BLOCK (FC)
            self.k_size = constants.IMG_PATCH_SIZE // 32
            self.fc_1 = torch.nn.Linear(in_features=128*self.k_size*self.k_size, out_features=10)
            self.dropout4 = torch.nn.Dropout(p=self.dropout_percentage)
            self.fc_2 = torch.nn.Linear(in_features=10, out_features=2)


        def forward(self, x):
            # BLOCK-1 
            #print("Shape before block 1:", x.shape)
            x = self.relu(self.batchnorm1(self.conv1(x)))
            x = self.maxpool1(x)

            # BLOCK-2
            #print("Shape before block 2:", x.shape)
            x = self.relu(self.batchnorm2_1(self.conv2_1(x)))
            x = self.maxpool2_1(x)
            x = self.relu(self.batchnorm2_2(self.conv2_2(x)))
            x = self.maxpool2_2(self.dropout2(x))

            # BLOCK-3
            #print("Shape before block 3:", x.shape)
            x = self.relu(self.batchnorm3_1(self.conv3_1(x)))
            x = self.maxpool3_1(x)
            x = self.relu(self.batchnorm3_2(self.conv3_2(x)))
            x = self.maxpool3_2(self.dropout3(x))

            # BLOCK-4
            x = self.relu(self.batchnorm4(self.conv4(x)))
            x = self.dropout4(x)

            # FINAL BLOCK (FC)
            #print("Shape before flattening:", x.shape)
            x = torch.flatten(x, start_dim = 1)
            #print("Shape before FC:", x.shape)
            x = self.relu(self.fc_1(x))
            x = self.fc_2(self.dropout4(x))
            return x
        
    return adv_cnn()

def ResNet():
    class ResNet_simple(torch.nn.Module):
        def __init__(self):
            super(adv_cnn, self).__init__()
            self.dropout_percentage = 0.1
            self.relu = torch.nn.ReLU()

            # BLOCK-1
            self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            self.batchnorm1 = torch.nn.BatchNorm2d(32)
            self.maxpool1 = torch.nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1))

            # BLOCK-2
            self.conv2_1 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            self.batchnorm2_1 = torch.nn.BatchNorm2d(32)
            self.maxpool2_1 = torch.nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1))

            self.conv2_2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            self.batchnorm2_2 = torch.nn.BatchNorm2d(64)
            self.dropout2 = torch.nn.Dropout(p=self.dropout_percentage)
            self.maxpool2_2 = torch.nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1))
            
            # BLOCK-3
            self.conv3_1 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            self.batchnorm3_1 = torch.nn.BatchNorm2d(64)
            self.maxpool3_1 = torch.nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1))

            self.conv3_2 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            self.batchnorm3_2 = torch.nn.BatchNorm2d(128)
            self.dropout3 = torch.nn.Dropout(p=self.dropout_percentage)
            self.maxpool3_2 = torch.nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1))

            # BLOCK-4
            self.conv4 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            self.batchnorm4 = torch.nn.BatchNorm2d(128)
            self.dropout4 = torch.nn.Dropout(p=self.dropout_percentage)
            
            # FINAL BLOCK (FC)
            self.k_size = constants.IMG_PATCH_SIZE // 32
            self.fc_1 = torch.nn.Linear(in_features=128*self.k_size*self.k_size, out_features=10)
            self.dropout4 = torch.nn.Dropout(p=self.dropout_percentage)
            self.fc_2 = torch.nn.Linear(in_features=10, out_features=2)


        def forward(self, x):
            # BLOCK-1 
            #print("Shape before block 1:", x.shape)
            x = self.relu(self.batchnorm1(self.conv1(x)))
            x = self.maxpool1(x)

            # BLOCK-2
            #print("Shape before block 2:", x.shape)
            x = self.relu(self.batchnorm2_1(self.conv2_1(x)))
            x = self.maxpool2_1(x)
            x = self.relu(self.batchnorm2_2(self.conv2_2(x)))
            x = self.maxpool2_2(self.dropout2(x))

            # BLOCK-3
            #print("Shape before block 3:", x.shape)
            x = self.relu(self.batchnorm3_1(self.conv3_1(x)))
            x = self.maxpool3_1(x)
            x = self.relu(self.batchnorm3_2(self.conv3_2(x)))
            x = self.maxpool3_2(self.dropout3(x))

            # BLOCK-4
            x = self.relu(self.batchnorm4(self.conv4(x)))
            x = self.dropout4(x)

            # FINAL BLOCK (FC)
            #print("Shape before flattening:", x.shape)
            x = torch.flatten(x, start_dim = 1)
            #print("Shape before FC:", x.shape)
            x = self.relu(self.fc_1(x))
            x = self.fc_2(self.dropout4(x))
            return x
        
    return ResNet_simple()
    

def ResNet_try():
    class ResNet18_try(torch.nn.Module):
        def __init__(self):
            super(ResNet18_try, self).__init__()
            
            self.dropout_percentage = 0.5
            self.relu = torch.nn.ReLU()
            
            # BLOCK-1 (starting block) input=(224x224) output=(56x56)
            # Input (3x128x128) output (32x64x64)
            self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5,5), stride=(2,2), padding=(2,2))
            self.batchnorm1 = torch.nn.BatchNorm2d(32)
            self.maxpool1 = torch.nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1))
            
            # BLOCK-2 (1) input=(56x56) output = (56x56)
            # Input (32x64x64) output (64x32x32)
            self.conv2_1_1 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            self.batchnorm2_1_1 = torch.nn.BatchNorm2d(64)
            self.conv2_1_2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            self.batchnorm2_1_2 = torch.nn.BatchNorm2d(64)
            self.concat_adjust_2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), stride=(1,1), padding=(0,0))
            self.dropout2_1 = torch.nn.Dropout(p=self.dropout_percentage)
            # BLOCK-2 (2)
            self.conv2_2_1 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            self.batchnorm2_2_1 = torch.nn.BatchNorm2d(64)
            self.conv2_2_2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            self.batchnorm2_2_2 = torch.nn.BatchNorm2d(64)
            self.dropout2_2 = torch.nn.Dropout(p=self.dropout_percentage)
            
            # BLOCK-3 (1) input=(56x56) output = (28x28)
            # Input (64x32x32) output (128x16x16)
            self.conv3_1_1 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=(2,2), padding=(1,1))
            self.batchnorm3_1_1 = torch.nn.BatchNorm2d(128)
            self.conv3_1_2 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            self.batchnorm3_1_2 = torch.nn.BatchNorm2d(128)
            self.concat_adjust_3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1,1), stride=(2,2), padding=(0,0))
            self.dropout3_1 = torch.nn.Dropout(p=self.dropout_percentage)
            # BLOCK-3 (2)
            self.conv3_2_1 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            self.batchnorm3_2_1 = torch.nn.BatchNorm2d(128)
            self.conv3_2_2 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            self.batchnorm3_2_2 = torch.nn.BatchNorm2d(128)
            self.dropout3_2 = torch.nn.Dropout(p=self.dropout_percentage)
            
            # Final Block input=(7x7) 
            # Input (128x16x16)
            k_size = constants.IMG_PATCH_SIZE // 8
            self.avgpool = torch.nn.AvgPool2d(kernel_size=(k_size,k_size), stride=(1,1))
            self.fc = torch.nn.Linear(in_features=1*1*128, out_features=2)
            # END
        
        def forward(self, x):
            
            # block 1 --> Starting block
            #print("Shape before block 1:", x.shape)
            x = self.relu(self.batchnorm1(self.conv1(x)))
            op1 = self.maxpool1(x)
            #print("Shape after block 1:", x.shape)
            

            # block2 - 1
            #print("Shape before block 2:", x.shape)
            x = self.relu(self.batchnorm2_1_1(self.conv2_1_1(op1)))    # conv2_1 
            x = self.batchnorm2_1_2(self.conv2_1_2(x))                 # conv2_1
            x = self.dropout2_1(x)
            # block2 - Adjust
            op1 = self.concat_adjust_2(op1)  # Skip connection
            # block2 - Concatenate 1
            op2_1 = self.relu(x + op1)
            # block2 - 2
            x = self.relu(self.batchnorm2_2_1(self.conv2_2_1(op2_1)))  # conv2_2 
            x = self.batchnorm2_2_2(self.conv2_2_2(x))                 # conv2_2
            x = self.dropout2_2(x)
            # op - block2
            op2 = self.relu(x + op2_1)
            #print("Shape after block 2:", x.shape)
        
            
            # block3 - 1[Convolution block]
            #print("Shape before block 3:", x.shape)
            x = self.relu(self.batchnorm3_1_1(self.conv3_1_1(op2)))    # conv3_1
            x = self.batchnorm3_1_2(self.conv3_1_2(x))                 # conv3_1
            x = self.dropout3_1(x)
            # block3 - Adjust
            op2 = self.concat_adjust_3(op2) # SKIP CONNECTION
            # block3 - Concatenate 1
            op3_1 = self.relu(x + op2)
            # block3 - 2[Identity Block]
            x = self.relu(self.batchnorm3_2_1(self.conv3_2_1(op3_1)))  # conv3_2
            x = self.batchnorm3_2_2(self.conv3_2_2(x))                 # conv3_2 
            x = self.dropout3_2(x)
            # op - block3
            op3 = self.relu(x + op3_1)
            #print("Shape after block 3:", x.shape)

            # FINAL BLOCK - classifier 
            #print("Shape before avgpooling:", x.shape)
            x = self.avgpool(op3)
            #print("Shape after avgpooling:", x.shape)
            x = x.reshape(x.shape[0], -1)
            #print("Shape before fc:", x.shape)
            x = self.fc(x)

            return x
        
    return ResNet18_try()


def ResNet():
    class ResNet18(torch.nn.Module):
        def __init__(self, n_classes):
            super(ResNet18, self).__init__()
            
            self.dropout_percentage = 0.3
            self.relu = torch.nn.ReLU()
            
            # BLOCK-1 (starting block) input=(224x224) output=(56x56)
            # Input (128x128) output (32x32)
            self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            self.batchnorm1 = torch.nn.BatchNorm2d(64)
            self.maxpool1 = torch.nn.MaxPool2d(kernel_size=(3,3), stride=(1,1), padding=(1,1))
            
            # BLOCK-2 (1) input=(56x56) output = (56x56)
            # Input (32x32) output (32x32)
            self.conv2_1_1 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            self.batchnorm2_1_1 = torch.nn.BatchNorm2d(64)
            self.conv2_1_2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            self.batchnorm2_1_2 = torch.nn.BatchNorm2d(64)
            self.dropout2_1 = torch.nn.Dropout(p=self.dropout_percentage)
            # BLOCK-2 (2)
            self.conv2_2_1 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            self.batchnorm2_2_1 = torch.nn.BatchNorm2d(64)
            self.conv2_2_2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            self.batchnorm2_2_2 = torch.nn.BatchNorm2d(64)
            self.dropout2_2 = torch.nn.Dropout(p=self.dropout_percentage)
            
            # BLOCK-3 (1) input=(56x56) output = (28x28)
            # Input (32x32) output (16x16)
            self.conv3_1_1 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=(2,2), padding=(1,1))
            self.batchnorm3_1_1 = torch.nn.BatchNorm2d(128)
            self.conv3_1_2 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            self.batchnorm3_1_2 = torch.nn.BatchNorm2d(128)
            self.concat_adjust_3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1,1), stride=(2,2), padding=(0,0))
            self.dropout3_1 = torch.nn.Dropout(p=self.dropout_percentage)
            # BLOCK-3 (2)
            self.conv3_2_1 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            self.batchnorm3_2_1 = torch.nn.BatchNorm2d(128)
            self.conv3_2_2 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            self.batchnorm3_2_2 = torch.nn.BatchNorm2d(128)
            self.dropout3_2 = torch.nn.Dropout(p=self.dropout_percentage)
            
            # BLOCK-4 (1) input=(28x28) output = (14x14)
            # Input (16x16) output (8x8)
            self.conv4_1_1 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=(2,2), padding=(1,1))
            self.batchnorm4_1_1 = torch.nn.BatchNorm2d(256)
            self.conv4_1_2 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            self.batchnorm4_1_2 = torch.nn.BatchNorm2d(256)
            self.concat_adjust_4 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1,1), stride=(2,2), padding=(0,0))
            self.dropout4_1 = torch.nn.Dropout(p=self.dropout_percentage)
            # BLOCK-4 (2)
            self.conv4_2_1 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            self.batchnorm4_2_1 = torch.nn.BatchNorm2d(256)
            self.conv4_2_2 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            self.batchnorm4_2_2 = torch.nn.BatchNorm2d(256)
            self.dropout4_2 = torch.nn.Dropout(p=self.dropout_percentage)
            
            # BLOCK-5 (1) input=(14x14) output = (7x7)
            # Input (8x8) output (4x4)
            self.conv5_1_1 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride=(2,2), padding=(1,1))
            self.batchnorm5_1_1 = torch.nn.BatchNorm2d(512)
            self.conv5_1_2 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            self.batchnorm5_1_2 = torch.nn.BatchNorm2d(512)
            self.concat_adjust_5 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1,1), stride=(2,2), padding=(0,0))
            self.dropout5_1 = torch.nn.Dropout(p=self.dropout_percentage)
            # BLOCK-5 (2)
            self.conv5_2_1 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            self.batchnorm5_2_1 = torch.nn.BatchNorm2d(512)
            self.conv5_2_2 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            self.batchnorm5_2_2 = torch.nn.BatchNorm2d(512)
            self.dropout5_2 = torch.nn.Dropout(p=self.dropout_percentage)
            
            # Final Block input=(7x7) 
            # Input (4x4)
            self.avgpool = torch.nn.AvgPool2d(kernel_size=(4,4), stride=(2,2), padding=(1,1))
            self.fc = torch.nn.Linear(in_features=8*8*512, out_features=1000)
            self.out = torch.nn.Linear(in_features=1000, out_features=n_classes)
            # END
        
        def forward(self, x):
            
            # block 1 --> Starting block
            x = self.relu(self.batchnorm1(self.conv1(x)))
            op1 = self.maxpool1(x)
            
            
            # block2 - 1
            x = self.relu(self.batchnorm2_1_1(self.conv2_1_1(op1)))    # conv2_1 
            x = self.batchnorm2_1_2(self.conv2_1_2(x))                 # conv2_1
            x = self.dropout2_1(x)
            # block2 - Adjust - No adjust in this layer as dimensions are already same
            # block2 - Concatenate 1
            op2_1 = self.relu(x + op1)
            # block2 - 2
            x = self.relu(self.batchnorm2_2_1(self.conv2_2_1(op2_1)))  # conv2_2 
            x = self.batchnorm2_2_2(self.conv2_2_2(x))                 # conv2_2
            x = self.dropout2_2(x)
            # op - block2
            op2 = self.relu(x + op2_1)
        
            
            # block3 - 1[Convolution block]
            x = self.relu(self.batchnorm3_1_1(self.conv3_1_1(op2)))    # conv3_1
            x = self.batchnorm3_1_2(self.conv3_1_2(x))                 # conv3_1
            x = self.dropout3_1(x)
            # block3 - Adjust
            op2 = self.concat_adjust_3(op2) # SKIP CONNECTION
            # block3 - Concatenate 1
            op3_1 = self.relu(x + op2)
            # block3 - 2[Identity Block]
            x = self.relu(self.batchnorm3_2_1(self.conv3_2_1(op3_1)))  # conv3_2
            x = self.batchnorm3_2_2(self.conv3_2_2(x))                 # conv3_2 
            x = self.dropout3_2(x)
            # op - block3
            op3 = self.relu(x + op3_1)
            
            
            # block4 - 1[Convolition block]
            x = self.relu(self.batchnorm4_1_1(self.conv4_1_1(op3)))    # conv4_1
            x = self.batchnorm4_1_2(self.conv4_1_2(x))                 # conv4_1
            x = self.dropout4_1(x)
            # block4 - Adjust
            op3 = self.concat_adjust_4(op3) # SKIP CONNECTION
            # block4 - Concatenate 1
            op4_1 = self.relu(x + op3)
            # block4 - 2[Identity Block]
            x = self.relu(self.batchnorm4_2_1(self.conv4_2_1(op4_1)))  # conv4_2
            x = self.batchnorm4_2_2(self.conv4_2_2(x))                 # conv4_2
            x = self.dropout4_2(x)
            # op - block4
            op4 = self.relu(x + op4_1)

            
            # block5 - 1[Convolution Block]
            x = self.relu(self.batchnorm5_1_1(self.conv5_1_1(op4)))    # conv5_1
            x = self.batchnorm5_1_2(self.conv5_1_2(x))                 # conv5_1
            x = self.dropout5_1(x)
            # block5 - Adjust
            op4 = self.concat_adjust_5(op4) # SKIP CONNECTION
            # block5 - Concatenate 1
            op5_1 = self.relu(x + op4)
            # block5 - 2[Identity Block]
            x = self.relu(self.batchnorm5_2_1(self.conv5_2_1(op5_1)))  # conv5_2
            x = self.batchnorm5_2_1(self.conv5_2_1(x))                 # conv5_2
            x = self.dropout5_2(x)
            # op - block5
            op5 = self.relu(x + op5_1)


            # FINAL BLOCK - classifier 
            #print("Shape before avgpooling:", x.shape)
            x = self.avgpool(op5)
            #print("Shape after avgpooling:", x.shape)
            x = x.reshape(x.shape[0], -1)
            #print("Shape before fc:", x.shape)
            x = self.relu(self.fc(x))
            x = self.out(x)

            return x
        
    return ResNet18(n_classes = 2)