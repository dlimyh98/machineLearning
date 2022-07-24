import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

##### Optional Speedups #####
transform = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


################################################# DATASETS AND DATALOADERS #####################################################
## torch.utils.data.Dataset    = stores the samples and their corresponding labels
## torch.utils.data.Dataloader = wraps an iterable around Dataset to enable easy access to the sample

class CatDogDataset(torch.utils.data.Dataset):
    ## custom Dataset class must implement __init__, __len__, __getitem__

    ## run once when instantiating Dataset object
    def __init__(self, image_directory, image_size=224, is_train=True, transform=None):
        self.image_directory = image_directory
        self.image_size = image_size
        self.is_train = is_train
        self.transform = transform
        self.image_ids = []
        self.image_paths = []
        self.image_labels = []

        ## label the images
        if self.is_train:
            self._initialize_train()
        elif not self.is_train:
            self._initialize_test()
    
    def _initialize_train(self):
        for label_idx in [0, 1]:
            label_name = "cats" if label_idx == 0 else "dogs"    # cats = 0, dogs = 1
            for dirname, _, filenames in os.walk(f"{self.image_directory}/train/{label_name}"):
                ## eg. filenames = cat_1.jpg
                for filename in filenames:
                    self.image_ids.append(filename[:-4])
                    self.image_paths.append(f"{self.image_directory}/train/{label_name}/{filename}")
                    self.image_labels.append(label_idx)
    
    def _initialize_test(self):
        for dirname, _, filenames in os.walk(f"{self.image_directory}/test"):
            for filename in filenames:
                ## eg. filenames = test_1.jpg
                self.image_ids.append(filename[:-4])
                self.image_paths.append(f"{self.image_directory}/test/{filename}")
                self.image_labels.append(-1)

    ## returns number of samples in dataset
    def __len__(self):
        return len(self.image_labels)

    ## loads and returns sample from dataset at the given index idx
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = self.image_paths[idx]
        image_label = self.image_labels[idx]
        resize_and_crop = transforms.Compose([transforms.Resize(self.image_size),
                                              transforms.CenterCrop(self.image_size),
                                              transforms.ConvertImageDtype(torch.float32)])
        image_original = resize_and_crop(torchvision.io.read_image(image_path))
        image_transformed = image_original

        if self.transform:
            image_transformed = torchvision.transforms.functional.resize(image_original, (64,64))

        return image_id, image_original, image_transformed, image_label


dataset = CatDogDataset("C:/Users/Damien/Downloads/CS3244/AssignmentImages", is_train=True, transform=transform)
test_dataset = CatDogDataset("C:/Users/Damien/Downloads/CS3244/AssignmentImages", is_train=False, transform=transform)

#dataset = CatDogDataset("/kaggle/input/cs3244-assignment-2", is_train=True, transform=transform)
#test_dataset = CatDogDataset("/kaggle/input/cs3244-assignment-2", is_train=False, transform=transform)


##################################### Splitting dataset into Training and Validation #################################################
batch_size = 50
validation_split = 0.2
shuffle_dataset = True
random_seed = 42

## Creating data indices for training and validation splits:
dataset_size = len(dataset)     # 4000 cats + 4000 dogs = 8000 total
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

## Sampling elements randomly from given list of indices, without replacement
training_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

training_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=training_sampler)
validation_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=validation_sampler)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

'''
# Display image and label.
training_id, training_image_original, training_image_transformed, training_image_label = next(iter(training_dataloader))
print(f"Feature batch shape: {training_image_transformed.size()}")    ## tensor[50,3,64,64], 50 images per batch, 3 channels (RGB), 64x64 height and width
print(f"Labels batch shape: {training_image_label.size()}")           ## tensor[50], one label for each of the 50 images per batch
img = training_image_transformed[0].squeeze()
img = img.swapaxes(0,1)
img = img.swapaxes(1,2)
label = training_image_label[0]

print(f"Label: {label}")
plt.imshow(img, cmap="gray")
plt.show()
'''


##################################### Define Convolution Neural Network #################################################
class Net(nn.Module):
    def __init__(self):
        super().__init__()    # super() returns object representing parent class, which we then construct with __init__

        self.conv1 = nn.Conv2d(3, 50, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(50, 100, 7)
        self.pool2 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(100 * 12 * 12, 120)
        self.fc2 = nn.Linear(120, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

'''
learnable_parameters = list(net.parameters())
print(len(learnable_parameters))
print(learnable_parameters[0].size())

for name, param in net.named_parameters():
    if param.requires_grad:
        print(name, param.data)
'''


##################################### Define Loss Function and Optimizer #################################################
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9,0.999), eps=1e-08)


##################################### Train the Network #################################################
for epoch in range(10):  # loop over the dataset multiple times
    running_loss = 0.0

    ## training_dataloader = 6400 random samples
    ## minibatch of 50 sent to CNN in one go
    for i, data in enumerate(training_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        images_id, images_original, images_transformed, images_label = data

        # clear the parameter gradients
        optimizer.zero_grad()

        # Forward Propagation, Backward Propagation, Optimize(minimize the errors)
        outputs = net(images_transformed)
        loss = criterion(outputs.squeeze(1), images_label.float())
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # print every 10 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
            running_loss = 0.0

print('Finished Training')


##################################### Validate the Network #################################################
correct = 0
total = len(val_indices)

with torch.no_grad():
    for data in validation_dataloader:
        images_id, images_original, images_transformed, images_label = data
        imageds_transformed = images_transformed.to(device)
        images_label = images_label.to(device)
        outputs = torch.nn.Sigmoid()(net(images_transformed))
        predicted = outputs.data.round()

        for i in range(len(images_id)):
            if (images_label[i] == int(predicted[i].item())):
                correct += 1

print("Number of correctly classified images = ", correct) 
print(f'Accuracy of the network on the 1600 validation images: {100 * correct // total} %')


##################################### Test the Network #################################################
output_ids = []
output_predictions = []

with torch.no_grad():
    for data in test_dataloader:
        ids, _, inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = torch.nn.Sigmoid()(net(inputs))
        predicted = outputs.data.round()
        for i in range(len(ids)):
            output_ids.append(ids[i])
            output_predictions.append(int(predicted[i].item()))

df = pd.DataFrame(data={"Id": output_ids, "Category": output_predictions})
df.to_csv("submission.csv", index=False)


'''
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        # in_channels  = number of channels in input image
        # out_channels = number of channels produced by convolution, number of kernels used
        # padding      = padding applied to BOTH sides of input
        # trainable parameters = parameters UPDATED when network is trained, for (n,n) kernel have (n*n+1) paramameters including bias

        ## input image = (3 * 277 * 277)
        ##               - (277 * 277) RED
        ##               - (277 * 277) GREEN
        ##               - (277 * 277) BLUE
        ## 96 kernels of size (3 * 11 * 11) used, producing 96 out_channels
        ##    - each RGB "layer" (277 * 277) of image convoluted with (11 * 11) layer of ALL 96 kernels, 
        ##      convolution result SUMMED across ALL 96 kernels to produce output on SINGLE square of ONE out_channel
        ##    - slide the kernel across input image according to stride, repeat above

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(11,11), stride=4, padding=0)
        self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5,5), stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3,3), stride= 1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3,3), stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3,3), stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=9216, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096 , out_features=1)


    def forward(self,x):
        ## Convolutional and Pooling Layers
        x = F.relu(self.conv1(x))
        x = self.maxPool(x)
        x = F.relu(self.conv2(x))
        x = self.maxPool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.maxPool(x)

        ## Fully Connected Layers
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

alexNet = AlexNet()
'''
