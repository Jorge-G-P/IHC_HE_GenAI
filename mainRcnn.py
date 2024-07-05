#from engine import train_one_epoch, evaluate
import torch
#import utils
import maskRcnn
from dataset import SimDataset
from maskdataset import MaskDataset
import config
from train import train_model
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import re

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Regular expression for numerical sorting
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

# Paths to the image and mask directories
imgs_train_path = config.imgs_train_path
imgs_val_path = config.imgs_val_path
masks_train_path = config.masks_train_path
masks_val_path = config.masks_val_path

# Function to get all file paths from a folder
def get_file_paths(folder):
    return sorted([os.path.join(folder, filename) for filename in os.listdir(folder)], key=numericalSort)

# Get file paths for images and masks
imgs_train_paths = get_file_paths(imgs_train_path)
imgs_val_paths = get_file_paths(imgs_val_path)
masks_train_paths = get_file_paths(masks_train_path)
masks_val_paths = get_file_paths(masks_val_path)

# Define the transformations (COMENTAR)
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # imagenet
])

# Create the datasets
train_set = MaskDataset(imgs_train_paths, masks_train_paths, transform=trans)
val_set = MaskDataset(imgs_val_paths, masks_val_paths, transform=trans)



# Create the dataloaders
batch_size = 25

dataloaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
    'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
}



#num_class = 1
#model = ResNetUNet(num_class).to(device)
# optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)
# model = train_model(model, dataloaders, optimizer_ft, exp_lr_scheduler, device, num_epochs=1)


###########################
# our dataset has two classes only - background and person
num_classes = 2

# # use our dataset and defined transformations
# dataset = PennFudanDataset('data/PennFudanPed', get_transform(train=True))
# dataset_test = PennFudanDataset('data/PennFudanPed', get_transform(train=False))

# # split the dataset in train and test set
# indices = torch.randperm(len(dataset)).tolist()
# dataset = torch.utils.data.Subset(dataset, indices[:-50])
# dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

# # define training and validation data loaders
# data_loader = torch.utils.data.DataLoader(
#     dataset,
#     batch_size=2,
#     shuffle=True,
#     collate_fn=utils.collate_fn
# )

# data_loader_test = torch.utils.data.DataLoader(
#     dataset_test,
#     batch_size=1,
#     shuffle=False,
#     collate_fn=utils.collate_fn
# )

# get the model using our helper function
model = maskRcnn.get_model_instance_segmentation(num_classes)
model.to(device)


optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=0.005, 
#                          momentum=0.9,
                          weight_decay=0.0005)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 2
model = train_model(model, dataloaders, optimizer, exp_lr_scheduler, device, num_epochs, batch_size)


# # construct an optimizer
# params = [p for p in model.parameters() if p.requires_grad]
# optimizer = torch.optim.SGD(
#     params,
#     lr=0.005,
#     momentum=0.9,
#     weight_decay=0.0005
# )

# # and a learning rate scheduler
# lr_scheduler = torch.optim.lr_scheduler.StepLR(
#     optimizer,
#     step_size=3,
#     gamma=0.1
# )

# for epoch in range(num_epochs):
#     # train for one epoch, printing every 10 iterations
#     train_one_epoch(model, optimizer, dataloaders['train'], device, epoch, print_freq=10)
#     # update the learning rate
#     lr_scheduler.step()
#     # evaluate on the test dataset
#     evaluate(model, dataloaders['test'], device=device)

