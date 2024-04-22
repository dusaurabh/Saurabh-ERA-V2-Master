import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import models
from models.Assignment_11_models import ResNet18
import git

train_transforms = A.Compose(
    [
    A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2434, 0.2615)),
    A.PadIfNeeded(min_height=36, min_width=36),
    A.RandomCrop(height=32, width=32),
    A.HorizontalFlip(),
    A.CoarseDropout(max_holes=16, max_height=16, max_width=8, min_holes=16, min_height=16, min_width=8, fill_value=(0.4914, 0.4822, 0.4465), mask_fill_value=None),
    ToTensorV2()
    ]
)

test_transforms = A.Compose(
    [
    A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2434, 0.2615)),
    ToTensorV2()
    ]
)

def train_test_data_loader(train,test):
    SEED = 1
    
    # CUDA?
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)
    
    # For reproducibility
    torch.manual_seed(SEED)
    
    if cuda:
        torch.cuda.manual_seed(SEED)
    
    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=True, batch_size=512, num_workers=0, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
    
    # train dataloader
    train_loader = torch.utils.data.DataLoader(train, **dataloader_args)
    
    # test dataloader
    test_loader = torch.utils.data.DataLoader(test, **dataloader_args)
    return (train_loader, test_loader)




# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
def get_training_images(train_loader):
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    import torchvision
    # show images
    imshow(torchvision.utils.make_grid(images[:4]))
    # print labels
    output = ' '.join(f'{classes[labels[j]]:5s}' for j in range(4))
    return output


# Define function to get misclassified images
def get_misclassified_images(model, test_loader, device):
    misclassified_images = []
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            incorrect_pred = pred.squeeze() != target
            misclassified_images.extend([(img, pred, target) for img, pred, target in zip(data[incorrect_pred], pred[incorrect_pred], target[incorrect_pred])])
            if len(misclassified_images) >= 10:
                break
    return misclassified_images

import os
import sys
import shutil
import time

destination_path = "GRAD-CAM"

# Function to delete directory with retries
def delete_directory_with_retry(directory_path, max_retries=3, retry_delay=1):
    for attempt in range(max_retries):
        try:
            shutil.rmtree(directory_path)
            print(f"Directory '{directory_path}' deleted successfully.")
            return True  # Deletion successful
        except PermissionError as e:
            print(f"PermissionError: {e}. Retrying deletion in {retry_delay} seconds...")
            time.sleep(retry_delay)
    print(f"Failed to delete directory '{directory_path}' after {max_retries} attempts.")
    return False  # Deletion failed

# Attempt to delete the directory
if os.path.exists(destination_path) and os.listdir(destination_path):
    if delete_directory_with_retry(destination_path):
        # Now clone the repository
        git.Repo.clone_from("https://github.com/jacobgil/pytorch-grad-cam.git", destination_path)
    else:
        print("Error: Directory deletion failed. Cloning aborted.")
else:
    # Now clone the repository
    git.Repo.clone_from("https://github.com/jacobgil/pytorch-grad-cam.git", destination_path)

# Get the full path to the repository directory
repo_path = os.path.abspath('GRAD-CAM')
# Append the repository path
sys.path.append(repo_path)


from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def grad_cam_func(model, misclassified_images):
    # Strip the model of its classification layer
    target_layers = [model.layer4[-1]]
    
    # Construct the GradCAM object
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # Define the target category
    target_category = 3  # For example, index 3 corresponds to class 3
    
    # Define the target for GradCAM
    targets = [ClassifierOutputTarget(target_category)]
    
    # Iterate over misclassified images and apply Grad-CAM
    for img, pred, target in misclassified_images:
        # Move the image tensor to CPU and convert to numpy array
        img_np = img.cpu().permute(1, 2, 0).numpy()
        
        # Apply preprocessing transforms to the input numpy array
        preprocessed_input = test_transforms(image=img_np)['image']
        
        # Convert preprocessed input back to tensor
        input_tensor = torch.from_numpy(preprocessed_input).permute(2, 0, 1)
        
        # Compute Grad-CAM
        grayscale_cam = cam(input_tensor=input_tensor.unsqueeze(0), targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        # Visualize Grad-CAM on the original image
        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        
        # Display the visualization
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img_np)
        axes[0].set_title(f'Original Image (Predicted: {pred.item()}, Actual: {target.item()})')
        axes[0].axis('off')
        axes[1].imshow(visualization)
        axes[1].set_title('GradCAM Overlay')
        axes[1].axis('off')
        plt.show()





