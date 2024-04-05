import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

train_transforms = A.Compose(
    [
    A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2434, 0.2615)),
    A.PadIfNeeded(min_height=36, min_width=36),
    A.RandomCrop(height=32, width=32),
    A.HorizontalFlip(),
    A.CoarseDropout(max_holes=1, max_height=8, max_width=8, min_holes=1, min_height=8, min_width=8, fill_value=(0.4914, 0.4822, 0.4465), mask_fill_value=None),
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
def get_training_images(train_loader,):
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




train_losses = []
test_losses = []
train_acc = []
test_acc = []

def train(model, device, train_loader, optimizer, epoch):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = F.cross_entropy(y_pred, target)
    train_losses.append(loss)

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm
    
    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    test_acc.append(100. * correct / len(test_loader.dataset))