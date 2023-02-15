import torchvision
import torch
from typing import Tuple, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2

class Cifar10Net(torchvision.datasets.CIFAR10):

    def __init__(self, root='~/data/cifar10', train=True, download=True, transform=None):
        super().__init__(root=root, 
                         train=train,
                         download=download,
                         transform=transform)
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        
        return image, label

def get_train_transforms():
    train_transforms = A.Compose([
        A.RandomCrop(height=32, width=32),
        A.Cutout(num_holes=1, max_h_size=16, max_w_size=16, p=1),
        A.Normalize(
                mean = (0.4914, 0.4822, 0.4465),
                std = (0.2470, 0.2435, 0.2616),
                p =1.0
            ),
        ToTensorV2()
    ])
    return train_transforms

def get_device():
    cuda = torch.cuda.is_available()
    return 'cuda' if cuda else 'cpu'

def get_test_transforms():
    test_transforms = A.Compose([

        A.Normalize(
            mean = (0.4914, 0.4822, 0.4465),
            std = (0.2470, 0.2435, 0.2616),
            p =1.0
        ),
        ToTensorV2()
    ])
    return test_transforms

def get_dataloader_args(batch_size):
    cuda = get_device()
    dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
    return dataloader_args

def load_data(train=True, dataloader_args=dict(shuffle=True, batch_size=64),
              transforms=None):
    dataset = Cifar10Net(root='./data', train=train, download=True, transform=transforms)
    return torch.utils.data.DataLoader(dataset, **dataloader_args)