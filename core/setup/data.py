import os
import torch
import random
import torchvision.transforms as T
from torchvision import datasets, transforms

def set_transform(dataset):
    if dataset.lower() == 'cifar10' or dataset.lower() == 'cifar100':
        transform_train = T.Compose([ 
            T.Resize(32), 
            T.RandomCrop(32, padding=4), 
            T.RandomHorizontalFlip(),
            T.ToTensor()])
        transform_test = T.Compose([T.Resize(32), T.ToTensor()])
    elif  dataset.lower() == 'mnist':
        transform_train = transforms.Compose([
            T.ToTensor(28),
            T.Normalize((0.1307,), (0.3081,)) 
        ])
        transform_test = T.Compose([T.Resize(28), T.ToTensor()])
    else:
        raise
    return transform_train, transform_test


def load_data(root, dataset, batch_size, if_shuffle, logger=None):
    train_transforms, test_transforms = set_transform(dataset)
    if dataset.lower() == 'cifar10':
        logger.info("using cifar10..")
        train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=train_transforms)
        test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=test_transforms)
    elif dataset.lower() == 'cifar100':
        logger.info("using cifar100..")
        train_dataset = datasets.CIFAR100(root=root, train=True, download=True, transform=train_transforms)
        test_dataset = datasets.CIFAR100(root=root, train=False, download=True, transform=test_transforms)
    elif dataset.lower() == 'mnist':
        logger.info("using mnist..")
        train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=train_transforms)
        test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=test_transforms)
    elif dataset.lower() == 'tin200':
        logger.info("using tin200..")
        train_dataset = datasets.ImageFolder(os.path.join(root, 'tiny-imagenet-200', 'train'), transform=train_transforms)
        test_dataset = datasets.ImageFolder(os.path.join(root, 'tiny-imagenet-200', 'val'), transform=test_transforms)
    elif 'pacs' in dataset.lower():
        logger.info("using pacs..")
        train_dataset = datasets.ImageFolder(os.path.join(root, 'pacs', dataset.split('-')[1]), transform=train_transforms)
        test_dataset = None
    else:
        raise
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,  num_workers=4, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,  num_workers=4, shuffle=if_shuffle)
    return train_dataset, test_dataset, train_loader, test_loader

# class SelectedRotateCIFAR10(datasets.CIFAR10):
#     def __init__(
#         self,
#         root: str,
#         train: bool = True,
#         transform: Optional[Callable] = None,
#         target_transform: Optional[Callable] = None,
#         download: bool = False,
#         original=True, rotation=True, rotation_transform=None
#     ):
#         super().__init__(root, train, transform, target_transform, download)

#     def __getitem__(self, index):

#         img_input = self.data[index]
#         target = self.targets[index]

#         if self.transform is not None:
#             img_input = Image.fromarray(img_input)
#             img = self.transform(img_input)
#         else:
#             img = img_input

#         results = []
#         results.append(img)
#         results.append(target)
#         return results

#     def set_dataset_size(self, subset_size):
#         num_train = len(self.targets)
#         indices = list(range(num_train))
#         random.shuffle(indices)
#         self.data = [self.data[i] for i in indices[:subset_size]]
#         self.targets = [self.targets[i] for i in indices[:subset_size]]
#         return len(self.targets)