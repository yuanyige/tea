import torch
from torchvision import datasets, transforms

def load_dataset(dataset, data_path, batch_size):
    if dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)) 
        ])
        train_dataset = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
