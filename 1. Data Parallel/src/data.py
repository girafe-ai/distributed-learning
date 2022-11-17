import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


def get_dataloader():
    dataset = MNIST('./mnist', transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))
    return DataLoader(dataset, batch_size=16)
