import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = torchvision.datasets.CIFAR10(download=True, root='./data', train=True, transform=transform)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

def get_cifar10() -> DataLoader:
    return train_loader