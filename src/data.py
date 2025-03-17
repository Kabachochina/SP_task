from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


def get_dataloaders(batch_size=64):
    training_data = datasets.MNIST(
        root="../data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.MNIST(
        root="../data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    return train_dataloader, test_dataloader