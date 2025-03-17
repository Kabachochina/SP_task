from torch.utils.data import Dataset
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        image = Image.open(img_path).convert('L')

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label
