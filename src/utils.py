import os

import torch
import torchvision

import torchvision.utils as vutils
from torch.utils.data import DataLoader

from src.attack import Attack
from src.custom_dataset import CustomDataset
from src.model import ConvNet
from src.train import train_loop, test_loop, adversarial_test_loop

import math
from scipy import stats


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))


def load_test_model(device, loss_fn, test_dataloader):
    # Загрузка и тестирование сохранённой модели
    loaded_model = ConvNet().to(device)
    load_model(loaded_model, path='../saved_models/cnn.ckpt')

    test_loop(test_dataloader, loaded_model, loss_fn, device)


def save_attacked_images(dataloader, model, device, save_dir, data_path):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    all_paths = []
    all_labels = []
    attack = Attack(
        model=model,
        epsilon=0.3,
        steps=100,
        step_size=0.01,
        random_start=False
    )

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        attacked_images = attack.perturb(images, labels)

        for i in range(attacked_images.shape[0]):
            label_i = labels[i].item()
            img_tensor = attacked_images[i].cpu()

            filename = f'attacked_{batch_idx}_{i}.png'
            filepath = os.path.join(save_dir, filename)
            vutils.save_image(img_tensor, filepath)

            all_paths.append(filepath)
            all_labels.append(label_i)

        torch.save({
            'paths' : all_paths,
            'labels' : all_labels
        }, data_path)


def load_attacked_dataset(data_path, batch_size=64):
    data = torch.load(data_path)
    all_paths = data['paths']
    all_labels = data['labels']

    transform = torchvision.transforms.ToTensor()

    attacked_dataset = CustomDataset(
        paths=all_paths,
        labels=all_labels,
        transform=transform
    )

    attacked_dataloader = DataLoader(attacked_dataset, batch_size=batch_size, shuffle=False)

    return attacked_dataloader


def train_model(model, device, loss_fn, epochs, train_dataloader, test_dataloader):
    print("Using device:", device)

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=3e-4
    )

    # train model
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        train_loop(train_dataloader, model, loss_fn, optimizer, device)
        test_loop(test_dataloader, model, loss_fn, device)


def adversarial_train_model(model, device, loss_fn, epochs, attack, train_dataloader, test_dataloader):
    print("Start adversarial_training")

    model.to(device)
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=3e-4
    )

    # adversarial train model
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}')
        train_loop(train_dataloader, model, loss_fn, optimizer, device, attack)
        test_loop(test_dataloader, model, loss_fn, device)


def confidence_interval(accuracy, count_examples, confidence=0.95):
    z = stats.norm.ppf((1 + confidence) / 2)
    margin = z * math.sqrt((accuracy * (1 - accuracy)) / count_examples)
    return accuracy - margin, accuracy + margin
