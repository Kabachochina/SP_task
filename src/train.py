import torch
from torch import nn
from torch.utils.data import DataLoader
from src.attack import Attack
from typing import Callable


def train_loop(
        dataloader : DataLoader,
        model : nn.Module,
        loss_fn : Callable,
        optimizer : torch.optim,
        device : torch.device,
        attack : Attack = None
):
    model.train()
    for batch, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        inputs = images
        if attack is not None:
            adv_images = attack.perturb(images, labels)
            inputs = adv_images

        pred = model(inputs)
        loss = loss_fn(pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            print(f"loss: {loss.item():>7f}  [{batch * len(inputs):>5d}/{len(dataloader.dataset):>5d}]")

def test_loop(
        dataloader : DataLoader,
        model : nn.Module,
        loss_fn : Callable,
        device : torch.device
) -> float:
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for (images, labels) in dataloader:
            images, labels = images.to(device), labels.to(device)

            pred = model(images)
            test_loss += loss_fn(pred, labels).item() # can be output for check loss
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()

    accuracy = correct / len(dataloader.dataset)
    print(f"Test Accuracy: {(100 * accuracy):.2f}%")
    return accuracy

def adversarial_test_loop(
        dataloader : DataLoader,
        model : nn.Module,
        loss_fn : Callable,
        device : torch.device,
        attack : Attack
) -> float:
    model.eval()
    test_loss = 0
    correct = 0
    model.requires_grad_(False)

    for (images, labels) in dataloader:
        images, labels = images.to(device), labels.to(device)

        inputs = attack.perturb(images, labels)

        pred = model(inputs)

        test_loss += loss_fn(pred, labels).item()
        correct += (pred.argmax(1) == labels).type(torch.float).sum().item()

    accuracy = correct / len(dataloader.dataset)
    print(f"Adversarial test Accuracy: {(100 * accuracy):.2f}%")
    return accuracy