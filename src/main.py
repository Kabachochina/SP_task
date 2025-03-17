import torch
from torch import nn

from model import ConvNet
from src.attack import Attack
from src.data import get_dataloaders
from src.train import test_loop, adversarial_test_loop
from src.utils import save_attacked_images, adversarial_train_model, save_model, train_model, load_model, \
    confidence_interval
from utils import load_attacked_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Test adversarial_cnn_epoch_5_eps_0.3_steps_10_step_size_0.02 and cnn_epoch_5:")

protected_model = ConvNet().to(device)
natural_model = ConvNet().to(device)

load_model(
    model=protected_model,
    path='../saved_models/adversarial_cnn_epoch_5_eps_0.3_steps_10_step_size_0.02.ckpt'
)
load_model(
    model=natural_model,
    path='../saved_models/cnn_epoch_5.ckpt'
)

eps = 0.3

attack = Attack(
    model=protected_model,
    epsilon=eps,
    steps=10,
    step_size=0.02,
    random_start=False
)

print("Protected model:")

protected_adv_accuracy = adversarial_test_loop(
    dataloader=get_dataloaders()[1],
    model=protected_model,
    loss_fn=nn.CrossEntropyLoss(),
    device=device,
    attack=attack
)

protected_nat_accuracy = test_loop(
    dataloader=get_dataloaders()[1],
    model=protected_model,
    loss_fn=nn.CrossEntropyLoss(),
    device=device
)

protected_adv_ci = confidence_interval(
    accuracy=protected_adv_accuracy,
    count_examples=len(get_dataloaders()[1])
)

protected_nat_ci = confidence_interval(
    accuracy=protected_nat_accuracy,
    count_examples=len(get_dataloaders()[1])
)

print(f'adv CI: [{protected_adv_ci[0]:.3f}, {protected_adv_ci[1]:.3f}]')
print(f'nat CI: [{protected_nat_ci[0]:.3f}, {protected_nat_ci[1]:.3f}]')

print("Natural model:")

attack = Attack(
    model=natural_model,
    epsilon=eps,
    steps=10,
    step_size=0.02,
    random_start=False
)

natural_adv_accuracy = adversarial_test_loop(
    dataloader=get_dataloaders()[1],
    model=natural_model,
    loss_fn=nn.CrossEntropyLoss(),
    device=device,
    attack=attack
)

natural_nat_accuracy = test_loop(
    dataloader=get_dataloaders()[1],
    model=natural_model,
    loss_fn=nn.CrossEntropyLoss(),
    device=device
)

natural_adv_ci = confidence_interval(
    accuracy=natural_adv_accuracy,
    count_examples=len(get_dataloaders()[1])
)

natural_nat_ci = confidence_interval(
    accuracy=natural_nat_accuracy,
    count_examples=len(get_dataloaders()[1])
)

print(f'adv CI: [{natural_adv_ci[0]:.3f}, {natural_adv_ci[1]:.3f}]')
print(f'nat CI: [{natural_nat_ci[0]:.3f}, {natural_nat_ci[1]:.3f}]')


import matplotlib.pyplot as plt

models = ['С защитой', 'Без защиты']
accuracies = [protected_adv_accuracy, natural_adv_accuracy]
ci_lower = [protected_adv_accuracy - protected_adv_ci[0], natural_adv_accuracy - natural_adv_ci[0]]
ci_upper = [protected_adv_ci[1] - protected_adv_accuracy, natural_adv_ci[1] - natural_adv_accuracy]

plt.figure(figsize=(8, 6))
plt.bar(models, accuracies, yerr=[ci_lower, ci_upper], capsize=10, color=['green', 'red'])
plt.ylabel('Accuracy')
plt.title('Сравнение моделей с 95% ДИ')
plt.ylim(0, 1.1)
plt.show()
