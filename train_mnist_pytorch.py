
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pickle
import os

torch.manual_seed(42)
np.random.seed(42)

LAYER_COUNTS = [2, 4, 6, 8]
HIDDEN_SIZE = 400
N_EPOCHS = 30
LR = 1e-3
BATCH_SIZE = 256
TARGET_ACC = 0.982
SAVE_DIR = './saved_models'
os.makedirs(SAVE_DIR, exist_ok=True)


class MLP(nn.Module):
    # Simple feedforward network with ReLU activations.
    def __init__(self, n_hidden, hidden_size=400):
        super().__init__()
        layers = []
        in_sz = 784
        for _ in range(n_hidden):
            layers += [nn.Linear(in_sz, hidden_size), nn.ReLU()]
            in_sz = hidden_size
        layers.append(nn.Linear(in_sz, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def get_weights_biases(self):
        # Pull out weights and biases as numpy arrays for later use,Transpose so shape is (n_in, n_out) which is easier to work with.
        weights, biases = [], []
        for m in self.net:
            if isinstance(m, nn.Linear):
                weights.append(m.weight.detach().cpu().numpy().T)
                biases.append(m.bias.detach().cpu().numpy())
        return weights, biases


def get_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)) # flatten 28x28 to 784
    ])
    train_ds = torchvision.datasets.MNIST('./data', train=True,
                                           download=True, transform=transform)
    test_ds = torchvision.datasets.MNIST('./data', train=False,
                                           download=True, transform=transform)
    return (DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
            DataLoader(test_ds, batch_size=1000, shuffle=False))


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    correct, total, loss_sum = 0, 0, 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        total += len(y)
    return loss_sum / total, correct / total


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            correct += (model(X).argmax(1) == y).sum().item()
            total += len(y)
    return correct / total


def train(n_hidden):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining {n_hidden}-layer network on {device}")

    model = MLP(n_hidden).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    train_loader, test_loader = get_loaders()

    for epoch in range(1, N_EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader,
                                           optimizer, criterion, device)
        te_acc = evaluate(model, test_loader, device)

        if epoch % 5 == 0 or epoch == 1 or te_acc >= TARGET_ACC:
            print(f" Epoch {epoch:2d} | loss {tr_loss:.3f} | "
                  f"train {tr_acc*100:.1f}% | test {te_acc*100:.1f}%")

        if te_acc >= TARGET_ACC:
            print(f" Reached {te_acc*100:.2f}%, stopping early.")
            break

    weights, biases = model.get_weights_biases()
    path = os.path.join(SAVE_DIR, f'model_L{n_hidden}.pkl')
    with open(path, 'wb') as f:
        pickle.dump({'weights': weights, 'biases': biases,
                     'n_hidden': n_hidden, 'final_acc': te_acc}, f)
    print(f" Saved to {path}")
    return te_acc


if __name__ == '__main__':
    results = {}
    for L in LAYER_COUNTS:
        results[L] = train(L)

    print("\n--- Final Results ---")
    for L, acc in results.items():
        print(f" {L} hidden layers: {acc*100:.2f}%")
    print("\nDone. Run convert_all_models.py next.")